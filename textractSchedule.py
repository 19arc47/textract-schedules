import boto3
import sys
import argparse
import os
import json
import logging
import csv
import ast
import cv2
import time
from PIL import Image
from botocore.exceptions import ClientError
from PyPDF2 import PdfReader, PdfWriter
from pdf2image import convert_from_path
import subprocess

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()

# Initialize Textract and S3 clients
textract = boto3.client('textract')
s3_client = boto3.client('s3')

def start_textract_text_detection_job(s3_bucket_name, s3_document_name):
    logger.info(f"Starting Textract text detection job for {s3_document_name}")
    response = textract.start_document_analysis(
        DocumentLocation={
            'S3Object': {
                'Bucket': s3_bucket_name,
                'Name': s3_document_name
            }
        },
        FeatureTypes=['TABLES', 'FORMS']
    )
    return response['JobId']

def is_textract_job_complete(job_id, timeout=120):
    start_time = time.time()
    while True:
        response = textract.get_document_analysis(JobId=job_id)
        status = response['JobStatus']
        if status in ['SUCCEEDED', 'FAILED']:
            if status == 'FAILED':
                logger.error(f"Job {job_id} failed.")
            return status == 'SUCCEEDED'
        elif time.time() - start_time > timeout:
            logger.warning(f"Job {job_id} timed out after {timeout} seconds. Considering it as terminated.")
            return False

        logger.info(f"Job status: {status}. Waiting for completion...")
        time.sleep(5)

def get_textract_job_results(job_id):
    pages = []
    next_token = None
    while True:
        if next_token:
            response = textract.get_document_analysis(JobId=job_id, NextToken=next_token)
        else:
            response = textract.get_document_analysis(JobId=job_id)
        pages.append(response)
        next_token = response.get('NextToken')
        if not next_token:
            break

    #with open("out.txt", 'w') as f:
    #    json.dump(pages, f, indent=4)
    logger.info("get_textract_job_results")
    return pages

def upload_file_to_s3(file_path, bucket_name, s3_key):
    s3_client.upload_file(file_path, bucket_name, s3_key)
    logger.info(f"Uploaded {file_path} to s3://{bucket_name}/{s3_key}")

"""
def optimize_jpeg(image_file):
    optimized_file = f"{os.path.splitext(image_file)[0]}_optimized.jpeg"
    command = [
        'convert',  # Ensure ImageMagick is installed
        image_file,
        #'-resize', '99%',
        #'-quality', '95',
        optimized_file
    ]
    
    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as e:
        logger.error(f"Error optimizing {image_file}: {e}")
    
    return optimized_file
"""

def parse_rule(rule):
    return ast.parse(rule, mode='eval')

def eval_logic(node, title):
    operators_map = {
        ast.And: all,
        ast.Or: any
    }
    if isinstance(node, ast.BoolOp):
        op = operators_map[type(node.op)]
        return op(eval_logic(value, title) for value in node.values)
    elif isinstance(node, ast.Compare):
        left = eval_logic(node.left, title)
        for op, comparator in zip(node.ops, node.comparators):
            if isinstance(op, ast.Eq) and isinstance(comparator, ast.Str):
                if comparator.s.lower() in title.lower():
                    return True
        return False
    elif isinstance(node, ast.Str):
        return node.s.lower() in title.lower()
    else:
        raise ValueError(f"Unsupported operation: {ast.dump(node)}")

def match_rule(title, tree):
    try:
        return eval_logic(tree.body, title)
    except Exception as e:
        logger.error(f"Error matching rule: {e}")
        return False

def extract_table_titles_bounding_boxes_and_first_column(responses, rules, image_file):
    tables = []

    # Convert PDF page to image (if applicable)
    if image_file.endswith('.pdf'):
        images = convert_from_path(image_file)
        original_image = images[0]  # Assume first page; adjust if needed
    else:
        original_image = Image.open(image_file)

    img_width, img_height = original_image.size

    base_name = os.path.splitext(os.path.basename(image_file))[0]
    subfolder_path = os.path.join(os.path.dirname(image_file), base_name)

    if not os.path.exists(subfolder_path):
        os.makedirs(subfolder_path)

    block_dict = {}
    for response in responses:
        for block in response.get('Blocks', []):
            block_dict[block['Id']] = block

    for block in block_dict.values():
        if block['BlockType'] == 'TABLE':
            bounding_box = block['Geometry']['BoundingBox']
            left = int(bounding_box['Left'] * img_width)
            top = int(bounding_box['Top'] * img_height)
            width = int(bounding_box['Width'] * img_width)
            height = int(bounding_box['Height'] * img_height)
            right = left + width
            bottom = top + height
            bounding_box_str = f"{{'Left': {bounding_box['Left']}, 'Top': {bounding_box['Top']}, 'Width': {bounding_box['Width']}, 'Height': {bounding_box['Height']}}}"

            title = None
            for relationship in block.get('Relationships', []):
                if relationship['Type'] == 'TABLE_TITLE':
                    title_block_id = relationship['Ids'][0]
                    title_block = block_dict.get(title_block_id, None)
                    if title_block:
                        title = extract_text_from_relationships(title_block, block_dict)
                    break

            if not title:
                title = "No title found"

            first_column_text = extract_first_column_text(block, block_dict)

            if all(match_rule(title.lower(), tree) for tree in rules):
                tables.append({
                    'title': title,
                    'bounding_box': bounding_box_str,
                    'first_column': first_column_text
                })

                # Extract and save the table image
                table_image = original_image.crop((left, top, right, bottom))
                safe_title = title.replace(' ', '_').replace('/', '_')
                table_image.save(os.path.join(subfolder_path, f"{safe_title}_table.png"))
                
                logger.info(f"Table Title found: {title}")
    
    return tables

def extract_text_from_relationships(block, block_dict):
    text = []
    for relationship in block.get('Relationships', []):
        if relationship['Type'] == 'CHILD':
            for child_id in relationship['Ids']:
                word_block = block_dict.get(child_id, None)
                if word_block and word_block.get('BlockType') == 'WORD':
                    text.append(word_block.get('Text', ''))
    return ' '.join(text)

def zextract_first_column_text(table_block, block_dict):
    first_column_text = []
    for relationship in table_block.get('Relationships', []):
        if relationship['Type'] == 'CHILD':
            for cell_id in relationship['Ids']:
                cell_block = block_dict.get(cell_id, None)
                if cell_block and cell_block.get('BlockType') == 'CELL' and cell_block.get('ColumnIndex') == 1:
                    cell_text = extract_text_from_relationships(cell_block, block_dict)
                    #if any(keyword in cell_text.lower() for keyword in ["tag", "mark", "number", "#"]):
                    first_column_text.append(cell_text)
    return first_column_text  # Now returns a list of filtered text entries

def extract_first_column_text(table_block, block_dict): # DOOR VERSION
    keywords = ["tag", "mark", "number", "#"]  # Define the keywords to search for
    first_column_index = None

    # Step 1: Find the index of the first column that contains any of the keywords
    for relationship in table_block.get('Relationships', []):
        if relationship['Type'] == 'CHILD':
            for cell_id in relationship['Ids']:
                cell_block = block_dict.get(cell_id, None)
                if cell_block and cell_block.get('BlockType') == 'CELL':
                    cell_text = extract_text_from_relationships(cell_block, block_dict)
                    if any(keyword in cell_text.lower() for keyword in keywords):
                        first_column_index = cell_block['ColumnIndex']
                        break
            if first_column_index is not None:
                break

    # Step 2: Extract text from the identified column
    if first_column_index is not None:
        first_column_text = []
        for relationship in table_block.get('Relationships', []):
            if relationship['Type'] == 'CHILD':
                for cell_id in relationship['Ids']:
                    cell_block = block_dict.get(cell_id, None)
                    if cell_block and cell_block.get('BlockType') == 'CELL' and cell_block.get('ColumnIndex') == first_column_index:
                        cell_text = extract_text_from_relationships(cell_block, block_dict)
                        first_column_text.append(cell_text)
        return first_column_text
    else:
        return []  # No matching column found


def save_results_to_csv(results, output_file):
    with open(output_file, 'w', newline='') as csvfile:
        fieldnames = ['Image File', 'Table Title', 'Bounding Boxes', 'First Column']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            writer.writerow({
                'Image File': result['image_file'],
                'Table Title': ', '.join(result['tables']),
                'Bounding Boxes': '[' + ', '.join(result['bounding_boxes']) + ']',
                'First Column': json.dumps(result['first_columns'])  # Save as JSON to preserve list structure
            })
    logger.info(f"Results saved to {output_file}")  

def are_all_jobs_complete(job_ids):
    incomplete_jobs = job_ids.copy()
    
    while incomplete_jobs:
        for job_id in incomplete_jobs[:]:
            if is_textract_job_complete(job_id):
                logger.info(f"Job {job_id} completed successfully.")
                incomplete_jobs.remove(job_id)
            else:
                logger.info(f"Job {job_id} is still processing.")
        
        if incomplete_jobs:
            logger.info(f"Waiting for {len(incomplete_jobs)} job(s) to complete...")
            time.sleep(5)
    
    logger.info("All Textract jobs have been completed.")

def convert_image_to_pdf(image_file):
    pdf_file = f"{os.path.splitext(image_file)[0]}.pdf"
    with Image.open(image_file) as img:
        img.convert('RGB').save(pdf_file, format='PDF')
    logger.info(f"Converted {image_file} to {pdf_file}")
    return pdf_file

def main():
    parser = argparse.ArgumentParser(description="Extract tables from a series of images or a folder of images and save the results to a CSV file")
    parser.add_argument("--s3-bucket", required=True, help="S3 bucket name to store the images")
    parser.add_argument("--prefix", help="Prefix of the image file names (e.g., 'mech') for batch processing")
    parser.add_argument("--file", help="Single image file to process")
    parser.add_argument("--folder", help="Folder containing JPEG files to process")
    parser.add_argument("--limit", type=int, default=1, help="Number of images to process in the series when using prefix")
    parser.add_argument("--output", default="output.csv", help="Output CSV file to save the results")
    parser.add_argument("--log", default="process.log", help="Log file to save the log details")
    parser.add_argument("--rules", nargs='+', help="Logical rules to filter table titles (e.g., \"(door AND schedule) OR (window AND schedule)\")")
    
    args = parser.parse_args()

    file_handler = logging.FileHandler(args.log)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)

    results = []
    job_ids = []

    # Determine which files to process
    if args.file:
        image_files = [args.file]
    elif args.prefix:
        image_files = [f"{args.prefix}-{i:03d}.jpeg" for i in range(1, args.limit + 1)]
    elif args.folder:
        image_files = [os.path.join(args.folder, f) for f in os.listdir(args.folder) if f.endswith('.jpeg') or f.endswith('.jpg') or f.endswith('.pdf')] 
    else:
        logger.error("You must specify either a --file, --prefix, or --folder.")
        sys.exit(1)

    for image_file in image_files:
        if not os.path.isfile(image_file):
            logger.error(f"Error: The file {image_file} does not exist.")
            continue
        #image_file = optimize_jpeg(image_file)

        # Convert files larger than 10MB to PDF format
        #if os.path.getsize(image_file) > 1 * 1024 * 1024:
            #logger.info(f"File {image_file} exceeds 10MB, converting to PDF format...")
            #image_file = convert_image_to_pdf(image_file)
            #image_file = fix_pdf_xref_table(image_file)
        
        s3_document_name = os.path.basename(image_file)
        upload_file_to_s3(image_file, args.s3_bucket, s3_document_name)

        job_id = start_textract_text_detection_job(args.s3_bucket, s3_document_name)
        job_ids.append(job_id)
    
    # Wait for all jobs to complete
    are_all_jobs_complete(job_ids)

    # Process results after all jobs are complete
    for image_file, job_id in zip(image_files, job_ids):
        logger.info(f"Working on {image_file}")
        responses = get_textract_job_results(job_id)
        
        # Extract table titles, bounding boxes, and first column text based on rules
        parsed_rules = [parse_rule(rule) for rule in args.rules or []]
        tables = extract_table_titles_bounding_boxes_and_first_column(responses, parsed_rules, image_file)
        logger.info("Got tables")
        if tables:
            logger.info(f"Total tables found: {len(tables)}")
            results.append({
                'image_file': image_file,
                'tables': [table['title'] for table in tables],
                'bounding_boxes': [table['bounding_box'] for table in tables],
                'first_columns': [table['first_column'] for table in tables]
            })
        else: 
            logger.info("No tables found.")

    # Save all the accumulated results to a CSV file
    save_results_to_csv(results, args.output)
    logger.info(f"Results saved to {args.output}")

if __name__ == "__main__":
    main()

