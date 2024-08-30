import boto3
import sys
import argparse
import os
import csv
import json
import logging
import time
from PIL import Image, ImageDraw, ImageFont
from pdf2image import convert_from_path
from botocore.exceptions import ClientError

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

    logger.info("get_textract_job_results")
    return pages

def upload_file_to_s3(file_path, bucket_name, s3_key):
    s3_client.upload_file(file_path, bucket_name, s3_key)
    logger.info(f"Uploaded {file_path} to s3://{bucket_name}/{s3_key}")

def split_image(image, patch_size, overlap):
    width, height = image.size
    patches = []

    step_size = patch_size - overlap

    for top in range(0, height, step_size):
        for left in range(0, width, step_size):
            right = min(left + patch_size, width)
            bottom = min(top + patch_size, height)
            patch = image.crop((left, top, right, bottom))
            patches.append((patch, (left, top, right, bottom)))
    logging.info(f"We have {len(patches)}")
    return patches

def recombine_results(patch_results, original_size):
    combined_boxes = []
    combined_texts = []

    for patch_text, patch_boxes, patch_coords in patch_results:
        left_offset, top_offset, right_offset, bottom_offset = patch_coords
        for box, text in zip(patch_boxes, patch_text):
            adjusted_box = {
                'Left': (box['Left'] * (right_offset - left_offset) + left_offset) / original_size[0],
                'Top': (box['Top'] * (bottom_offset - top_offset) + top_offset) / original_size[1],
                'Width': box['Width'] * (right_offset - left_offset) / original_size[0],
                'Height': box['Height'] * (bottom_offset - top_offset) / original_size[1]
            }
            combined_boxes.append(adjusted_box)
            combined_texts.append(text)

    return combined_boxes, combined_texts


def extract_text_and_boxes_from_responses(responses):
    block_dict = {}
    texts = []
    bounding_boxes = []

    for response in responses:
        for block in response.get('Blocks', []):
            block_dict[block['Id']] = block
            if block['BlockType'] == 'WORD':
                bounding_boxes.append(block['Geometry']['BoundingBox'])
                texts.append(block.get('Text', ''))

    return texts, bounding_boxes

def save_results_to_csv(results, output_file):
    with open(output_file, 'w', newline='') as csvfile:
        fieldnames = ['Bounding Box', 'Text']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for bounding_box, text in results:
            writer.writerow({
                'Bounding Box': json.dumps(bounding_box),
                'Text': text
            })
    logger.info(f"Results saved to {output_file}")

def draw_bounding_boxes(image, bounding_boxes, texts, output_image_file):
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.load_default()
    except IOError:
        font = None

    for box, text in zip(bounding_boxes, texts):
        left = int(box['Left'] * image.width)
        top = int(box['Top'] * image.height)
        width = int(box['Width'] * image.width)
        height = int(box['Height'] * image.height)
        right = left + width
        bottom = top + height

        # Draw the rectangle
        draw.rectangle([left, top, right, bottom], outline="red", width=2)

        # Draw the text
        if font:
            draw.text((left, top - 10), text, fill="blue", font=font)
        else:
            draw.text((left, top - 10), text, fill="blue")

    image.save(output_image_file)
    logger.info(f"Annotated image saved to {output_image_file}")


def process_pdf_file(pdf_file, s3_bucket, patch_size, overlap):
    images = convert_from_path(pdf_file)
    results = []

    for page_number, image in enumerate(images):
        logger.info(f"Processing page {page_number + 1} of {pdf_file}")
        original_size = image.size
        patches = split_image(image, patch_size, overlap)
        patch_results = []

        for i, (patch, coords) in enumerate(patches):
            patch_file_name = f"{os.path.splitext(os.path.basename(pdf_file))[0]}_page_{page_number + 1}_patch_{i + 1}.png"
            patch.save(patch_file_name)

            s3_document_name = os.path.basename(patch_file_name)
            upload_file_to_s3(patch_file_name, s3_bucket, s3_document_name)

            job_id = start_textract_text_detection_job(s3_bucket, s3_document_name)

            if not is_textract_job_complete(job_id):
                logger.error(f"Job {job_id} did not complete successfully.")
                continue

            responses = get_textract_job_results(job_id)
            texts, bounding_boxes = extract_text_and_boxes_from_responses(responses)
            patch_results.append((texts, bounding_boxes, coords))

            # Cleanup patch file
            os.remove(patch_file_name)

        combined_boxes, combined_texts = recombine_results(patch_results, original_size)
        results.extend(zip(combined_boxes, combined_texts))

        # Draw bounding boxes and save the image
        output_image_file = f"{os.path.splitext(pdf_file)[0]}_page_{page_number + 1}_annotated.png"
        draw_bounding_boxes(image, combined_boxes, combined_texts, output_image_file)

    return results

def main():
    parser = argparse.ArgumentParser(description="Extract text labels from lighting plans in PDFs using Textract and save results to a CSV file")
    parser.add_argument("--s3-bucket", required=True, help="S3 bucket name to store the images")
    parser.add_argument("--file", required=True, help="PDF file to process")
    parser.add_argument("--patch-size", type=int, default=1000, help="Size of image patches (in pixels)")
    parser.add_argument("--overlap", type=int, default=100, help="Overlap between patches (in pixels)")
    parser.add_argument("--output", default="output.csv", help="Output CSV file to save the results")
    parser.add_argument("--log", default="process.log", help="Log file to save the log details")

    args = parser.parse_args()

    file_handler = logging.FileHandler(args.log)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)

    if not os.path.isfile(args.file):
        logger.error(f"Error: The file {args.file} does not exist.")
        sys.exit(1)

    results = process_pdf_file(args.file, args.s3_bucket, args.patch_size, args.overlap)

    save_results_to_csv(results, args.output)

if __name__ == "__main__":
    main()
