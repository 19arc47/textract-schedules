# textract-schedules

RUN example: python3 textractSchedule.py --folder FOLDER_NAME --output OUTPUT_FILE_NAME.csv  --s3-bucket schedules-arconex-use1 --rules "('window' or 'fenestration') and 'schedule'"

This will run the script for all files in the target folder


extract labels:
Python script that leverages AWS Textract to extract text labels from PDF files, particularly focusing on lighting plans. The script processes each page of the PDF by splitting it into image patches, sending each patch to Textract for analysis, and then recombining the results to provide both annotated images and a CSV file listing all extracted text and their corresponding bounding boxes.
Features
PDF to Image Conversion: Converts each page of a PDF into an image for further processing.
Image Splitting: Splits images into smaller patches with configurable sizes and overlap to ensure complete text extraction, even for dense or small text areas.
AWS Textract Integration: Sends image patches to AWS Textract for text detection, supporting complex layouts such as tables and forms.
Bounding Box Annotation: Draws bounding boxes around detected text and saves annotated images for visual verification.
CSV Output: Generates a CSV file containing the bounding boxes and the corresponding extracted text, useful for further data analysis or integration into other systems.
Robust Logging: Provides detailed logging throughout the processing pipeline, enabling easy debugging and monitoring.

Usage
Prerequisites
Python 3.x
AWS SDK for Python (Boto3)
AWS Textract enabled and properly configured
Required Python packages (see requirements.txt)

Use: python script.py --s3-bucket your-bucket-name --file your-pdf-file.pdf --patch-size 1000 --overlap 100 --output results.csv
Parameters:
--s3-bucket: The name of the S3 bucket where the image patches will be uploaded.
--file: The path to the PDF file to be processed.
--patch-size: The size of the image patches (in pixels). Default is 1000.
--overlap: The overlap between image patches (in pixels). Default is 100.
--output: The name of the CSV file where the extracted results will be saved. Default is output.csv.
--log: The name of the log file. Default is process.log.

Example Output
Annotated images with bounding boxes around detected text.
A CSV file containing the bounding boxes and the extracted text.
