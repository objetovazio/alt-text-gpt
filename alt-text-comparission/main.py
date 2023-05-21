import os
import logging
from dotenv import load_dotenv
from ImageCaptionProcessor import ImageCaptionProcessor

logging.basicConfig(
    filename='./alt-text-comparission/output.log',
    format="[Log.%(levelname)s][%(asctime)s]: %(message)s",
    level=logging.INFO
)

os.environ.get("REPLICATE_API_TOKEN")

def main():
     # Load .env file
    load_dotenv()

    images_dir_path = os.getenv('IMAGES_DIR_PATH')
    csv_path = os.getenv('CAPTIONS_CSV_PATH')
    filtered_caption_path = os.getenv('FILTERED_CAPTION_PATH')
    generated_caption_path = os.getenv('GENERATED_CAPTION_PATH')
    
    try:
        icp = ImageCaptionProcessor(images_dir_path, csv_path)
        icp.extract_captions(filtered_caption_path)
        icp.generate_photos_captions(generated_caption_path)
        
    except Exception as e:
        logging.error("Failed to process image captions.", exc_info=True)

    

if __name__ == "__main__":
    main()
