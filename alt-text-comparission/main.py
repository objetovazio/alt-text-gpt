import os
import logging
from dotenv import load_dotenv
from py_class.TextExtractor import TextExtractorV1
from py_class.ImageCaptionProcessor import ImageCaptionProcessor

logging.basicConfig(
    filename='./output.log',
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt='%Y-%m-%dT%H:%M:%S',
    level=logging.INFO
)

os.environ.get("REPLICATE_API_TOKEN")

def main():
     # Load .env file
    load_dotenv()

    images_dir_path = os.getenv('IMAGES_DIR_PATH')
    csv_path = os.getenv('ORIGINAL_CAPTIONS_PATH')
    FILTERED_CAPTIONS_PATH = os.getenv('FILTERED_CAPTIONS_PATH')
    generated_caption_path = os.getenv('GENERATED_CAPTION_PATH')

    output_caption_similarity_path = os.getenv('TEXT_SIMILARITY_PATH')
    
    extractor = TextExtractorV1()
    try:
        icp = ImageCaptionProcessor(images_dir_path, csv_path, extractor)
        icp.extract_captions(FILTERED_CAPTIONS_PATH)
        icp.generate_photos_captions(generated_caption_path)
        icp.compare_captions(FILTERED_CAPTIONS_PATH, generated_caption_path, output_caption_similarity_path)
        
    except Exception as e:
        logging.error("Failed to process image captions.", exc_info=True)

if __name__ == "__main__":
    main()
