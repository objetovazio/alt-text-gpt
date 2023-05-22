import os
import logging
from dotenv import load_dotenv
from py_class.TextExtractorV1 import TextExtractorV1
from py_class.TextExtractorV2 import TextExtractorV2
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

    # Directory for the Photos Folder :)
    images_dir_path = os.getenv('IMAGES_DIR_PATH')

    # Path for the Kaggle - Eye For Blind Dataset Captions CSV File
    csv_path = os.getenv('ORIGINAL_CAPTIONS_PATH')

    # Path Path for the captions files with ONLY images that are inside images_dir_path. (Around 50)
    filtered_captions_path = os.getenv('FILTERED_CAPTIONS_PATH')

    # Generated captions path, which I calls a TextExtractor to generate the file, based of filtered_captions_path file.
    generated_caption_path = os.getenv('GENERATED_CAPTION_PATH')
    generated_caption_path = "/mnt/c/github/alt-text-gpt/alt-text-comparission/files/generated-chatgpt-captions.csv"

    # Output file for analisys of the caption similarity
    output_caption_similarity_path = os.getenv('TEXT_SIMILARITY_V1_PATH')

    # Output file for analisys of the caption similarity
    output_caption_similarity_path = os.getenv('TEXT_SIMILARITY_V2_PATH')
    
    # extractor = TextExtractorV1()
    extractor_v2 = TextExtractorV2()

    try:
        # image_url = "https://thumbs.dreamstime.com/b/woman-playing-basketball-5338648.jpg"
        # text_from_image = extractor_v2.extract_text_from_image_url(image_url)
        # logging.info(f"#PraTodosVerem: {text_from_image}")

        icp = ImageCaptionProcessor(images_dir_path, csv_path, extractor_v2)
        icp.extract_captions(filtered_captions_path)
        icp.generate_photos_captions(generated_caption_path)
        icp.compare_captions(filtered_captions_path, generated_caption_path, output_caption_similarity_path)

    except Exception as e:
        logging.error("Failed to process image captions.", exc_info=True)

if __name__ == "__main__":
    main()
