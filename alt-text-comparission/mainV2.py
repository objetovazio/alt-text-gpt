import os
import logging
from dotenv import load_dotenv
from py_class.TextExtractorV2 import TextExtractorV2
from py_class.ImageCaptionProcessor import ImageCaptionProcessor
import datetime

# Create a timestamp for the log file name
timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

# Configure logging to write to a new log file with the timestamp
log_file = f"./{timestamp}.log"

script_path = os.path.dirname(os.path.realpath(__file__))

# Configure logging to write to file
logging.basicConfig(
    filename=f'{script_path}/logs/{log_file}',
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
    level=logging.INFO,
    filemode='a'
)

# Create a console handler to display log messages on the console
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)

# Create a formatter for console output
console_formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

# Add the console handler to the root logger
root_logger = logging.getLogger()
root_logger.addHandler(console_handler)
console_handler.setFormatter(console_formatter)

os.environ.get("REPLICATE_API_TOKEN")

def main():
    # Load .env file
    load_dotenv()

    # Directory for the Photos Folder :) 
    images_dir_path = os.getenv('IMAGES_DIR_PATH')

    # Diretory for files folder
    files_path = os.getenv('FILES_PATH')

    # Path for the Kaggle - Eye For Blind Dataset Captions CSV File - Original File
    original_captions_path = f"{files_path}/{os.getenv('ORIGINAL_CAPTIONS_FILE')}"

    # Path for filtered captions - File with only captions from the local path
    filtered_captions_path = f"{files_path}/{os.getenv('FILTERED_CAPTIONS_FILE')}"

    # Generated captions path, which I calls a TextExtractor to generate the file, based of filtered_captions_path file.
    generated_caption_path = f"{files_path}/{os.getenv('GENERATED_CAPTION_CHATGPT_FILE')}"

    # Output file for analisys of the caption similarity
    filename = os.getenv('TEXT_SIMILARITY_V2_FILE').format(timestamp=timestamp)
    output_caption_similarity_path = f"{files_path}/{filename}"

    extractor = TextExtractorV2(True)

    try:
        icp = ImageCaptionProcessor(images_dir_path, original_captions_path, extractor, check_in_file=True)
        
        # Filtra legendas de imagens contidas na pasta images_dir_path para um novo arquivo menor.
        icp.extract_captions(filtered_captions_path)

        # Executa a geracao de legendas para as fotos contidas em images_dir_path.
        icp.generate_photos_captions(generated_caption_path, True)

        # Efetua comparação entre legendas geradas.
        icp.compare_captions(filtered_captions_path, generated_caption_path, output_caption_similarity_path)

    except Exception as e:
        logging.error("Failed to process image captions.", exc_info=True)

if __name__ == "__main__":
    main()
