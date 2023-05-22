import logging
import replicate
import os;
from .AbstractTextExtractor import AbstractTextExtractor
from urllib.parse import urlparse
import pandas as pd

class TextExtractorV1(AbstractTextExtractor):
    def __init__(self):
        super().__init__()
        pass

    def extract_text_from_image_path(self, image_path):
        logging.debug(f"Extracting caption from {image_path}.")

        try:
            # Check if caption exists in the CSV file
            existing_caption = self.caption_exists_in_csv(image_path)

            if existing_caption:
                logging.info(f"Caption already exists in CSV for {image_path}. Returning existing caption.")
                return existing_caption

            with open(image_path, "rb") as image:
                output_text = replicate.run(
                    model,
                    input={"image": image},
                )
                logging.info(f"Caption successfully extracted from {image_path}.")
        except Exception as e:
            logging.error(f"Failed to extract caption from {image_path} due to {str(e)}")
            return None

        return output_text

    def extract_text_from_image_url(self, image_url):
        logging.debug(f"Extracting caption from {image_url}.")

        try:
            # Check if caption exists in the CSV file
            existing_caption = self.caption_exists_in_csv(image_url)

            if existing_caption:
                logging.info(f"Caption already exists in CSV for {image_url}. Returning existing caption.")
                return existing_caption

            model = "salesforce/blip:2e1dddc8621f72155f24cf2e0adbde548458d3cab9f00c0139eea840d0ac4746"

            output_text = replicate.run(
                model,
                input={"image": image_url},
            )

            if ": " in output_text:
                output_text = output_text.split(": ")[1]
            else:
                logging.warning(f"Unexpected output format for {image_url}. Returning full output_text.")

            logging.info(f"Caption successfully extracted from {image_url}.")

        except Exception as e:
            logging.error(f"Failed to extract caption from {image_url} due to {str(e)}")
            return None

        return output_text

    """ GAMBIARRA """
    def caption_exists_in_csv(self, image_path):
        if image_path.startswith('http://') or image_path.startswith('https://'):
            parsed_url = urlparse(image_path)
            image_name = os.path.basename(parsed_url.path)
        else:
            image_name = os.path.basename(image_path)

        # Load the CSV file into a DataFrame
        df = pd.read_csv(os.getenv('GENERATED_CAPTION_PATH'))

        # Check if the caption exists in the DataFrame
        caption = df[df['image'] == image_name]['caption'].values
        if caption:
            return caption[0]

        return None






