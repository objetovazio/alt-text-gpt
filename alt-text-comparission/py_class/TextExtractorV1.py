import logging
import replicate
import os
from .AbstractTextExtractor import AbstractTextExtractor
from urllib.parse import urlparse
import pandas as pd

class TextExtractorV1(AbstractTextExtractor):
    def __init__(self, get_from_file=False):
        super().__init__()
        self.get_from_file = get_from_file
        pass

    def extract_text_from_image_path(self, image_path):
        logging.debug(f"Extracting caption from {image_path}.")

        try:
            # Get value from file
            if self.get_from_file:
                caption = self.get_caption_from_file(image_path)

                if(caption):
                    return caption, []

            with open(image_path, "rb") as image:
                model = "salesforce/blip:2e1dddc8621f72155f24cf2e0adbde548458d3cab9f00c0139eea840d0ac4746"
                output_text = replicate.run(
                    model,
                    input={"image": image},
                )
            if ": " in output_text:
                output_text = output_text.split(": ")[1]
            else:
                logging.warning(f"Unexpected output format for {image_path}. Returning full output_text.")

            logging.info(f"Caption successfully extracted from {image_path}.")

        except Exception as e:
            logging.error(f"Failed to extract caption from {image_path} due to {str(e)}")
            return None, []

        return output_text, None

    def extract_text_from_image_url(self, image_url):
        logging.debug(f"Extracting caption from {image_url}.")

        try:
            # Get value from file
            if self.get_from_file:
                caption = self.get_caption_from_file(image_url)

                if caption:
                    return caption

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
    def get_caption_from_file(self, image_path):
        if image_path.startswith('http://') or image_path.startswith('https://'):
            parsed_url = urlparse(image_path)
            image_name = os.path.basename(parsed_url.path)
        else:
            image_name = os.path.basename(image_path)

        # Load the CSV file into a DataFrame
        gambiarra_dir = os.getenv('GAMBIARRA_DIR')

        df = pd.read_csv(gambiarra_dir)

        # Check if the caption exists in the DataFrame
        caption = df[df['image'] == image_name]['caption'].values

        if caption and len(caption) > 0:
            return caption[0]

        return None

