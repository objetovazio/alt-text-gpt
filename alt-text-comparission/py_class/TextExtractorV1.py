import logging
import replicate
from .AbstractTextExtractor import AbstractTextExtractor

class TextExtractorV1(AbstractTextExtractor):
    def __init__(self):
        super().__init__()
        pass
    
    def extract_text_from_image_path(self, image_path):
        logging.debug(f"Extracting caption from {image_path}.")
        
        try:
            with open(image_path, "rb") as image:
                model = "salesforce/blip:2e1dddc8621f72155f24cf2e0adbde548458d3cab9f00c0139eea840d0ac4746"
                
                output_text = replicate.run(
                    model,
                    input={"image": image},
                )
                logging.info(f"Caption successfully extracted from {image_path}.")
        except Exception as e:
            logging.error(f"Failed to extract caption from {image_path} due to {str(e)}")
            return None

        return output_text
    # end - extract_text_from_image_path()
    
    def extract_text_from_image_url(self, image_url):
        logging.debug(f"Extracting caption from {image_url}.")
        
        try:
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
    # end - extract_text_from_image_url()
