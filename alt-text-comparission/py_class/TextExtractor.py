from abc import ABC, abstractmethod
import logging
import replicate

class AbstractTextExtractor(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def extract_text_from_image_path(self, image_path):
        pass

    @abstractmethod
    def extract_text_from_image_url(self, image_url):
        pass

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
        pass
    # end - extract_text_from_image_url()
    