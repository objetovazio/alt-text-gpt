from abc import ABC, abstractmethod

class AbstractTextExtractor(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def extract_text_from_image_path(self, image_path):
        pass

    @abstractmethod
    def extract_text_from_image_url(self, image_url):
        pass