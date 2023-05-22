import os
import logging
import requests
import openai
from .TextExtractorV1 import TextExtractorV1

class TextExtractorV2(TextExtractorV1):
    def __init__(self):
        super().__init__()
        self.ep_url = os.getenv('EVERYPIXEL_API_URL')  # get the EveryPixel API URL from environment variables
        self.ep_username = os.getenv('EVERYPIXEL_USERNAME')  # get the EveryPixel username from environment variables
        self.ep_api_key = os.getenv('EVERYPIXEL_API_KEY')  # get the EveryPixel API key from environment variables
        self.open_api_key = os.getenv('OPEN_API_KEY')

        self.auth = (self.ep_username, self.ep_api_key)

    def extract_text_from_image_path(self, image_path):
        salesforce_caption = super().extract_text_from_image_path(image_path)
        
        language = "English"
        
        # Add new steps here
        keywords = self.get_keywords_from_image_path(image_path)
        if keywords is not None:
            prompt = f"""
            Oi, ChatGPT. Recebi a legenda '{salesforce_caption}' e palavras-chave  '{keywords}' para descrever uma imagem.

            Por favor, revise e melhore a legenda, usando as palavras-chave quando apropriado. Não invente informações. 

            Lembre-se de adaptar a frase para cada imagem específica, fornecendo informações verdadeiras e concisas, e utilizando sinônimos para palavras complexas, conforme necessário. 

            Lembre-se de ignorar informações redundantes. Caso falte algumas das informações, ignore o tópico.

            Para descrever, use a seguinte formula textual:

            1. Tipo de imagem: [Aponte se é fotografia, cartum, tirinha, ilustração. Breve descrição com até 4 palavras]
            2. Pessoa: [sexo] [etnia], [posição na imagem]
            3. Cabelo: [descrição do tipo e cor usando sinônimos]
            4. Roupa: [descrição do tipo e cor usando sinônimos]
            5. Objeto: [breve descrição com até 4 palavras]
            6. Uso do objeto: [breve descrição com até 4 palavras]
            7. Ambiente: [breve descrição com até 4 palavras]

            A legenda final deve ser clara, direta e acessível para pessoas com deficiência visual.
            """
            print(prompt)
            generated_text = self._call_chat_gpt_api(prompt)
            logging.info(f"Generated text: {generated_text}")
            return generated_text, keywords
    # end - extract_text_from_image_path()
        
    def extract_text_from_image_url(self, image_url):
        salesforce_caption = super().extract_text_from_image_url(image_url)

        language = "English"

        # Add new steps here
        keywords = self.get_keywords_from_image_url(image_url)
        if keywords is not None:
            prompt = f"""
            Oi, ChatGPT. Recebi a legenda '{salesforce_caption}' e palavras-chave  '{keywords}' para descrever uma imagem.

            Por favor, revise e melhore a legenda, usando as palavras-chave quando apropriado. Não invente informações. 

            Lembre-se de adaptar a frase para cada imagem específica, fornecendo informações verdadeiras e concisas, e utilizando sinônimos para palavras complexas, conforme necessário. 

            Lembre-se de ignorar informações redundantes. Caso falte algumas das informações, ignore o tópico.

            Para descrever, use a seguinte formula textual:

            1. Tipo de imagem: [Aponte se é fotografia, cartum, tirinha, ilustração. Breve descrição com até 4 palavras]
            2. Pessoa: [sexo] [etnia], [posição na imagem]
            3. Cabelo: [descrição do tipo e cor usando sinônimos]
            4. Roupa: [descrição do tipo e cor usando sinônimos]
            5. Objeto: [breve descrição com até 4 palavras]
            6. Uso do objeto: [breve descrição com até 4 palavras]
            7. Ambiente: [breve descrição com até 4 palavras]

            A legenda final deve ser clara, direta e acessível para pessoas com deficiência visual.
            """
            generated_text = self._call_chat_gpt_api(prompt)
            logging.info(f"Generated text: {generated_text}")
            return generated_text
    # end - extract_text_from_image_url()
    
    def get_keywords_from_image_path(self, image_path):
        logging.debug(f"Getting keywords from {image_path}.")
        try:
            with open(image_path, "rb") as image:
                data = {'data': image}
                params = {'num_keywords': 10}
                
                response = requests.post(
                    self.ep_url,
                    files=data,
                    params=params,
                    auth=self.auth
                )
                
            if response.status_code == 200:
                keywords = response.json().get('keywords', [])
                keyword_scores = [(keyword['keyword'], keyword['score']) for keyword in keywords]
                logging.info(f"Keywords successfully extracted from {image_path}: {keyword_scores}.")
                return keyword_scores
            else:
                logging.warning(f"Failed to get keywords from {image_path}.")
                return None
                
        except Exception as e:
            logging.error(f"Failed to extract keywords from {image_path} due to {str(e)}")
            return None
    # end - get_keywords_from_image_path()

    def get_keywords_from_image_url(self, image_url):
        logging.debug(f"Getting keywords from {image_url}.")
        try:
            params = {'url': image_url, 'num_keywords': 10}
            
            response = requests.get(
                self.ep_url,
                params=params,
                auth=self.auth
            )
            
            if response.status_code == 200:
                keywords = response.json().get('keywords', [])
                filtered_keywords = [(keyword['keyword'], keyword['score']) for keyword in keywords if keyword['score'] > 0.9]
                logging.info(f"Keywords successfully extracted from {image_url}: {filtered_keywords}.")
                return filtered_keywords
            else:
                logging.warning(f"Failed to get keywords from {image_url}.")
                return None
                
        except Exception as e:
            logging.error(f"Failed to extract keywords from {image_url} due to {str(e)}")
            return None
    # end - get_keywords_from_image_url()

    def _call_chat_gpt_api(self, prompt):
        try:
            openai.api_key = self.open_api_key

            response = openai.Completion.create(
                engine="text-davinci-003",  # Use the desired ChatGPT engine
                prompt=prompt,
                max_tokens=100,  # Adjust the maximum number of tokens as needed
                temperature=0.7,  # Adjust the temperature for text generation
                n=1,  # Generate a single response
                stop=None,  # Set stop condition if needed
            )

            generated_text = response.choices[0].text.strip()
            return generated_text
        except Exception as e:
            logging.error(f"Failed to call ChatGPT API due to {str(e)}")
            raise Exception(f"Failed to call ChatGPT API due to {str(e)}")
    