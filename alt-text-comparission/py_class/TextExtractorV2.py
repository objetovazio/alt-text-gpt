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

    def _get_prompt(self, salesforce_caption, keywords, language="Inglês"):
        prompt = f"""
       Siga os passos OBRIGATORIAMENTE.
        Entrada:
            legenda:  {salesforce_caption}
            tags: {keywords}
            output_langue: {language}
        
        Saída:
            legenda_gerada: resultado da legenda gerada baseada nos inputs.

        Formato de saída:
            Output: {{legenda_gerada}}

        Vamos escrever uma legenda para descrever uma imagem seguindo alguns passos. 
            Irei fornecer 3 inputs: legenda, tags e idioma. Utilizaremos esses 3 inputs para reescrever a legenda. A descrição de cada input é:
                legenda: legenda Original da imagem em inglês;
                tags: Array de tuplas contendo dois valores:
                    - Palavra: String que representa algum elemento da imagem. 
                    - Score: Float que refere-se à avaliação ou estimativa de um atributo ou qualidade relacionado a imagem referente da legenda. Quanto mais próximo de 1, melhor a sua estimativa. Quanto mais próximo de zero, pior.
                output_langue: Idioma no qual você deve retornar a nova legenda.
                
            
            Siga os proximos passos:

                1. Leia a legenda e analise o seu contexto. Leia as tags e ordene-as pelo score do maior para menor.

                2. Itere em ordem sobre cada tag e analise se ela pode ser adicionada ao contexto da legenda para melhora-la. Para cada tag, dependedo de do contexto, utilize uma ou mais intruçoes a seguir para a reescrita da legenda na ordem a seguir:
                    - Tipo de imagem: [Aponte se é fotografia, cartum, tirinha, ilustração. Breve descrição com até 4 palavras]
                    - Pessoa: [sexo] [cor da pele], [posição na imagem]
                    - Cabelo: [descrição do tipo e cor usando sinônimos]
                    - Roupa: [descrição do tipo e cor usando sinônimos]
                    - Objeto: [breve descrição com até 4 palavras]
                    - Uso do objeto: [breve descrição com até 4 palavras]
                    - Ambiente: [breve descrição com até 4 palavras]

                    Ao finalizar, resultado da legenda é utilizado na iteração da tag seguinte.

                3. Após iterar sobre todas as tags, faça uma verificação na legenda original, na legenda de saída e nas tags. Garanta que toda informação existente na legenda de saída está, de certe forma, presente na legenda original ou nas tags.

                4. Faça a traduçao obrigatoriamente da legenda no idioma da entrada output_langue.

                5. Sua resposta final deve ser apenas o texto de legenda_gerada;
        """
        return prompt

    def extract_text_from_image_path(self, image_path):
        salesforce_caption = super().extract_text_from_image_path(image_path) # Chama Versao1
        
        # Add new steps here
        keywords = self.get_keywords_from_image_path(image_path)
        if keywords is not None:
            prompt = self._get_prompt(salesforce_caption, keywords)
            print(prompt)
            generated_text = self._call_chat_gpt_api(prompt)
            logging.info(f"Generated text: {generated_text}")
            return generated_text, keywords
    # end - extract_text_from_image_path()
        
    def extract_text_from_image_url(self, image_url):
        salesforce_caption = super().extract_text_from_image_url(image_url)

        # Add new steps here
        keywords = self.get_keywords_from_image_url(image_url)
        if keywords is not None:
            prompt = self._get_prompt(salesforce_caption, keywords)
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
                keyword_scores = [(keyword['keyword'], keyword['score']) for keyword in keywords if keyword['score'] >= 0.6]
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
                filtered_keywords = [(keyword['keyword'], keyword['score']) for keyword in keywords if keyword['score'] >= 0.6]
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
    