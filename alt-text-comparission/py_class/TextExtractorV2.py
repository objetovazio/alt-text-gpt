import os
import logging
import requests
import openai
from .TextExtractorV1 import TextExtractorV1

class TextExtractorV2(TextExtractorV1):
    def __init__(self, get_from_file):
        super().__init__(get_from_file)
        self.ep_url = os.getenv('EVERYPIXEL_API_URL')  # get the EveryPixel API URL from environment variables
        self.ep_username = os.getenv('EVERYPIXEL_USERNAME')  # get the EveryPixel username from environment variables
        self.ep_api_key = os.getenv('EVERYPIXEL_API_KEY')  # get the EveryPixel API key from environment variables
        self.open_api_key = os.getenv('OPEN_API_KEY')

        self.auth = (self.ep_username, self.ep_api_key)

    def _get_prompt(self, salesforce_caption, keywords, language="Portuguese"):
        prompt = f"""
You will act as an advanced image analyzer and caption writer. Given an original English caption of an image, a set of tags associated with the image, and an output language, you will need to rewrite the caption. Your response should be simple and direct, use Output section as format.
The inputs are:

Inputs:
    Caption: {salesforce_caption}
    Tags: {keywords}
    Output Language: {language}

Output Format:
    'Caption: {{generated_caption}}'

Important: The generated_caption must be completely based on the input parameters. NEVER add information that is not present in the parameters, such as characteristics, colors, objects, etc. All necessary information is within the input parameters.

Here are the steps you should follow:

1. Read the caption and analyze its context. Read the tags, which will be provided in the format [('word', score)], and order them by score from highest to lowest. If no Tags is giver, jump directly to step 4.

2. Iterate in order over each tag and analyze if it can be added to the caption's context to enhance it. Tags can refer to anything in the image, such as people, objects, activities, colors, etc. When a tag is evaluated, it should be inserted into the description in one of the following ways:
    - Image type: [Indicate whether it is a photograph, cartoon, comic strip, illustration. Brief description with up to 4 words]
    - Ethnicity: Ethnicities should be dealt with in more detail, do not write it directly. Point out one or two characteristics of the ethnicity. Skin color should be described using the IBGE terms: white, black, brown, indigenous, or yellow.
    - Hair: [description of type and color using synonyms]
    - Clothing: [description of type and color using synonyms]
    - Object: [brief description with up to 4 words]
    - Object usage: [brief description with up to 4 words]
    - Environment: [brief description with up to 4 words]

3. After iterating over all the tags, cross-check the original caption, the output caption, and the tags. Ensure that all information present in the output caption is, in some way, present in the original caption or in the tags.

4. Translate the final caption to the language specified in the 'output_language' field, considering cultural and regional nuances when applicable.

5. Give the result in output format.
"""
        return prompt

    def extract_text_from_image_path(self, image_path):
        salesforce_caption, keywords = super().extract_text_from_image_path(image_path) # Chama Versao1
        
        # Add new steps here
        keywords = self.get_keywords_from_image_path(image_path)
        if keywords is not None:
            prompt = self._get_prompt(salesforce_caption, keywords, 'English')
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
    