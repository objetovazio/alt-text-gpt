import os
import pandas as pd
import logging
from functools import lru_cache

import tensorflow as tf
import tensorflow_hub as hub

class ImageCaptionProcessor:
    def __init__(self, image_dir_path, csv_path, text_extractor, check_in_file=False):
        self.model1 = self.load_tf_model1()
        self.model2 = self.load_tf_model2()
        self.image_dir_path = image_dir_path
        self.csv_path = csv_path
        self.text_extractor = text_extractor
        self.check_in_file = check_in_file
        
        if not os.path.isdir(self.image_dir_path):
            logging.error(f"{self.image_dir_path} is not a directory or does not exist.")
            raise ValueError(f"{self.image_dir_path} is not a directory or does not exist.")

        if not os.path.isfile(self.csv_path):
            logging.error(f"{self.csv_path} is not a file or does not exist.")
            raise ValueError(f"{self.csv_path} is not a file or does not exist.")
        
        logging.info("ImageCaptionProcessor instantiated successfully.")
    # end - __init__()

    @staticmethod
    @lru_cache(maxsize=1)
    def load_tf_model1():
        return hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")
    
    @staticmethod
    @lru_cache(maxsize=1)
    def load_tf_model2():
        return hub.load("https://tfhub.dev/google/tf2-preview/nnlm-en-dim128/1")

    def load_csv_data(self):
        self.data = pd.read_csv(self.csv_path)
        logging.info("CSV data loaded successfully.")
        return self.data
    # end - load_csv_data()

    def extract_captions(self, output_csv_path):
        logging.info("Start extracting captions from existing photos.")

        # Check if the output file exists
        if os.path.isfile(output_csv_path):
            # Open the existing file with pandas
            output_csv = pd.read_csv(output_csv_path)
        else:
            # Create a new DataFrame with 'image' and 'caption' columns
            output_csv = pd.DataFrame(columns=['image', 'caption'])
            output_csv.to_csv(output_csv_path, index=False)

        self.load_csv_data()
        valid_rows = []

        for idx, row in self.data.iterrows():
            if os.path.isfile(os.path.join(self.image_dir_path, row['image'])):
                if row['image'] in output_csv['image'].values:
                    logging.info(f"Caption for image {row['image']} already exists in the output file. Skipping...")
                    continue
                valid_rows.append(row)

        if valid_rows:
            valid_df = pd.DataFrame(valid_rows)
            output_csv = pd.concat([output_csv, valid_df], ignore_index=True)
            output_csv.to_csv(output_csv_path, index=False)
            logging.info("Captions successfully extracted and written to the output file.")
        else:
            logging.warning("No new valid images found.")

        logging.info("Extract captions completed.")
    # end - extract_captions()

    def retrieve_caption_from_csv(self, image_name):
        df = pd.read_csv('/mnt/c/github/alt-text-gpt/alt-text-comparission/files/generated-salesforce-captions.csv')
        caption = df.loc[df['image'] == image_name, 'caption'].values
        if caption and len(caption) > 0:
            return caption[0]
        return None

    def generate_photos_captions(self, output_csv_path, regenerate=False):
        logging.info("Starting to generate photo captions.")

        # Check if the output file exists
        if os.path.isfile(output_csv_path):
            # Open the existing file with pandas
            existing_df = pd.read_csv(output_csv_path)
        else:
            # Create a new DataFrame with 'image' and 'caption' columns
            existing_df = pd.DataFrame(columns=['image', 'keywords', 'caption'])
            existing_df.to_csv(output_csv_path, mode='a', index=False)
        
        new_rows = []
            
        for image_file in os.listdir(self.image_dir_path):

            if(self.check_in_file):
                already_generated = len(existing_df.loc[existing_df['image'] == image_file].values) > 0
                if(already_generated and not regenerate):
                    logging.info(f"{image_file} existis in file... skipping text generation")
                    continue

            if image_file.endswith('.jpg') or image_file.endswith('.png'):
                image_path = os.path.join(self.image_dir_path, image_file)
                caption, keywords = self.text_extractor.extract_text_from_image_path(image_path)
                
                if caption:
                    new_row = {'image': image_file, 'keywords': keywords, 'caption': caption}
                    new_rows.append(new_row)
                    logging.info(f"Generated caption for {image_file} and added it to the output file.")
                else:
                    logging.warning(f"Failed to generate caption for {image_file}. Skipping...")

        if new_rows:
            new_df = pd.DataFrame(new_rows)
            new_df.to_csv(output_csv_path, mode='a', index=False, header=not os.path.isfile(output_csv_path))
            logging.info("Generated captions added to the output file.")
        else:
            logging.info("No new captions generated.")

        logging.info("Generate photo captions completed.")
    # end - generate_photos_captions()
    
    def compare_captions_nnlm(self, captions_csv_path, generated_csv_path, output_csv_path):
        model = hub.load("https://tfhub.dev/google/tf2-preview/nnlm-en-dim128/1")

        captions_df = pd.read_csv(captions_csv_path)
        generated_df = pd.read_csv(generated_csv_path)

        results = []

        for _, gen_row in generated_df.iterrows():
            gen_image = gen_row['image']
            gen_caption = gen_row['caption']
            gen_vector = model([gen_caption])

            captions_same_image = captions_df[captions_df['image'] == gen_image]

            for _, caption_row in captions_same_image.iterrows():
                caption = caption_row['caption'].split(' .')[0]
                caption_vector = model([caption])

                cosine_similarity = -tf.keras.losses.cosine_similarity(gen_vector, caption_vector, axis=-1).numpy()[0]
                cosine_similarity_formatted = format(cosine_similarity, '.5f')
                results.append({"image": gen_image, "generated_caption": gen_caption, "original_caption": caption, "value": cosine_similarity_formatted})

        pd.DataFrame(results).to_csv(output_csv_path, index=False)
        logging.info("Comparisons between captions completed and saved to the output file.")
    # end - compare_captions_nnlm()

    def compare_captions(self, captions_csv_path, generated_csv_path, output_csv_path):
        model1 = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")
        model2 = hub.load("https://tfhub.dev/google/tf2-preview/nnlm-en-dim128/1")

        captions_df = pd.read_csv(captions_csv_path)
        generated_df = pd.read_csv(generated_csv_path)

        results = []

        for _, gen_row in generated_df.iterrows():
            gen_image = gen_row['image']
            gen_caption = gen_row['caption']
            gen_vector1 = model1([gen_caption])
            gen_vector2 = model2([gen_caption])

            captions_same_image = captions_df[captions_df['image'] == gen_image]

            for _, caption_row in captions_same_image.iterrows():
                caption = caption_row['caption'].split(' .')[0]
                caption_vector1 = model1([caption])
                caption_vector2 = model2([caption])

                cosine_similarity1 = -tf.keras.losses.cosine_similarity(gen_vector1, caption_vector1, axis=-1).numpy()[0]
                cosine_similarity2 = -tf.keras.losses.cosine_similarity(gen_vector2, caption_vector2, axis=-1).numpy()[0]

                cosine_similarity1_formatted = format(cosine_similarity1, '.5f')
                cosine_similarity2_formatted = format(cosine_similarity2, '.5f')

                results.append({
                    "image": gen_image, 
                    "generated_caption": gen_caption, 
                    "original_caption": caption, 
                    "value1": cosine_similarity1_formatted,
                    "value2": cosine_similarity2_formatted
                })

        pd.DataFrame(results).to_csv(output_csv_path, index=False)
        logging.info("Comparisons between captions completed and saved to the output file.")
    # end - compare_captions()

