import os
import pandas as pd
import logging
import replicate

import tensorflow as tf
import tensorflow_hub as hub

logging.basicConfig(
    filename='./alt-text-comparission/output.log',
    format="[Log.%(levelname)s][%(asctime)s]: %(message)s",
    level=logging.DEBUG
)

class ImageCaptionProcessor:
    def __init__(self, image_dir_path, csv_path):
        self.image_dir_path = image_dir_path
        self.csv_path = csv_path
        if not os.path.isdir(self.image_dir_path):
            raise ValueError(f"{self.image_dir_path} is not a directory or does not exist.")
        if not os.path.isfile(self.csv_path):
            raise ValueError(f"{self.csv_path} is not a file or does not exist.")
        logging.info("ImageCaptionProcessor instaciated successfully.")
    #end

    def load_csv_data(self):
        self.data = pd.read_csv(self.csv_path)
        return self.data
    #end

    def extract_captions(self, output_csv_path):
        logging.info("Start Extract captions from existing photos.")

        # If the output file already exists, exit the function
        if os.path.isfile(output_csv_path):
            logging.info(f"Output file {output_csv_path} already exists. Skipping generation...")
            return

        self.load_csv_data()
        valid_rows = []

        for idx, row in self.data.iterrows():
            if os.path.isfile(os.path.join(self.image_dir_path, row['image'])):
                valid_rows.append(row)
        
        if valid_rows:
            valid_df = pd.DataFrame(valid_rows)
            valid_df.to_csv(output_csv_path, index=False)
        else:
            logging.info("No valid images found.")
        
        logging.info("Extract captions completed.")
    #end

    def extract_text_from_image(self, image_path):
        logging.debug(f"Extracting caption from {image_path}.")

        with open(image_path, "rb") as image:
            model = "salesforce/blip:2e1dddc8621f72155f24cf2e0adbde548458d3cab9f00c0139eea840d0ac4746"
            
            output_text = replicate.run(
                model,
                input={"image": image},
            )

        return output_text
    #end

    def generate_photos_captions(self, output_csv_path):
            logging.info("Starting to generate photo captions.")

            if os.path.isfile(output_csv_path):
                existing_df = pd.read_csv(output_csv_path)
            else:
                existing_df = pd.DataFrame(columns=['image', 'caption'])
                existing_df.to_csv(output_csv_path, mode='a', index=False)
            
            for image_file in os.listdir(self.image_dir_path):
                if image_file.endswith('.jpg') or image_file.endswith('.png'):  # assuming image files are either jpg or png
                    if image_file in existing_df['image'].values:  # check if caption already exists
                        logging.info(f"Caption for {image_file} already exists. Skipping...")
                        continue
                    
                    image_path = os.path.join(self.image_dir_path, image_file)
                    caption = self.extract_text_from_image(image_path).split(": ")[1]
                    df = pd.DataFrame([{"image": image_file, "caption": caption}])
                    df.to_csv(output_csv_path, mode='a', index=False, header=False)

            logging.info("Generate photo captions completed.")

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
    #end

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

