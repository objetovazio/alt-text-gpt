import os
import pandas as pd
import logging
import replicate

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
            logging.warning(f"Output file {output_csv_path} already exists. Skipping generation...")
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
            logging.warning("No valid images found.")
        
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
            logging.warning(f"Output file {output_csv_path} already exists. Skipping...")
            return

        first_row = True
        for image_file in os.listdir(self.image_dir_path):
            if image_file.endswith('.jpg') or image_file.endswith('.png'):  # assuming image files are either jpg or png
                image_path = os.path.join(self.image_dir_path, image_file)
                caption = self.extract_text_from_image(image_path).split(": ")[1]
                df = pd.DataFrame([{"image": image_file, "caption": caption}])
                df.to_csv(output_csv_path, mode='a', index=False, header=first_row)
                first_row = False

        logging.info("Generate photo captions completed.")
    #end

