import pandas as pd

def rename_csv_column(file, cols, new_name):
        for col in cols:
            if col in file.columns:
                file = file.rename(columns={col: new_name})
                return file # Rename only the first match

def process_csv(input_csv):
    images_df = pd.read_csv(input_csv)
    text_cols = ['class_name', 'word', 'text']
    image_cols = ['input_path', 'image_path']
    if 'text1' not in images_df.columns:
        images_df = rename_csv_column(images_df, text_cols, 'text1')
    if 'image1' not in images_df.columns:
        images_df = rename_csv_column(images_df, image_cols, 'image1')
    return images_df
