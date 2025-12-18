import pandas as pd
from PIL import Image
from IPython.display import display, HTML, clear_output
import os
import numpy as np
import time
import torch
from tqdm import tqdm
from pathlib import Path

def rename_csv_column(file, cols, new_name):
    for col in cols:
        if col in file.columns:
            file = file.rename(columns={col: new_name})
            return file # Rename only the first match
    return file

def process_csv(input_csv):
    images_df = pd.read_csv(input_csv)
    text_cols = ['class_name', 'word', 'text']
    image_cols = ['input_path', 'image_path', 'cropped_image_path']
    if 'text1' not in images_df.columns:
        images_df = rename_csv_column(images_df, text_cols, 'text1')
    if 'image1' not in images_df.columns:
        images_df = rename_csv_column(images_df, image_cols, 'image1')
    return images_df

def indexed_embeddings(embedding):
    curr_data = {}
    curr_embeddings = embedding.tolist() if isinstance(embedding, torch.Tensor) else embedding
    # new row with a separate column for each number in the 512 dimensions and one for the image_path as the row_id
    for i, value in enumerate(curr_embeddings):
        curr_data[f"{i}"] = value.item() if isinstance(value, torch.Tensor) else value
    return curr_data

def cleaned_doc_path(doc_path):
    if not os.path.isabs(doc_path.removeprefix("file://")):
        doc_path = str(Path(f"{os.getcwd()}/{doc_path.removeprefix('file://')}"))
    doc_path = doc_path.removesuffix(".docs")
    if not doc_path.startswith("file://"):
        doc_path = f"file://{doc_path}"
    return doc_path
       
def display_search_results(retrieved_docs, scores, sleep_time=0.3):
    clear_output(wait=True)
    html = "<table><tr>"
    temp_files = []
    for i, (img_path, text, score) in enumerate(zip(retrieved_docs.url, retrieved_docs.text, scores)):
        try:
            img = Image.open(img_path)
            img.thumbnail((200, 200))  # Resize for display

            # Save to temporary resized file
            timestamp = int(time.time())
            temp_path = f"_temp_{timestamp}_{i}.png"  # Unique filename per reload
            temp_files.append(temp_path)
            img.save(temp_path)
            # Add image + score in table cell
            html += f"""
            <td style='text-align: center; padding: 10px;'>
                <img src='{temp_path}'><br>
                <span>Cos sim: {score:.2f}</span>
                <span>Original label: {text}</span>
            </td>
            """

            if (i + 1) % 3 == 0:
                html += "</tr><tr>"

        except Exception as e:
            print(f"Error loading {img_path}: {e}")

    html += "</tr></table>"
    display(HTML(html))
    time.sleep(sleep_time)
    for f in temp_files:
        try:
            os.remove(f)
        except Exception as e:
            print(f"Could not delete temp file {f}: {e}")  

def save_df(df, filename, save_path=None, overwrite=False):
    """
    Save dataframe to CSV, appending to existing file if it exists.
    Avoids duplicate row_ids and handles new directory creation.
    """
    filepath = os.path.join(save_path, filename)
    # create directory if it does not exist
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    if not os.path.exists(filepath):
        df.to_csv(filepath, index=False)
        return
    try:
        existing_df = pd.read_csv(filepath)
        row_ids = df['row_id'].values
        if overwrite:
            # Remove rows from existing data that would be overwritten by new data
            existing_df = existing_df[~existing_df['row_id'].isin(row_ids)]
            existing_df.to_csv(filepath, index=False)             
        else:
            df = df[~df['row_id'].isin(existing_df['row_id'].values)]   
        df.to_csv(filepath, mode='a', header=False, index=False)      
    except pd.errors.EmptyDataError:
        # Handle case where existing file is empty
        df.to_csv(filepath, index=False)

def normalize_embeddings(embeddings):
    """Normalize embeddings (list of tensors or arrays) to unit L2 norm.
       Returns output in same format (NumPy or Tensor) as input.
    """
    normed = []
    for embedding in embeddings:
        if isinstance(embedding, torch.Tensor):
            norm = torch.norm(embedding, p=2, dim=-1, keepdim=True) + 1e-8
            normed.append(embedding / norm)
        elif isinstance(embedding, np.ndarray):
            norm = np.linalg.norm(embedding, axis=-1, keepdims=True) + 1e-8
            normed.append(embedding / norm)
        else:
            raise TypeError(f"Unsupported type: {type(embedding)}")
    return normed

# z-scoring embeddings
def zscore_embeddings(embeddings, add_rowwise_norm=False):
    mean_embedding = embeddings.mean(axis=0)
    std_embeddings = embeddings.std(axis=0)
    normalized_embeddings = (embeddings - mean_embedding) / (std_embeddings + 1e-8)
    if add_rowwise_norm:
        row_norms = np.linalg.norm(normalized_embeddings, axis=1, keepdims=True) + 1e-8
        normalized_embeddings = normalized_embeddings / row_norms
    return normalized_embeddings

# filter embeddings based on a text-image alignment value
def filter_embeddings(store, alignment_val=0.26):
    from vislearnlabpy.embeddings.embedding_store import EmbeddingStore
    curr_store:EmbeddingStore = store
    doc_list = curr_store.EmbeddingList
    filtered_store = EmbeddingStore()
    for description in tqdm(set(doc_list.text)):
        retrieved_docs, scores = curr_store.search_store(description, limit=100000, categories=[description])
        for img_path, text, embedding, score in zip(retrieved_docs.url, retrieved_docs.text, retrieved_docs.embedding, scores):
            if score >= alignment_val:
                filtered_store.add_embedding(embedding=embedding, url=img_path, text=text)
    print(f"Total rows in filtered_embeddings: {len(filtered_store.EmbeddingList)}")
    return filtered_store

    