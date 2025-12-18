import argparse
import torch 
from vislearnlabpy.embeddings.generate_embeddings import generate_image_embeddings, EmbeddingGenerator

def main():
    parser = argparse.ArgumentParser(description="Generate embeddings from models")
    parser.add_argument("--input_dir", type=str, help="Input directory with images")
    parser.add_argument("--input_csv", type=str, required=False, help="CSV with more detailed information about embeddings")
    parser.add_argument("--output_path", type=str, default="examples/output", help="Output embeddings path")
    parser.add_argument("--output_type", type=str, default="csv", help="Embedding format (npy, doc or csv)")
    parser.add_argument("--device", type=str, required=False, help="Device to run embedding generation on")
    parser.add_argument("--batch_size", type=int, default=1000, help="Batch size to save embeddings")
    parser.add_argument("--id_column", type=str, default="image1", help="Image id column in csv")
    parser.add_argument(
        "--overwrite",
        action='store_true',
        default=False,
        help="Whether to overwrite existing saved data"
    )
    parser.add_argument(
        "--normalize",
        action='store_true',
        default=False,
        help="Whether to normalize embeddings before saving them"
    )
    args = parser.parse_args()
    if args.device is None:
        input_device = "cuda:0" if torch.cuda.is_available() else "cpu"
    else:
        input_device = args.device
    if args.input_csv is None and args.input_dir is None:
        args.input_dir = "examples/input"
    generate_image_embeddings(args.input_dir, args.input_csv, args.output_path, args.overwrite,
                                args.batch_size, id_column=args.id_column, subdirs=True,
                                input_device=input_device, output_type=args.output_type)

if __name__ == "__main__":
    main()