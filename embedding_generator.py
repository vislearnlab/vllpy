import argparse
import torch 
from vislearnlabpy.embeddings.generate_embeddings import EmbeddingGenerator

def main():
    parser = argparse.ArgumentParser(description="Generate embeddings from models")
    parser.add_argument("--input_dir", type=str, help="Input directory with images")
    parser.add_argument("--input_csv", type=str, required=False, help="CSV with more detailed information about embeddings")
    parser.add_argument("--output_path", type=str, default="examples/output", help="Output embeddings path")
    parser.add_argument("--output_type", type=str, default="csv", help="Embedding format (npy or csv)")
    parser.add_argument("--device", type=str, required=False, help="Device to run embedding generation on")
    parser.add_argument("--batch_size", type=int, default=1000, help="Batch size to save embeddings")
    parser.add_argument(
        "--overwrite",
        action='store_true',
        default=False,
        help="Whether to overwrite existing saved data"
    )
    parser.add_argument(
        "--normalize",
        action='store_true',
        default=True,
        help="Whether to normalize embeddings before saving them"
    )
    args = parser.parse_args()
    if args.device is None:
        input_device = "cuda:0" if torch.cuda.is_available() else "cpu"
    else:
        input_device = args.device
    if args.input_csv is None and args.input_dir is None:
        args.input_dir = "examples/input"
    embeddingGenerator = EmbeddingGenerator(device=input_device, output_type=args.output_type)
    embeddingGenerator.generate_image_embeddings(args.output_path, args.overwrite, args.normalize, args.input_csv, args.input_dir,
                                                 args.batch_size)
    
if __name__ == "__main__":
    main()