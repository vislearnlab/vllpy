import argparse
from dataclasses import asdict
import torch
from vislearnlabpy.embeddings.generate_embeddings import EmbeddingConfig, EmbeddingGenerator


def main():
    parser = argparse.ArgumentParser(description="Generate embeddings from models")
    parser.add_argument("--input_dir", type=str, help="Input directory with images")
    parser.add_argument("--input_csv", type=str, required=False, help="CSV with image metadata")
    parser.add_argument("--output_path", type=str, default="examples/output", help="Output path")
    parser.add_argument("--output_type", type=str, default="csv", help="Embedding format: npy, doc, or csv")
    parser.add_argument("--device", type=str, required=False, help="Device (e.g. cuda:0, cpu)")
    parser.add_argument("--batch_size", type=int, default=1000, help="Batch size")
    parser.add_argument("--id_column", type=str, default="image1", help="Image id column in csv")
    parser.add_argument("--text_prompt", type=str, default="a photo of a ", help="Text prompt prefix for CLIP")
    parser.add_argument("--overwrite", action="store_true", default=False, help="Overwrite existing data")
    parser.add_argument("--normalize", action="store_true", default=False, help="Normalize embeddings")
    parser.add_argument("--subdirs", action="store_true", default=False, help="Preserve subdirectory structure")
    parser.add_argument("--model_name", type=str, default="clip", help="Model type to use for embeddings (e.g. clip, dinov3)")
    parser.add_argument("--ray_temp_dir", type=str, default="/Scratch/tmp/ray", help="Ray temp/spill directory, often set to /tmp/ray")
    args, extra = parser.parse_known_args()

    # parse extra kwargs like --num_actors 3 --gpu_per_actor 1.0
    config_kwargs = {}
    it = iter(extra)
    for token in it:
        if token.startswith("--"):
            key = token[2:].replace("-", "_")
            val = next(it, "true")
            # attempt numeric coercion
            for cast in (int, float):
                try:
                    val = cast(val)
                    break
                except ValueError:
                    pass
            config_kwargs[key] = val

    if args.input_csv is None and args.input_dir is None:
        args.input_dir = "examples/input"

    generator = EmbeddingGenerator.from_model(
        args.model_name,
        output_type=args.output_type,
        device=args.device,
        text_prompt=args.text_prompt,
        normalize_embeddings=args.normalize,
        ray_temp_dir=args.ray_temp_dir,
        **config_kwargs,
    )

    generator.generate_image_embeddings(
        input_dir=args.input_dir,
        input_csv=args.input_csv,
        output_path=args.output_path,
        overwrite=args.overwrite,
        batch_size=args.batch_size,
        id_column=args.id_column,
        subdirs=args.subdirs,
    )


if __name__ == "__main__":
    main()
