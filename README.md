# vllpy

This is a package with common utility functions, files and pipelines for the Visual Learning Lab. Creating a conda environment is recommended but optional. This package uses python=3.12.

```
conda create -n vislearnlabpy python=3.12
conda activate vislearnlabpy
```

Then, activate the environment and simply install vislearnlabpy via running the following pip command in your terminal. You will also have to install [PyTorch](https://pytorch.org/) and CLIP manually.

```
pip install git+https://github.com/openai/CLIP.git
pip install --upgrade vislearnlabpy
```

To install PyTorch on the Tversky server, run:
```
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

Here is as an example of how to generate npy embedding files from a list of images whose relative paths are defined in a CSV file
```
python embedding_generator.py --input_csv examples/input/inputs.csv --output_path examples/output --output_type npy --input_dir [working_directory] --overwrite
```

If the full paths are defined in the CSV file, you can similarly use 
```
python embedding_generator.py --input_csv [input_path] --output_path [output_path] --output_type npy
```
