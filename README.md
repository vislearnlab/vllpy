# vllpy

This is a package with common utility functions, files and pipelines for the Visual Learning Lab. This package uses python>=3.11.

It is recommended you create a conda environment to start using this package but this step is optional. To do so, run the commands below:

```
conda create -n vislearnlabpy python=3.12
conda activate vislearnlabpy
```

Then, activate the environment and simply install vislearnlabpy and CLIP by running the commands below in your terminal. 

```
pip install --upgrade vislearnlabpy
pip install git+https://github.com/openai/CLIP.git
```

You may also have to install [PyTorch](https://pytorch.org/) manually, ensuring that you have the right version but the right version may also be installed by default with CLIP. To install the right version of PyTorch on the Tversky server, run:
```
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

Here is as an example of how to generate a CSV file with embeddings from a list of images in a directory. You can also use this to generate npy files and doc files by changing the output type in the command below, and generate the embeddings from a CSV file instead by using `input_file` instead of `input_dir`
``` 
python embedding_generator.py --input_dir examples/input --output_path examples/output --output_type csv --overwrite
```

For more detailed examples, please look at the demo in the Jupyter notebooks within the `examples` folder.
