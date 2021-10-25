# Entropy Estimation via Backpropagation

This repository implements the methodology and experiments from my Master's thesis "Towards Measuring Discrete Entropy under Continuous Transformations -- A Case Study by Mapping into the Original Domain".

To run this code, Python 3 is required. Other requirements are as per python/requirements.txt. To install them, simply run:
```
pip3 install -r requirements.txt
```

It is advised to set up a separate virtual environment for this beforehand to ensure that there are no dependency conflicts with other packages in the global environment. On POSIX systems with bash or zsh shells:
```bash
python3 -m venv env
source env/bin/activate
pip --install --upgrade pip
```
For other systems follow the [official instructions](https://docs.python.org/3/library/venv.html).

To run the experiments from the thesis use:
```bash
python src/evaluation_setup.py
```
and
```bash
python src/exploration_setup.py
```
with the desired options.

For the exploration setup, you will also need to download the ImageNet64 dataset from the [ImageNet website](https://www.image-net.org/) and unpack it into `src/ImageNet64` such that the directory contains `train_data_batch_1` to `train_data_batch_10` as well as `val_data`.

Also make sure that the repository was cloned recursively or run
```bash
git submodule update --init --recursive
```
to receive the [original iResNet implementation](https://github.com/jhjacobsen/invertible-resnet).

Computational artifacts will automatically be saved to make reruns faster. By default these will be saved in `~/.reproducibles`. This can be configured by creating a file named `.reproducibles.yml` in the user's home directory with the content:
```yaml
base_dir: /path/to/reproducibles_dir
```
where `/path/to/reproducibles_dir` is the path to the directory in which files should be saved.
