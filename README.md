# Read it Twice: Towards Faithfully Interpretable Fact Verification by Revisiting Evidence

## PyTorch
The code is based on PyTorch 1.13+. You can find tutorials [here](https://pytorch.org/tutorials/).

## Data
Download the CHEF dataset and unzip it under the ```src/chef``` directory. You can find the download links [here](https://github.com/THU-BPM/CHEF).
The resulting directory structure should look like this:
```
    data/
        CHEF_train.json
        CHEF_test.json
    train.py
```

## Usage
Run the full model on CHEF dataset with default hyperparameters
```
python3 train.py
```
This will train and evaluate our model.
You may use ```--cuda [device]``` to specify the GPU device to use for training and evaluation.
You can find our checkpoints at [Google Drive](https://drive.google.com/drive/folders/1XrfQrrtR6NlHXXj0lAMFMlWR_f-ub831?usp=sharing).

## Acknowledgements
We thank the authors of the original CHEF dataset and the Transformers library for their contributions.
