<div align="center">    
 
# Jigsaw Toxic Comment Classification Implementations     

![CI testing](https://github.com/PyTorchLightning/deep-learning-project-template/workflows/CI%20testing/badge.svg?branch=master&event=push)


<!--  
Conference   
-->   
</div>
 
## Description   
Library to easily train and test classifiers for all 3 Jigsaw Toxic Comment Challenges.

Dependencies:
- 🤗 Transformers
- ⚡ Pytorch lightning 
- Kaggle API

| Challenge | Year | Description | Original Data Source | Top Leaderboard Score
|-|-|-|-|-|
| [Toxic Comment Classification Challenge](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge) | 2018 |  build a multi-headed model that’s capable of detecting different types of of toxicity like threats, obscenity, insults, and identity-based hate. | Wikipedia Comments | 0.98856
| [Jigsaw Unintended Bias in Toxicity Classification](https://www.kaggle.com/c/jigsaw-unintended-bias-in-toxicity-classification) | 2019 | build a model that recognizes toxicity and minimizes this type of unintended bias with respect to mentions of identities. You'll be using a dataset labeled for identity mentions and optimizing a metric designed to measure unintended bias. | Civil Comments | 0.94734
| [Jigsaw Multilingual Toxic Comment Classification](https://www.kaggle.com/c/jigsaw-multilingual-toxic-comment-classification) | 2020 | build effective multilingual models | Wikipedia Comments + Civil Comments | 0.9536


## How to run   
First, install dependencies   
```bash
# clone project   
git clone https://github.com/laurahanu/Jigsaw-toxic-comment-challenges

# create virtual env
python3 -m venv toxic-env
source toxic-env/bin/activate

# install project   
cd Jigsaw-toxic-comment-challenges 
pip install -e . 
pip install -r requirements.txt
 ```   
 Commands to download the Jigsaw data (requires to have the Kaggle API installed). 
 
 If you do not already have a kaggle account: 
 - you need to create one to be able to download the data
 
 - go to My Account and click on Create New API Token - this will download a kaggle.json file

 - make sure this file is located in ~/.kaggle
 ```bash

# create data directory

mkdir jigsaw_data
cd jigsaw_data

# download data

kaggle competitions download -c jigsaw-toxic-comment-classification-challenge

kaggle competitions download -c jigsaw-unintended-bias-in-toxicity-classification

kaggle competitions download -c jigsaw-multilingual-toxic-comment-classification

# unzip 

unzip -d jigsaw-toxic-comment-classification-challenge jigsaw-toxic-comment-classification-challenge.zip

unzip -d  jigsaw-unintended-bias-in-toxicity-classification jigsaw-unintended-bias-in-toxicity-classification.zip

unzip -d jigsaw-multilingual-toxic-comment-classification jigsaw-multilingual-toxic-comment-classification.zip

```

## Imports
This project is setup as a package which means you can now easily import any file into any other file like so:
```python
from project.datasets.mnist import mnist
from project.lit_classifier_main import LitClassifier
from pytorch_lightning import Trainer

# model
model = LitClassifier()

# data
train, val, test = mnist()

# train
trainer = Trainer()
trainer.fit(model, train, val)

# test using the best model!
trainer.test(test_dataloaders=test)
```
## Checkpoints

Trained checkpoints available for each.

| Model | Checkpoints | Score
|-|-|-|
| Toxic Comment Classification Challenge| path to checkpoint | N/a
| Unintended Bias Toxic Comment Classification Challenge | path to checkpoint | N/a
| MultilingualToxic Comment Classification Challenge | path to checkpoint | N/a

### Citation   
```
@article{YourName,
  title={Your Title},
  author={Your team},
  journal={Location},
  year={Year}
}
```   
