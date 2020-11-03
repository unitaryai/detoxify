<div align="center">    
 
# 🙊 Detoxify
##  Toxic Comment Classification with ⚡ Pytorch Lightning and 🤗 Transformers   

![CI testing](https://github.com/laurahanu/Jigsaw-toxic-comment-challenges/workflows/CI%20testing/badge.svg)
![Lint](https://github.com/laurahanu/Jigsaw-toxic-comment-challenges/workflows/Lint/badge.svg)

</div>
 
## Description   
Trained models & code to predict toxic comments on 3 Jigsaw challenges: Toxic comment classification, Unintended Bias in Toxic comments, Multilingual toxic comment classification.

Dependencies:
- 🤗 Transformers
- ⚡ Pytorch lightning 
- Kaggle API
- Python 3.6

| Challenge | Year | Goal | Original Data Source | Top Leaderboard Score
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

pip install -e Detoxify
cd Detoxify
pip install -r requirements.txt



 ```   

## Prediction
```bash

# Run on a comment, list of comments, or from a csv

python run_prediction.py --model_type multilingual --checkpoint model_path --input This is an example --device cpu 

# to see usage

python run_prediction.py --help

```

## Training

 If you do not already have a Kaggle account: 
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

```

 ```bash

# start training

python train.py --config configs/BERT_toxic_comment_classification_bias.json 

# monitor progress with tensorboard

tensorboard --logdir=./saved

# evaluate model

python evaluate.py --checkpoint saved/lightning_logs/checkpoints/checkpoint-0.pth --test_csv test_set.csv

```
## Checkpoints

Trained checkpoints available for each challenge.

| Model | Checkpoints | Score
|-|-|-|
| Toxic Comment Classification Challenge| path to checkpoint | N/a
| Unintended Bias Toxic Comment Classification Challenge | path to checkpoint | N/a
| MultilingualToxic Comment Classification Challenge | path to checkpoint | N/a

### Citation   
```
@article{Unitary,
  title={Detoxify},
  author={Unitary team},
  journal={Github. https://github.com/laurahanu/Jigsaw-toxic-comment-challenges},
  year={2020}
}
```   
