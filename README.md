<div align="center">    
 
# 🙊 Detoxify
##  Toxic Comment Classification with ⚡ Pytorch Lightning and 🤗 Transformers   

![CI testing](https://github.com/laurahanu/Jigsaw-toxic-comment-challenges/workflows/CI%20testing/badge.svg)
![Lint](https://github.com/laurahanu/Jigsaw-toxic-comment-challenges/workflows/Lint/badge.svg)

</div>
 
## Description   

Trained models & code to predict toxic comments on 3 Jigsaw challenges: Toxic comment classification, Unintended Bias in Toxic comments, Multilingual toxic comment classification.

Dependencies:
- For inference:
  - 🤗 Transformers
  - ⚡ Pytorch lightning 
- For training:
  - Kaggle API (to download data)


| Challenge | Year | Goal | Original Data Source | Top Leaderboard Score
|-|-|-|-|-|
| [Toxic Comment Classification Challenge](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge) | 2018 |  build a multi-headed model that’s capable of detecting different types of of toxicity like threats, obscenity, insults, and identity-based hate. | Wikipedia Comments | 0.98856
| [Jigsaw Unintended Bias in Toxicity Classification](https://www.kaggle.com/c/jigsaw-unintended-bias-in-toxicity-classification) | 2019 | build a model that recognizes toxicity and minimizes this type of unintended bias with respect to mentions of identities. You'll be using a dataset labeled for identity mentions and optimizing a metric designed to measure unintended bias. | Civil Comments | 0.94734
| [Jigsaw Multilingual Toxic Comment Classification](https://www.kaggle.com/c/jigsaw-multilingual-toxic-comment-classification) | 2020 | build effective multilingual models | Wikipedia Comments + Civil Comments | 0.9536

## Labels
All challenges have a toxicity label. The toxicity labels represent the aggregate ratings of up to 10 annotators according the following schema:
- **Very Toxic** (a very hateful, aggressive, or disrespectful comment that is very likely to make you leave a discussion or give up on sharing your perspective)
- **Toxic** (a rude, disrespectful, or unreasonable comment that is somewhat likely to make you leave a discussion or give up on sharing your perspective)
- **Hard to Say**
- **Not Toxic**

More information about the labelling schema can be found [here](https://www.kaggle.com/c/jigsaw-unintended-bias-in-toxicity-classification/data).

### Toxic Comment Classification Challenge
This challenge includes the following labels:

- `toxic`
- `severe_toxic`
- `obscene`
- `threat`
- `insult`
- `identity_hate`

### Jigsaw Unintended Bias in Toxicity Classification
This challenge has 2 types of labels: the main toxicity labels and some additional identity labels that represent the identities mentioned in the comments. 

Only identities with more than 500 examples in the test set (combined public and private) are included during training as additional labels and in the evaluation calculation. These identities are shown in bold.

- `toxicity`
- `severe_toxicity`
- `obscene`
- `threat`
- `insult`
- `identity_attack`
- `sexual_explicit`

Identity labels:
- **`male`**
- **`female`**
- `transgender`
- `other_gender`
- `heterosexual`
- **`homosexual_gay_or_lesbian`**
- `bisexual`
- `other_sexual_orientation`
- **`christian`**
- **`jewish`**
- **`muslim`**
- `hindu`
- `buddhist`
- `atheist`
- `other_religion`
- **`black`**
- **`white`**
- `asian`
- `latino`
- `other_race_or_ethnicity`
- `physical_disability`
- `intellectual_or_learning_disability`
- **`psychiatric_or_mental_illness`**
- `other_disability`


### Jigsaw Multilingual Toxic Comment Classification

Since this challenge combines the data from the previous 2 challenges, it includes all labels from above, however the final evaluation is only on:

- `toxicity`

## How to run   

First, install dependencies   
```bash
# clone project   

git clone https://github.com/laurahanu/detoxify

# create virtual env

python3 -m venv toxic-env
source toxic-env/bin/activate

# install project   

pip install -e detoxify
cd detoxify
pip install -r requirements.txt


 ```   

## Prediction
```bash

# Run on a comment or from a csv

# model_type options: original, bias, multilingual

python run_prediction.py 'shut up, you are a liar' --model_type original --device cpu --checkpoint model_path

# to see usage

python run_prediction.py --help

```
## Examples

| Example Input | Model Type | Language | Toxicity | Severe_toxicity | Obscene | Threat | Insult | Identity_attack | Sexually_explicit
|-|-|-|-|-|-|-|-|-|-|
| shut up, you are a liar | Original | en | 0.988 | 0.015 | 0.356 | 0.002 | 0.838 | 0.007 | N/A
| I am a jewish woman who is blind | Bias | en | 0.171 | 0.001 | 0.004 | 0.004 |  0.032 | 0.088 | 0.001
| tais-toi, tu es un menteur | Multilingual | fr | 0.987 | N/A | N/A | N/A | N/A | N/A | N/A
| cállate, eres mentirosa | Multilingual | es | 0.945 | N/A | N/A | N/A | N/A | N/A | N/A
| kapa çeneni, sen bir yalancısın | Multilingual | tr | 0.952 | N/A | N/A | N/A | N/A | N/A | N/A
| stai zitto, tu sei un bugiardo | Multilingual | it | 0.986 | N/A | N/A | N/A | N/A | N/A | N/A
| заткнись, ты лжец | Multilingual | ru | 0.941 | N/A | N/A | N/A | N/A | N/A | N/A
| cala a boca, você é um mentiroso | Multilingual | pt | 0.81 | N/A | N/A | N/A | N/A | N/A | N/A

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
## Start Training
 ### Toxic Comment Classification Challenge

 ```bash

python create_val_set.py

python train.py --config configs/Toxic_comment_classification_BERT.json
``` 
 ### Unintended Bias in Toxicicity Challenge

```bash

python train.py --config configs/Unintended_bias_toxic_comment_classification_RoBERTa.json
```
 ### Multilingual Toxic Comment Classification

 This is trained in 2 stages. First, train on all available data, and second, train only on the translated versions of the first challenge. 
 
 The [translated data](https://www.kaggle.com/miklgr500/jigsaw-train-multilingual-coments-google-api) can be downloaded from Kaggle in french, spanish, italian, portuguese, turkish, and russian (the languages available in the test set).

 ```bash

# stage 1

python train.py --config configs/Multilingual_toxic_comment_classification_XLMR.json

# stage 2

python train.py --config configs/Multilingual_toxic_comment_classification_XLMR_stage2.json

```
### Monitor progress with tensorboard

 ```bash

tensorboard --logdir=./saved
```
## Model Evaluation

### Toxic Comment Classification Challenge

This challenge is evaluated on the mean AUC score of all the labels.

```bash

python evaluate.py --checkpoint saved/lightning_logs/checkpoints/example_checkpoint.pth --test_csv test.csv

```
### Unintended Bias in Toxicicity Challenge

This challenge is evaluated on a novel bias metric that combines different AUC scores to balance overall performance. More information on this metric [here](https://www.kaggle.com/c/jigsaw-unintended-bias-in-toxicity-classification/overview/evaluation).

```bash

python evaluate.py --checkpoint saved/lightning_logs/checkpoints/example_checkpoint.pth --test_csv test.csv

# to get the final bias metric
python model_eval/compute_bias_metric.py

```
### Multilingual Toxic Comment Classification

This challenge is evaluated on the AUC score of the main toxic label.

```bash

python evaluate.py --checkpoint saved/lightning_logs/checkpoints/example_checkpoint.pth --test_csv test.csv

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
