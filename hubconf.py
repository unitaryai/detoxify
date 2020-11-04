dependencies = ['torch','transformers']
from train import get_model

MODEL_DEFS = {
            'original': {'model_type': "bert-base-uncased",
                           'model_name': "BertForSequenceClassification",
                           'num_classes': 6},
            'bias': {'model_type': "roberta-base",
                     'model_name': "RobertaForSequenceClassification",
                     'num_classes': 16},
            'multilingual': {'model_type': "xlm-roberta-base",
                             'model_name': "XLMRobertaForSequenceClassification",
                             'num_classes': 1},
            }

def toxic_bert():
    model = get_model(**MODEL_DEFS['original'])
    checkpoint = 'https://github.com/laurahanu/detoxify/releases/download/v0.1-alpha/toxic_original.ckpt'
    model.load_state_dict(torch.hub.load_state_dict_from_url(checkpoint, progress=False))
    return model

def toxic_roberta_bias():
    model = get_model(**MODEL_DEFS['bias'])
    checkpoint = 'https://github.com/laurahanu/detoxify/releases/download/v0.1-alpha/toxic_bias.ckpt'
    model.load_state_dict(torch.hub.load_state_dict_from_url(checkpoint, progress=False))
    return model

def toxic_xlmr_multilingual():
    model = get_model(**MODEL_DEFS['multilingual'])
    checkpoint = 'https://github.com/laurahanu/detoxify/releases/download/v0.1-alpha/toxic_multilingual.ckpt'
    model.load_state_dict(torch.hub.load_state_dict_from_url(checkpoint, progress=False))
    return model