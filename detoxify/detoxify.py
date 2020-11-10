import torch
import transformers

MODEL_URLS = {
    "original": "https://github.com/unitaryai/detoxify/releases/download/v0.1-alpha/toxic_original-c1212f89.ckpt",
    "unbiased": "https://github.com/unitaryai/detoxify/releases/download/v0.1-alpha/toxic_bias-4e693588.ckpt",
    "multilingual": "https://github.com/unitaryai/detoxify/releases/download/v0.1-alpha/toxic_multilingual-bbddc277.ckpt",
}

PRETRAINED_MODEL = None


def get_model_and_tokenizer(
    model_type, model_name, tokenizer_name, num_classes, state_dict
):
    model_class = getattr(transformers, model_name)
    model = model_class.from_pretrained(
        pretrained_model_name_or_path=None,
        config=model_type,
        num_labels=num_classes,
        state_dict=state_dict,
    )
    tokenizer = getattr(transformers, tokenizer_name).from_pretrained(model_type)

    return model, tokenizer


def load_checkpoint(model_type="original", checkpoint=None):
    if checkpoint is None:
        checkpoint_path = MODEL_URLS[model_type]
        loaded = torch.hub.load_state_dict_from_url(checkpoint_path)
    else:
        loaded = torch.load(checkpoint)
        if "config" not in loaded or "state_dict" not in loaded:
            raise ValueError(
                "Checkpoint needs to contain the config it was trained \
                    with as well as the state dict"
            )
    class_names = loaded["config"]["dataset"]["args"]["classes"]

    model, tokenizer = get_model_and_tokenizer(
        **loaded["config"]["arch"]["args"], state_dict=loaded["state_dict"]
    )

    return model, tokenizer, class_names


def load_model(model_type, checkpoint=None):
    if checkpoint is None:
        model, _, _ = load_checkpoint(model_type=model_type)
    else:
        model, _, _ = load_checkpoint(checkpoint=checkpoint)
    return model


class Detoxify:
    """Detoxify
    Easily predict if a comment or list of comments is toxic.
    Can initialize 3 different model types from model type or checkpoint path:
        - original:
            BERT model trained on data from the Jigsaw Toxic Comment
            Classification Challenge
        - unbiased:
            RoBERTa model trained on data from the Jigsaw Unintended Bias in
            Toxicity Classification Challenge
        - multilingual:
            XLM-RoBertA model trained on data from the Jigsaw Multilingual
            Toxic Comment Classification Challenge
    Args:
        model_type(str): model type to be loaded, can be either original,
                         unbiased or multilingual
        checkpoint(str): checkpoint path, defaults to None
    Returns:
        results(dict): dictionary of output scores for each class
    """

    def __init__(self, model_type="original", checkpoint=PRETRAINED_MODEL):
        super(Detoxify, self).__init__()
        self.model, self.tokenizer, self.class_names = load_checkpoint(
            model_type=model_type, checkpoint=checkpoint
        )
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)

    @torch.no_grad()
    def predict(self, text):
        self.model.eval()
        inputs = self.tokenizer(
            text, return_tensors="pt", truncation=True, padding=True
        ).to(self.model.device)
        out = self.model(**inputs)[0]
        scores = torch.sigmoid(out).cpu().detach().numpy()
        results = {}
        for i, cla in enumerate(self.class_names):
            results[cla] = (
                scores[0][i]
                if isinstance(text, str)
                else [scores[ex_i][i].tolist() for ex_i in range(len(scores))]
            )
        return results


def toxic_bert():
    return load_model("original")


def unbiased_toxic_roberta():
    return load_model("unbiased")


def multilingual_toxic_xlm_r():
    return load_model("multilingual")
