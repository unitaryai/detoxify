from pytorch_lightning import Trainer, seed_everything

from project.lit_classifier_main import LitClassifier


def test_lit_classifier():
    seed_everything(1234)

    model = LitClassifier()
    train, val, test = mnist()
    trainer = Trainer(limit_train_batches=50, limit_val_batches=20, max_epochs=2)
    trainer.fit(model, train, val)

    results = trainer.test(test_dataloaders=test)
    assert results[0]['test_acc'] > 0.7
