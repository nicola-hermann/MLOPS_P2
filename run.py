from mlopsp2.datamodule import GLUEDataModule
from mlopsp2.transformer import GLUETransformer
import pytorch_lightning as L

epochs = 3  # do not change this
logger = False  # use your experiment tracking tool's logger

L.seed_everything(42)

dm = GLUEDataModule(
    model_name_or_path="distilbert-base-uncased",
    task_name="mrpc",
)
dm.setup("fit")
model = GLUETransformer(
    model_name_or_path="distilbert-base-uncased",
    num_labels=dm.num_labels,
    eval_splits=dm.eval_splits,
    task_name=dm.task_name,
)

trainer = L.Trainer(
    max_epochs=epochs,
    accelerator="auto",
    devices=1,
    logger=logger,
)
trainer.fit(model, datamodule=dm)
