from lightning.pytorch.callbacks import ModelCheckpoint
import lightning as L

def get_trainer(epochs:int, wandb_logger, checkpoint_path:str, wandb_run_name:str):
    model_checkpoint = ModelCheckpoint(
        dirpath=checkpoint_path,
        filename=wandb_run_name,
        save_top_k=1,
        verbose=False,
        monitor="val_loss",
        mode="min",
        save_weights_only=True,
    )

    return L.Trainer(
        max_epochs=epochs,
        accelerator="auto",
        devices=1,
        logger=wandb_logger,
        callbacks=[model_checkpoint],
    )