import argparse
import pytorch_lightning as L
from mlopsp2.datamodule import GLUEDataModule
from mlopsp2.transformer import GLUETransformer
from mlopsp2.trainer import get_trainer
from lightning.pytorch.loggers import WandbLogger
import wandb
from dotenv import load_dotenv
import os

load_dotenv(override=False)


def main():
    args = parse_args()

    wandb.login(key=os.getenv("WANDB_API_KEY"), relogin=True)

    print("Login successful")

    L.seed_everything(args.seed)

    run_name = (
        f"mrpc_"
        + f"lr-{args.learning_rate}_"
        + f"opt-{args.optimizer}_"
        + f"bs-{args.train_batch_size}_"
        + f"wd-{args.weight_decay}_"
        + f"ws-{args.warmup_steps}_"
        + f"lrs-{args.lr_scheduler}"
    )

    dm = GLUEDataModule(
        model_name_or_path="distilbert-base-uncased",
        task_name="mrpc",
        train_batch_size=args.train_batch_size,
        eval_batch_size=args.eval_batch_size,
    )
    dm.setup("fit")

    model = GLUETransformer(
        model_name_or_path="distilbert-base-uncased",
        num_labels=dm.num_labels,
        task_name="mrpc",
        learning_rate=args.learning_rate,
        lr_scheduler=args.lr_scheduler,
        warmup_steps=args.warmup_steps,
        weight_decay=args.weight_decay,
        train_batch_size=args.train_batch_size,
        eval_batch_size=args.eval_batch_size,
        optimizer=args.optimizer,
    )

    with wandb.init(
        project=args.wandb_project, entity=args.wandb_entity, name=run_name, config=vars(args)
    ):
        logger = WandbLogger()
        trainer = get_trainer(wandb.config.epochs, logger, wandb.config.save_path, run_name)
        trainer.fit(model=model, datamodule=dm)
        wandb.log_model(f"{wandb.config.save_path}/{run_name}.ckpt", name=run_name)
        wandb.finish()


def parse_args():
    parser = argparse.ArgumentParser(description="Train a GLUE model with PyTorch Lightning")

    # Add hyperparameter arguments
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs for training")
    parser.add_argument("--learning_rate", type=float, default=4e-5, help="Learning rate")
    parser.add_argument("--optimizer", type=str, default="AdamW", help="Optimizer (AdamW, SGD)")
    parser.add_argument("--warmup_steps", type=int, default=30, help="Number of warmup steps")
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay")
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="linear",
        help="Learning rate scheduler (linear, cosine, constant)",
    )
    parser.add_argument("--train_batch_size", type=int, default=64, help="Training batch size")
    parser.add_argument("--eval_batch_size", type=int, default=64, help="Evaluation batch size")
    parser.add_argument("--save_path", type=str, default="checkpoints", help="Model Save Path")
    parser.add_argument("--seed", type=int, default=31, help="Seed for reproducibility")

    # WANB arguments
    parser.add_argument("--wandb_project", type=str, default="mlops-project", help="Wandb project")
    parser.add_argument(
        "--wandb_entity",
        type=str,
        default="nicola-hermann-hochschule-luzern",
        help="Team or user name",
    )

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    main()
