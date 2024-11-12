import argparse
import pytorch_lightning as L
from mlopsp2.datamodule import GLUEDataModule
from mlopsp2.transformer import GLUETransformer
from mlopsp2.trainer import get_trainer
from lightning.pytorch.loggers import WandbLogger


def main():
    args = parse_args()

    L.seed_everything(42)

    run_name = (
        f"{args.task_name}_"
        + f"lr={args.learning_rate}_"
        + f"opt={args.optimizer}_"
        + f"bs={args.train_batch_size}_"
        + f"wd={args.weight_decay}_"
        + f"ws={args.warmup_steps}_"
        + f"lrs={args.lr_scheduler}"

    dm = GLUEDataModule(
        model_name_or_path=args.model_name_or_path,
        task_name=args.task_name,
        train_batch_size=args.train_batch_size,
        eval_batch_size=args.eval_batch_size,
    )
    dm.setup("fit")

    model = GLUETransformer(
        model_name_or_path=args.model_name_or_path,
        num_labels=dm.num_labels,
        task_name=args.task_name,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        weight_decay=args.weight_decay,
        train_batch_size=args.train_batch_size,
        eval_batch_size=args.eval_batch_size,
    )

    logger = WandbLogger()

    trainer = get_trainer(args.epochs, logger, "checkpoints", "run")

    
    


def parse_args():
    parser = argparse.ArgumentParser(description="Train a GLUE model with PyTorch Lightning")

    # Add hyperparameter arguments
    parser.add_argument('--epochs', type=int, default=3, help='Number of epochs for training')
    parser.add_argument('--learning_rate', type=float, default=2e-5, help='Learning rate')
    parser.add_argument('--warmup_steps', type=int, default=0, help='Number of warmup steps')
    parser.add_argument('--weight_decay', type=float, default=0.0, help='Weight decay')
    parser.add_argument('--train_batch_size', type=int, default=32, help='Training batch size')
    parser.add_argument('--eval_batch_size', type=int, default=64, help='Evaluation batch size')
    parser.add_argument('--task_name', type=str, default='mrpc', help='GLUE task name')
    parser.add_argument('--model_name_or_path', type=str, default='distilbert-base-uncased', help='Pretrained model path or name')

    # WandB related arguments
    parser.add_argument('--wandb_project', type=str, default='mlops-project', help='WandB project name')
    parser.add_argument('--wandb_entity', type=str, default='nicola-hermann-hochschule-luzern', help='WandB entity/username')

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    main()