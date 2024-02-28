#!/usr/bin/env python
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
import wandb
import argparse
from helper_x3d import MilestonesFinetuning, VideoModel, VideoDataModule


if __name__ =='__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=2)
    parser.add_argument('--num-workers', type=int, default=0)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--milestone', type=int, default=5)
    parser.add_argument('--backbone', type=str, default="x3d_m")
    parser.add_argument('--resize', type=int, default=1)
    parser.add_argument('--unfreeze_top_layers', type=int, default=3)
    parser.add_argument('-t', '--train_csv_path', type=str, default='/home/mkhoatd/repos/ai4life2/ai4life/data_csv_2/train_info.csv')
    parser.add_argument('-v', '--val_csv_path', type=str, default='/home/mkhoatd/repos/ai4life2/ai4life/data_csv_2/val_info.csv')
    parser.add_argument('-r', '--root_data_dir', type=str, default='/home/mkhoatd/repos/ai4life2/ai4life/data_stable')
    parser.add_argument('-o', '--output_data_dir', type=str, default='./output')

    args = parser.parse_args()
    print(dir(args))

    config = {"lr": args.lr,
              "batch_size": args.batch_size,
              "num_workers": args.num_workers,
              "milestone": args.milestone,
              "epochs": args.epochs,
              "backbone": args.backbone,
              "resize": args.resize,
              "unfreeze_top_layers": args.unfreeze_top_layers,
              "train_csv_path": args.train_csv_path,
              "val_csv_path": args.val_csv_path,
              "root_data_dir": args.root_data_dir
              }

    finetuning_callback = MilestonesFinetuning(milestone=config["milestone"], unfreeze_top_layers=config["unfreeze_top_layers"], train_bn=False)
    # wandb_key = "7cef42da986b9a35aabf18181bc73a867a875b8f"
    # wandb.login(key=wandb_key)
    wandb_logger = WandbLogger(project='test-x3d', offline=False, name=config["backbone"], config=config)

    trainer = pl.Trainer(
                        max_epochs=config["epochs"], 
                        callbacks=[finetuning_callback],
                        log_every_n_steps=25,
                        precision="16-mixed",
                        default_root_dir=args.output_data_dir,
                        logger=wandb_logger,
                        )

    model = VideoModel(backbone=config["backbone"], 
                         lr=config["lr"], 
                         milestone=config["milestone"])

    data = VideoDataModule(train_csv_path=config["train_csv_path"],
                           val_csv_path=config["val_csv_path"],
                            backbone=config["backbone"], 
                            batch_size=config["batch_size"], 
                            num_workers=config["num_workers"], 
                            resize=config["resize"],
                            root_data_dir=config["root_data_dir"])
    
    trainer.fit(model, data)
    