import argparse
import random
import numpy as np
import torch
import torch.optim as optim
import lightning.pytorch as pl
from lightning.pytorch.loggers import WandbLogger

import datasets

from models.vit import ViT, MViT, resize_positional_embedding_
from models.vision_transformers import VisionTransformers


TARGET_TASK_MAP = {
    "dfc": "multi-label",
}

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def parse_args():
    parser = argparse.ArgumentParser()

    #  Parameters
    parser.add_argument(
        "--train_batch_size",
        default=32,
        type=int,
        help="Total batch size for training.",
    )
    parser.add_argument(
        "--eval_batch_size", default=32, type=int, help="Total batch size for eval."
    )
    parser.add_argument("--num_devices", default=1, type=int, help="Number of GPU")
    parser.add_argument("--device", default="gpu", type=str, help="gpu or cpu")
    parser.add_argument(
        "--learning_rate",
        default=3e-2,
        type=float,
        help="The initial learning rate for SGD.",
    )
    parser.add_argument(
        "--num_epochs",
        default=20,
        type=int,
        help="Total number of training epochs to perform.",
    )
    parser.add_argument("--depth", default=12, type=int, help="Total number of blocks.")
    parser.add_argument("--patch_size", default=12, type=int, help="Patch Size.")
    parser.add_argument("--num_classes", default=8, type=int, help="Number of classes")
    parser.add_argument(
        "--num_classes_ft", default=8, type=int, help="Number of classes finetuned"
    )
    parser.add_argument("--num_heads", default=16, type=int, help="Number of heads")
    parser.add_argument(
        "--accumulate_grad",
        default=1,
        type=int,
        help="Accumulate gradient every N batches",
    )
    parser.add_argument("--seed", default=42, type=int, help="Seed")
    parser.add_argument(
        "--num_channels", default=13, type=int, help="Number of channels"
    )
    parser.add_argument("--dataset", default="dfc", type=str, help="Dataset Selected")
    parser.add_argument("--opt", default="adamw", type=str, help="Optimizer")
    parser.add_argument("--mult", default=1, type=int, help="Multiplication Factor")
    parser.add_argument("--warmup", default=10, type=int, help="Warmup epochs")
    parser.add_argument("--pr_rate", default=0.2, type=float, help="Pruning coef")
    parser.add_argument(
        "--arch",
        default="vit",
        type=str,
        help="Architecture desired - Vit, DeepViT etc.",
    )
    parser.add_argument("--imgsize", nargs="+", type=int)
    parser.add_argument("--prune", action="store_true")
    parser.add_argument("--multimodal", action="store_true")
    parser.add_argument("--lr_scheduler", action="store_true")
    parser.add_argument("--pretrained", action="store_true")

    return parser.parse_args()


def setup_logger(
    dataset,
    arch,
    patch_size,
    batch_size,
    depth,
    learning_rate,
    prune,
    pr_rate,
):
    # initialize weightts and biases
    name = "{}/{}_arch_{}_patch_{}_bs_{}_depth_{}_lr_{}_prune_{}_pr_rate_{}".format(
        dataset,
        dataset,
        arch,
        patch_size,
        batch_size,
        depth,
        learning_rate,
        prune,
        pr_rate,
    )
    wandb = WandbLogger(project="ViT_pruning", entity="", name=name)
    return wandb


def setup_model(
    arch,
    imgsize,
    num_classes,
    depth,
    patch_size,
    num_heads,
    prune,
    num_channels,
    pr_rate,
    pretrained,
    multimodal,
):
    if arch == "vit":
        if multimodal:
            model = MViT(
                image_size=tuple(imgsize),
                patch_size=patch_size,
                num_classes=num_classes,
                dim=1024,
                depth=depth,
                heads=num_heads,
                mlp_dim=2048,
                device=device,
                m1_channels=2,
                m2_channels=13,
                prune=prune,
                dropout=0.0,
                emb_dropout=0.0,
                l0_penalty=pr_rate,
                multimodal=multimodal,
            )
        else:
            model = ViT(
                image_size=tuple(imgsize),
                patch_size=patch_size,
                num_classes=num_classes,
                dim=1024,
                depth=depth,
                heads=num_heads,
                mlp_dim=2048,
                device=device,
                prune=prune,
                channels=num_channels,
                dropout=0.0,
                emb_dropout=0.0,
                l0_penalty=pr_rate,
                multimodal=multimodal,
            )
    else:
        print("Architecture not yet supported")

    if pretrained:
        state_dict = torch.load(
            "checkpoints/pretrained_model.pth",
            map_location="cpu",
        )
        posemb = state_dict["pos_embedding"]
        posemb_new = model.state_dict()["pos_embedding"]
        state_dict["pos_embedding"] = resize_positional_embedding_(
            posemb=posemb, posemb_new=posemb_new
        )
        # Modifications to load partial state dict
        expected_missing_keys = []
        expected_missing_keys += [
            "to_patch_embedding.1.weight",
            "to_patch_embedding.1.bias",
        ]
        expected_missing_keys += ["mlp_head.1.weight", "mlp_head.1.bias"]

        for key in expected_missing_keys:
            state_dict.pop(key)

        model.load_state_dict(state_dict, strict=False)

    return model


def create_dataset(dataset, mult, imgsize, train_batch_size, eval_batch_size):
    if dataset == "dfc":
        data_module = datasets.dfc.DFCDataModule()
        data_module.setup(mult, train_batch_size, eval_batch_size)
    elif dataset == "cityscapes":
        data_module = datasets.cityscapes.CityscapesDataModule()
        data_module.setup(mult, imgsize, train_batch_size, eval_batch_size)
    else:
        print("The dataset doesn't exist.")
        return

    return data_module


def setup_criterion_optimizer_scheduler(
    dataset, opt, learning_rate, lr_scheduler, model
):
    if TARGET_TASK_MAP[dataset] == "single-label":
        criterion = torch.nn.CrossEntropyLoss().to(device)
    elif TARGET_TASK_MAP[dataset] == "multi-label":
        criterion = torch.nn.BCELoss(reduction="mean").to(device)
    else:
        raise ValueError("Invalid target specified")

    if opt == "adam":
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    elif opt == "adamw":
        optimizer = optim.AdamW(model.parameters(), betas=(0.9, 0.95), lr=learning_rate)
    elif opt == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    else:
        raise ValueError("Invalid optimizer specified")

    scheduler = None
    if lr_scheduler:
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=300)

    return criterion, optimizer, scheduler


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main():
    args = parse_args()

    pl.seed_everything(args.seed)
    # set_seed(args.seed)

    # Setup wandb logger
    wandb_logger = setup_logger(
        args.dataset,
        args.arch,
        args.patch_size,
        args.train_batch_size,
        args.depth,
        args.learning_rate,
        args.prune,
        args.pr_rate,
    )

    model = setup_model(
        args.arch,
        args.imgsize,
        args.num_classes,
        args.depth,
        args.patch_size,
        args.num_heads,
        args.prune,
        args.num_channels,
        args.pr_rate,
        args.pretrained,
        args.multimodal,
    )

    criterion, optimizer, scheduler = setup_criterion_optimizer_scheduler(
        args.dataset, args.opt, args.learning_rate, args.lr_scheduler, model
    )

    model_module = VisionTransformers(
        model, criterion, optimizer, scheduler, args.prune
    )

    data_module = create_dataset(
        args.dataset,
        args.mult,
        args.imgsize,
        args.train_batch_size,
        args.eval_batch_size,
    )

    trainer = pl.Trainer(
        accelerator=args.device,
        devices=args.num_devices,
        max_epochs=args.num_epochs,
        log_every_n_steps=args.accumulate_grad,
        accumulate_grad_batches=args.accumulate_grad,
        logger=wandb_logger,
    )

    if trainer.global_rank == 0:
        wandb_logger.experiment.config.update(
            {
                "learning_rate": args.learning_rate,
                "epochs": args.num_epochs,
                "batch_size": args.train_batch_size,
                "depth": args.depth,
                "patch_size": args.patch_size,
                "arch": args.arch,
            }
        )

    trainer.fit(
        model_module, data_module.train_dataloader(), data_module.val_dataloader()
    )


if __name__ == "__main__":
    main()
