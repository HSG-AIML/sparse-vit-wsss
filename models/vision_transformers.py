import lightning.pytorch as pl
import torch


class VisionTransformers(pl.LightningModule):
    def __init__(
        self, model, criterion, optimizer, scheduler, prune=False, lr_scheduler=False
    ):
        super().__init__()
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.prune = prune
        self.lr_scheduler = lr_scheduler

    def forward(self, img):
        return self.model(img)

    def configure_optimizers(self):
        if self.lr_scheduler:
            lr_scheduler_config = {
                "scheduler": self.scheduler,
                "interval": "epoch",
                "frequency": 1,
                "monitor": "val_loss",
                "strict": True,
                "name": None,
            }
            return {"optimizer": self.optimizer, "lr_scheduler": lr_scheduler_config}
        else:
            return self.optimizer

    def loss_function(self, logits, labels):
        return self.criterion(logits, labels)

    def get_penalty(self):
        penalty, sparsity_rate = 0, 0
        for layer_idx in range(self.model.depth):
            penalty += self.model.transformer.layers[layer_idx][0].fn.gate.get_penalty()
            sparsity_rate += self.model.transformer.layers[layer_idx][
                0
            ].fn.gate.get_sparsity_rate()
        return penalty, sparsity_rate

    def training_step(self, train_batch, batch_idx):
        y = train_batch["y"].float()
        if self.model.multimodal:
            s1 = train_batch["s1"].float()
            s2 = train_batch["img"].float()
            output = self.forward(s1, s2)
        else:
            data = train_batch["img"].float()
            output = self.forward(data)

        if self.prune:
            penalty, sparsity_rate = self.get_penalty()
            train_loss = self.loss_function(output, y) + penalty
        else:
            train_loss = self.loss_function(output, y)

        predicted = torch.round(output)
        correct = (predicted == y).sum().item()

        train_acc = correct / torch.numel(predicted)

        stats = {"train_loss": train_loss, "train_accuracy": train_acc}

        if self.prune:
            stats["sparsity_rate"] = sparsity_rate

        self.log_dict(stats, on_step=False, on_epoch=True, batch_size=2, sync_dist=True)

        return train_loss

    def validation_step(self, val_batch, batch_idx):
        y = val_batch["y"].float()
        if self.model.multimodal:
            s1 = val_batch["s1"].float()
            s2 = val_batch["img"].float()
            output = self.forward(s1, s2)
        else:
            data = val_batch["img"].float()
            output = self.forward(data)

        val_loss = self.loss_function(output, y)

        predicted = torch.round(output)
        correct = (predicted == y).sum().item()

        val_acc = correct / torch.numel(predicted)

        stats = {"val_loss": val_loss, "val_accuracy": val_acc}

        self.log_dict(stats, on_step=False, on_epoch=True, batch_size=2, sync_dist=True)
