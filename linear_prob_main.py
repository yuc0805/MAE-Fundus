import argparse
from typing import Any, Tuple, Union
from model import *
from datamodule import FundusDataModule


class MAE_linear_probe(pl.LightningModule):
    '''Frozen MAE encoder with trainable linear readout to class labels
    https://lightning.ai/docs/pytorch/stable/advanced/transfer_learning.html

    '''
    def __init__(
            self, 
            ckpt_path: str,
            device = 'cuda',
            num_class=5,
            ):
        super().__init__()
        mae_module = MAE()
        ckpt = torch.load(ckpt_path,map_location=device)['state_dict']
        mae_module.load_state_dict(ckpt)
        #self.mae = mae_module.mae

        self.feature_extractor = mae_module.backbone

        # Freeze the feature extractor
        for param in self.feature_extractor.parameters():
            param.requires_grad = False
        self.feature_extractor.eval()

        self.classifier = torch.nn.Linear(768, num_class)
        self.classifier.weight.data.normal_(mean=0.0, std=0.01)
        self.classifier.bias.data.zero_()

    def forward(self, x):
        #x = self.mae.embed(x)
        #x = x + self.mae.pos_encoder
        self.feature_extractor.eval()
        with torch.no_grad():
            x = self.feature_extractor.encode(x)
            x = x.mean(dim=1)  # average pool over the patch dimension
        x = self.classifier(x)
        return x
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-3)
        return optimizer

    def training_step(self, batch: Any, batch_idx: int, *args, **kwargs):
        x, labels = batch
        if isinstance(x, list):
            x = torch.stack(x).squeeze(0)
            #print('shape of revise x:',x.shape)
        if isinstance(labels, list):
            labels = torch.tensor(labels)


        pred = self.forward(x)
        loss = self.loss_fn(pred, labels)

        _, predicted = torch.max(pred, 1)
        correct = (predicted == labels).sum().item()

        self.log('train/loss', loss, prog_bar=True, sync_dist=True, on_step=False, on_epoch=True)
        self.log('val/acc', correct / len(labels), prog_bar=True, on_step=False, sync_dist=True, on_epoch=True)
        return {'loss': loss}
    
    def validation_step(self, batch: Any, batch_idx: int, *args, **kwargs):
        x, labels = batch

        if isinstance(x, list):
            x = torch.stack(x).squeeze(0)
            #print('shape of revise x:',x.shape)
        if isinstance(labels, list):
            labels = torch.tensor(labels)

        pred = self.forward(x)
        loss = self.loss_fn(pred, labels)
        _, predicted = torch.max(pred, 1)
        correct = (predicted == labels).sum().item()
        self.log('val/loss', loss, prog_bar=True, sync_dist=True, on_step=False, on_epoch=True)
        self.log('val/acc', correct / len(labels), prog_bar=True, on_step=False, sync_dist=True, on_epoch=True)

    def test_step(self, batch: Any, batch_idx: int, *args, **kwargs):
        x, labels = batch

        if isinstance(x, list):
            x = torch.stack(x).squeeze(0)
            #print('shape of revise x:',x.shape)
        if isinstance(labels, list):
            labels = torch.tensor(labels)
            
        pred = self.forward(x)
        loss = self.loss_fn(pred, labels)
        _, predicted = torch.max(pred, 1)
        correct = (predicted == labels).sum().item()
        self.log('test/loss', loss, prog_bar=True, sync_dist=True, on_step=False, on_epoch=True)
        self.log('test/acc', correct / len(labels), prog_bar=True, on_step=False, sync_dist=True, on_epoch=True)

    def loss_fn(self, x, y):
        fn = torch.nn.CrossEntropyLoss()
        return fn(x, y)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train and test MAE linear probe')
    parser.add_argument('--ckpt_path', type=str, default='lightning_logs/epoch=8-step=32418.ckpt', help='Path to the checkpoint file')
    parser.add_argument('--max_epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--device', type=str, default='cuda', help='cuda/cpu')
    parser.add_argument('--data_dir', type=str, default= '/Users/leo/Desktop/MAE-Fundus/data/EAMDR', 
                        help='Path to the data directory')
    parser.add_argument('--num_class', type=int, default=5, help='Number of classes')

    args = parser.parse_args()
    
    torch.set_float32_matmul_precision('medium')
    # Create an instance of the model
    model = MAE_linear_probe(ckpt_path=args.ckpt_path,
                             device=args.device,
                             num_class=args.num_class)
    
    transform = MAETransform()
    data_module = FundusDataModule(args.data_dir,transform=transform)
    data_module.setup(stage='fit')

    trainer = pl.Trainer(max_epochs=args.max_epochs,
                         devices="auto")

    val_loader_as_train_loader = data_module.val_dataloader()
    trainer.fit(model, train_dataloaders=val_loader_as_train_loader)
    trainer.test(model, dataloaders=data_module.test_dataloader())
# python -m linear_prob_main --device cpu 