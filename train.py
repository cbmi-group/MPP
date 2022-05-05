import argparse
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from Dataset import MyDataset, SCM_Dataset
from model.unet import UNet
from utils import DiceLoss

class Experiment(pl.LightningModule):

    def __init__(self, model, args):
        super().__init__()
        self.model = model
        self.loss_function = DiceLoss()
        self.lr = args.lr

    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        
        frames, masks = batch
        frames, masks = frames.cuda(), masks.cuda()
        loss = 0
        logits = self.model(frames)
        loss = self.loss_function(logits, masks)
        self.log('train_loss', loss, on_step=True, on_epoch=True,prog_bar=True, logger=True)

        return loss

    def configure_optimizers(self):
        optim = torch.optim.RMSprop(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=self.lr,
            alpha=0.99,
            eps=1e-08,
            weight_decay=0,
            momentum=0,
            centered=False
            )
        return optim

#Both supervised and unsurpervised training can use it, just need to change the dataset
parser = argparse.ArgumentParser(description='Training Script')

parser.add_argument('--stage', default='semi-supervised',
                    help='training stage (supervised or semi-supervised)')
parser.add_argument('--epochs', default=100, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--batch-size', default=1, type=int,
                    help='mini-batch size')
parser.add_argument('--lr', default=5e-4, type=float,
                    help='learning rate', dest='lr')

# model configs:
parser.add_argument('--scm-channels', default=15, type=int,
                    help='input channels for SCM network')

args = parser.parse_args()
if args.stage == 'supervised':
    train_dataset = MyDataset()
    model = UNet(1, 1)
    logger_name = 'UNet'
if args.stage == 'semi-supervised':
    train_dataset = SCM_Dataset()
    model = UNet(args.scm_channels, 1)
    logger_name = 'UNet_SCM'

train_loader = torch.utils.data.DataLoader(train_dataset,
                                            batch_size=args.batch_size, 
                                            shuffle=True, 
                                            num_workers=16, 
                                            pin_memory=True)

logger = TensorBoardLogger('tb_logs', name=logger_name)

unet_experiment = Experiment(model, args)

checkpoint_callback = ModelCheckpoint(monitor='train_loss',
                                        save_top_k=10,
                                        mode='min'
                                        )

trainer = pl.Trainer(max_epochs=args.epochs, 
                        gpus=[0,1], 
                        accelerator= 'ddp',
                        logger = logger,
                        callbacks = [checkpoint_callback],
                        precision=16
                        )

trainer.fit(unet_experiment, train_loader, train_loader)