import torch

from .base_method import Base_method
from ..models.dconvlstm_model import DConvLSTM_Model


class DConvLSTM(Base_method):
    r"""SwinLSTM
    Implementation of `SwinLSTM: Improving Spatiotemporal Prediction Accuracy using Swin
    Transformer and LSTM <http://arxiv.org/abs/2308.09891>`_.

    """

    def __init__(self, **args):
        super().__init__(**args)

    def _build_model(self, **args):
        return DConvLSTM_Model(self.hparams)

    def forward(self, batch_x, batch_y, **kwargs):
        """Forward the model"""
        # preprocess
        test_ims = torch.cat([batch_x, batch_y], dim=1).permute(0, 1, 3, 4, 2).contiguous()

        img_gen, _ = self.model(test_ims, return_loss=False)
        pred_y = img_gen[:, -self.hparams.aft_seq_length:].permute(0, 1, 4, 2, 3).contiguous()

        return pred_y

    def training_step(self, batch, batch_idx):
        batch_x, batch_y = batch
        ims = torch.cat([batch_x, batch_y], dim=1).permute(0, 1, 3, 4, 2).contiguous()

        img_gen, loss = self.model(ims)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss