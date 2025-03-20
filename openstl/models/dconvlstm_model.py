import torch
import torch.nn as nn

from openstl.modules.dconvlstm_modules import STconvertWithFusion


class DConvLSTM_Model(nn.Module):
    def __init__(self, configs, **kwargs):
        super(DConvLSTM_Model, self).__init__()
        T, C, H, W = configs.in_shape
        # assert H == W, 'Only support H = W for image input'
        self.configs = configs
        self.ST = STconvertWithFusion(img_size=(H, W), patch_size=configs.patch_size, in_chans=C,
                            embed_dim=configs.embed_dim, depths=configs.depths,
                            num_heads=configs.num_heads, window_size=configs.window_size)
        self.MSE_criterion = nn.MSELoss()

    def forward(self, frames_tensor, **kwargs):
        # [batch, length, height, width, channel] -> [batch, length, channel, height, width]
        T, C, H, W = self.configs.in_shape
        total_T = frames_tensor.shape[1]
        frames = frames_tensor.permute(0, 1, 4, 2, 3).contiguous()

        input_frames = frames[:, :T]
        states = None
        next_frames = []
        last_frame = input_frames[:, -1]

        for i in range(T - 1):
            output, states = self.ST(input_frames[:, i], states)
            next_frames.append(output)
        for i in range(total_T - T):
            output, states = self.ST(last_frame, states)
            next_frames.append(output)
            last_frame = output

        # [length, batch, channel, height, width] -> [batch, length, height, width, channel]
        next_frames = torch.stack(next_frames, dim=0).permute(1, 0, 3, 4, 2).contiguous()
        if kwargs.get('return_loss', True):
            loss = self.MSE_criterion(next_frames, frames_tensor[:, 1:])
        else:
            loss = None

        return next_frames, loss