import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils import data
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

#! Convolutional block
class VSConvBlock(nn.Module):
    """"""
    def __init__(self, in_ch, out_ch, activation,
                 dilation=1, padding=0, kernel_size=15, pad_type = "rp"):
        """"""
        super(VSConvBlock, self).__init__()
        self.padding = padding

        self.padding_layer = nn.ReflectionPad1d(self.padding) if pad_type == "rp" else nn.ConstantPad1d(self.padding, 0)
        self.conv = nn.Conv1d(in_ch, out_ch,
                              kernel_size=kernel_size,
                              stride=1,
                              dilation=dilation)
        self.batch = nn.BatchNorm1d(out_ch)
        self.activation = activation

    def forward(self, x):
        # Block 1
        x = self.padding_layer(x)
        x = self.conv(x)
        # x = self.batch(x)
        x = self.activation(x)

        return x


class DownSamplingBlock(nn.Module):
    """Encoder block of the Fully-convolutional Network"""
    def __init__(self, in_ch, out_ch, activation,
                 padding=0, dilation=1, kernel_size=15, pad_type="rp"):
        super(DownSamplingBlock, self).__init__()
        self.block = VSConvBlock(in_ch, out_ch,
                                 activation=activation,
                                 kernel_size=kernel_size,
                                 padding=padding,
                                 dilation=dilation, 
                                 pad_type=pad_type)

    def forward(self, x):
        x = self.block(x)
        return x[:, :, ::2], x


#! Add Bi-LSTM layer to the Convolutional block
class Conv_BiLSTM_Block(nn.Module):
    """
    VSConvBlock with additional Bi-LSTM layer for temporal modeling.
    """
    def __init__(self, in_ch, out_ch, activation,
                 dilation=1, padding=0, kernel_size=15, pad_type="rp", lstm_hidden_size=64, use_bilstm=False, dropout=False):
        """
        :param in_ch: Number of input channels
        :param out_ch: Number of output channels
        :param activation: Activation function
        :param dilation: Dilation for the convolution layer
        :param padding: Padding size
        :param kernel_size: Kernel size for convolution
        :param pad_type: Type of padding ("rp" for reflection padding, otherwise constant padding)
        :param lstm_hidden_size: Hidden size for the Bi-LSTM layer
        :param use_bilstm: Whether to include Bi-LSTM after convolution
        """
        super(Conv_BiLSTM_Block, self).__init__()
        self.padding = padding
        self.use_bilstm = use_bilstm
        self.dropout = dropout

        # Padding layer
        self.padding_layer = nn.ReflectionPad1d(self.padding) if pad_type == "rp" else nn.ConstantPad1d(self.padding, 0)

        # Convolutional layer
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size=kernel_size, stride=1, dilation=dilation)

        # Batch normalization layer
        self.batch = nn.BatchNorm1d(out_ch)

        # Activation function
        self.activation = activation

        # (Bi-)LSTM layer
        lstm_hidden_dim = lstm_hidden_size
        lstm_output_dim = 2 * lstm_hidden_dim if use_bilstm else lstm_hidden_dim
        if self.use_bilstm:
            self.lstm = nn.LSTM(out_ch, lstm_hidden_dim, bidirectional=True, batch_first=True)
        else:
            self.lstm = nn.LSTM(out_ch, lstm_hidden_dim, bidirectional=False, batch_first=True)
            
        # Dropout layer
        self.dropout_fun = nn.Dropout(p=0.3)

        # Fully connected layer to adjust LSTM output dimension to convolution output dimension
        self.lstm_fc = nn.Linear(lstm_output_dim, out_ch)

    def forward(self, x):
        # Padding and convolution
        x = self.padding_layer(x)
        x = self.conv(x)
        # x = self.batch(x)  # Uncomment if batch normalization is needed
        x = self.activation(x)
        
        # Prepare for Bi-LSTM (batch, channel, time) -> (batch, time, channel)
        x = x.permute(0, 2, 1)
        x, _ = self.lstm(x)  # Apply (Bi-)LSTM
        x = self.activation(x)  # Activation function
        
        if self.dropout:
            x = self.dropout_fun(x)
        
        x = self.lstm_fc(x)  # Fully connected layer to match output channels
        x = x.permute(0, 2, 1)  # Back to (batch, channel, time)

        return x


class DownSampling_BiLSTM_Block(nn.Module):
    """Encoder block of the Fully-convolutional Network"""
    def __init__(self, in_ch, out_ch, activation,
                 padding=0, dilation=1, kernel_size=15, pad_type="rp", lstm_hidden_size=64, use_bilstm=False, dropout=False):
        super(DownSampling_BiLSTM_Block, self).__init__()
        self.block = Conv_BiLSTM_Block(
            in_ch, out_ch,
            activation=activation,
            kernel_size=kernel_size,
            padding=padding,
            dilation=dilation,
            pad_type=pad_type,
            lstm_hidden_size=lstm_hidden_size,
            use_bilstm=use_bilstm,
            dropout=dropout
        )

    def forward(self, x):
        x = self.block(x)
        return x[:, :, ::2], x  # Downsampling by taking every second time step


class UpSamplingBlock(nn.Module):
    """Decoder block of the Fully-convolutional Network"""
    def __init__(self, in_ch, out_ch, activation,
                 padding=None, kernel_size=5, mode="linear", pad_type="rp", dropout=False):
        super(UpSamplingBlock, self).__init__()
        self.mode = mode
        self.padding = padding or (kernel_size // 2)
        self.dropout = dropout
        
        self.dropout_fun = nn.Dropout(p=0.3)

        # Convolution block
        self.conv = VSConvBlock(in_ch, out_ch,
                                activation=activation,
                                kernel_size=kernel_size,
                                padding=self.padding,
                                pad_type=pad_type)

        # Deconvolution block
        if not self.mode == "linear":
            self.deconv = nn.ConvTranspose1d(
                in_channels=in_ch - out_ch,
                out_channels=in_ch - out_ch,
                kernel_size=2,
                stride=2,
                padding=0,
                output_padding=0,
                bias=True,
                dilation=1
            )

            # self.deconv_activation = activation

    def forward(self, x, x_enc):
        if self.mode == "linear":
            x = F.interpolate(x, scale_factor=2,
                              mode='linear', align_corners=True)
        else:
            x = self.deconv(x)
            # x = self.deconv_activation(x)
        
        if self.dropout:
            x = self.dropout_fun(x)
        
        # Concat with Skip connection
        x = torch.cat([x, x_enc], dim=1)
        return self.conv(x)


class OutBlock(nn.Module):
    """Convolutional block similar to VSConvBlock.
    The network input is fed into this layer"""
    def __init__(self, in_ch, out_ch, activation, padding=0, pad_type="rp"):
        super(OutBlock, self).__init__()
        self.conv = VSConvBlock(in_ch, out_ch,
                                activation=activation,
                                kernel_size=1,
                                padding=padding, pad_type=pad_type)

    def forward(self, x, x_enc):
        x = torch.cat([x, x_enc], dim=1)
        return self.conv(x) - x_enc
