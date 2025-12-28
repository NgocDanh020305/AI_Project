import torch
import torch.nn as nn


class CRNN(nn.Module):
    def __init__(self, img_height, nc, n_class, nh):
        """
        img_height: chiều cao ảnh input (32)
        nc: số kênh input (1 cho grayscale)
        n_class: số lượng ký tự output (vocab size + 1 blank)
        nh: số hidden units của RNN (256)
        """
        super(CRNN, self).__init__()

        # --- CNN Layers ---
        ks = [3, 3, 3, 3, 3, 3, 2]  # kernel sizes
        ps = [1, 1, 1, 1, 1, 1, 0]  # paddings
        ss = [1, 1, 1, 1, 1, 1, 1]  # strides
        nm = [64, 128, 256, 256, 512, 512, 512]  # number of feature maps

        cnn = nn.Sequential()

        def convRelu(i, batchNormalization=False):
            nIn = nc if i == 0 else nm[i - 1]
            nOut = nm[i]
            cnn.add_module('conv{0}'.format(i),
                           nn.Conv2d(nIn, nOut, ks[i], ss[i], ps[i]))
            if batchNormalization:
                cnn.add_module('batchnorm{0}'.format(i), nn.BatchNorm2d(nOut))
            cnn.add_module('relu{0}'.format(i), nn.ReLU(True))

        # Layer 0, 1
        convRelu(0)
        cnn.add_module('pooling{0}'.format(0), nn.MaxPool2d(2, 2))  # 64x16x64
        convRelu(1)
        cnn.add_module('pooling{0}'.format(1), nn.MaxPool2d(2, 2))  # 128x8x32

        # Layer 2, 3
        convRelu(2, True)
        convRelu(3)
        # Pooling width ít hơn để giữ time-steps
        cnn.add_module('pooling{0}'.format(2),
                       nn.MaxPool2d((2, 2), (2, 1), (0, 1)))  # 256x4x33

        # Layer 4, 5
        convRelu(4, True)
        convRelu(5)
        cnn.add_module('pooling{0}'.format(3),
                       nn.MaxPool2d((2, 2), (2, 1), (0, 1)))  # 512x2x34

        # Layer 6
        convRelu(6, True)  # 512x1x33

        self.cnn = cnn

        # --- RNN Layers ---
        self.rnn = nn.Sequential(
            nn.LSTM(512, nh, bidirectional=True, batch_first=False),
        )
        self.embedding = nn.Linear(nh * 2, n_class)  # *2 vì bidirectional

    def forward(self, input):
        # Input: (B, 1, 32, 128)
        conv = self.cnn(input)
        # Output CNN: (B, 512, 1, W_new)

        b, c, h, w = conv.size()
        assert h == 1, "Chiều cao của conv output phải là 1"

        conv = conv.squeeze(2)  # (B, 512, W_new)
        conv = conv.permute(2, 0, 1)  # (W_new, B, 512) - RNN cần time-step đầu tiên

        # RNN
        # recurrent output: (W_new, B, nh*2)
        recurrent, _ = self.rnn(conv)

        T, b, h = recurrent.size()
        t_rec = recurrent.view(T * b, h)

        # Map sang vocab classes
        output = self.embedding(t_rec)  # [T * b, n_class]
        output = output.view(T, b, -1)  # [T, b, n_class]

        return output