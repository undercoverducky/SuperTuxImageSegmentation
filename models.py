import torch
import torch.nn.functional as F
import torchvision

class CNNClassifier(torch.nn.Module):

    class Block(torch.nn.Module):
        def __init__(self, n_input, n_output, stride=1):
            super().__init__()
            self.block = torch.nn.Sequential(
                torch.nn.Conv2d(n_input, n_output, kernel_size=3, padding=1, stride=stride),
                torch.nn.BatchNorm2d(n_output),
                torch.nn.ReLU(),
                torch.nn.Conv2d(n_output, n_output, kernel_size=3, padding=1),
                torch.nn.BatchNorm2d(n_output),
                torch.nn.ReLU(),
                #torch.nn.MaxPool2d(kernel_size=2, stride=2)
            )
            self.downsample = None
            if stride != 1 or n_input != n_output:
                self.downsample = torch.nn.Sequential(torch.nn.Conv2d(n_input, n_output, 1, stride=stride),
                                                      torch.nn.BatchNorm2d(n_output))

        def forward(self, x):
            identity = x
            if self.downsample is not None:
                identity = self.downsample(x)
            return self.block(x) + identity

    def __init__(self, layers=None, n_input_channels=3, normalize_input = False):
        super().__init__()
        self.normalize_input = normalize_input
        if layers is None:
            layers = [32, 64, 128]
        L = [
            torch.nn.Conv2d(n_input_channels, 32, kernel_size=7, stride=2),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(),
            #torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        ]
        c = 32
        for ly in layers:
            L.append(self.Block(c, ly, stride=ly // c))
            c = ly
        self.conv_layers = torch.nn.Sequential(*L)
        self.dropout = torch.nn.Dropout(0.5)
        self.gap = torch.nn.AdaptiveAvgPool2d(1)
        self.linear = torch.nn.Linear(c, 6)

    def forward(self, x):
        if self.normalize_input:
            transform = torchvision.transforms.Normalize(mean=[0.4701, 0.4308, 0.3839], std=[0.2595, 0.2522, 0.2541])
            x = transform(x)
        x = self.conv_layers(x)
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.linear(x)
        return x


class FCN(torch.nn.Module):

    class UpConvBlock(torch.nn.Module):
        def __init__(self, n_input, n_output, stride=1):
            super().__init__()
            self.up_block = torch.nn.Sequential(
                #torch.nn.ConvTranspose2d(n_input, n_input, kernel_size=3, padding=1, stride=stride),
                torch.nn.BatchNorm2d(n_output),
                torch.nn.ReLU(),
                torch.nn.Conv2d(n_output, n_output , kernel_size=3, padding=1),
                torch.nn.BatchNorm2d(n_output),
                torch.nn.ReLU(),
                )
            self.up_conv = torch.nn.ConvTranspose2d(n_input, n_output, kernel_size=3, padding=1, stride=stride)
            self.batch_norm = torch.nn.BatchNorm2d(n_output)
            self.relu = torch.nn.ReLU()
            if stride != 1 or n_input != n_output:
                self.onebyone = torch.nn.ConvTranspose2d(n_input, n_output, 1, stride=stride)

        def forward(self, x, output_size):
            identity = x
            if self.onebyone is not None:
                identity = self.batch_norm(self.onebyone(identity, output_size=output_size))
            x = self.up_conv(x, output_size=output_size)
            x = self.up_block(x)
            return x + identity

    class ConvBlock(torch.nn.Module):
        def __init__(self, n_input, n_output, stride=1):
            super().__init__()
            self.block = torch.nn.Sequential(
                torch.nn.Conv2d(n_input, n_output, kernel_size=3, padding=1, stride=stride),
                torch.nn.BatchNorm2d(n_output),
                torch.nn.ReLU(),
                torch.nn.Conv2d(n_output, n_output, kernel_size=3, padding=1),
                torch.nn.BatchNorm2d(n_output),
                torch.nn.ReLU(),
            )
            self.downsample = None
            if stride != 1 or n_input != n_output:
                self.downsample = torch.nn.Sequential(torch.nn.Conv2d(n_input, n_output, 1, stride=stride),
                                                      torch.nn.BatchNorm2d(n_output))

        def forward(self, x):
            identity = x
            if self.downsample is not None:
                identity = self.downsample(x)
            return self.block(x) + identity

    def __init__(self, layers=None, n_input_channels=3, n_output_channels=5, normalize_input=False):
        super().__init__()
        self.normalize_input = normalize_input
        if layers is None:
            layers = [64, 128]

        reverse_layers = layers.copy()
        reverse_layers.reverse()
        reverse_layers.append(32)
        #reverse_layers = reverse_layers[1:]

        self.f_conv = torch.nn.Conv2d(3, 32, kernel_size=7, padding=3, stride=1)
        self.f_batchnorm = torch.nn.BatchNorm2d(32)
        self.relu = torch.nn.ReLU()
        self.conv_L = []
        c = 32
        for ly in layers:
            self.conv_L.append(self.ConvBlock(c, ly, stride=ly // c))
            c = ly
        self.mid_conv = self.ConvBlock(c, 2*c, stride=2)
        self.mid_upconv = self.UpConvBlock(2*c, c, stride=2)
        self.upconv_L = []
        for ly in reverse_layers[1:]: #[1:]:
            self.upconv_L.append(self.UpConvBlock(c*2, ly, stride=c//ly))
            c = ly

        self.final_skip = torch.nn.Conv2d(c, 32, kernel_size=1, padding=0, stride=1)
        self.s_batchnorm = torch.nn.BatchNorm2d(32)
        self.classification_conv = torch.nn.Conv2d(32, 5, kernel_size = 1, padding=0, stride=1)

        self.conv_layers = torch.nn.Sequential(*self.conv_L)
        self.up_conv_layers = torch.nn.Sequential(*self.upconv_L)

    def forward(self, x):
        if self.normalize_input:
            transform = torchvision.transforms.Normalize(mean=[0.4701, 0.4308, 0.3839], std=[0.2595, 0.2522, 0.2541])
            x = transform(x)
        conv_x = self.relu(self.f_batchnorm(self.f_conv(x)))
        original_size = conv_x.size()
        skips = []
        for conv_layer in self.conv_L:
            conv_x = conv_layer(conv_x)
            skips.append(conv_x)
        in_upconv_x = self.mid_conv(conv_x)
        upconv_x = self.mid_upconv(in_upconv_x, conv_x.size())
        for upconv_layer in self.upconv_L:
            skip = skips.pop()
            goal_size = original_size
            if len(skips) > 0:
                goal_size = skips[-1].size()
            upconv_x = upconv_layer(torch.cat((upconv_x, skip), dim=1), goal_size)
        upconv_x = self.relu(self.s_batchnorm(self.final_skip(upconv_x)))
        logits = self.classification_conv(upconv_x)
        return logits



model_factory = {
    'cnn': CNNClassifier,
    'fcn': FCN,
}


def save_model(model, id):
    from torch import save
    from os import path
    for n, m in model_factory.items():
        if isinstance(model, m):
            return save(model.state_dict(), path.join(path.dirname(path.abspath(__file__)), f'{n}_{id}.th'))
    raise ValueError("model type '%s' not supported!" % str(type(model)))


def load_model(model):
    from torch import load
    from os import path
    if model == "cnn":
        r = model_factory[model](layers=[32, 64, 128], normalize_input=True)
    else:
        r = model_factory[model](layers=[64, 128], normalize_input=True)
    r.load_state_dict(load(path.join(path.dirname(path.abspath(__file__)), '%s.th' % model), map_location='cpu'))
    return r
