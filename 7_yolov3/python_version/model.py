import torch
import torch.nn as nn

architecture_config = [
    (7, 64, 2, 3),
    "M",
    (3, 192, 1, 1),
    "M",
    (1, 128, 1, 0),
    (3, 256, 1, 1),
    (1, 256, 1, 0),
    (3, 512, 1, 1),
    "M",
    [(1, 256, 1, 0), (3, 512, 1, 1), 4],
    (1, 512, 1, 0),
    (3, 1024, 1, 1),
    "M",
    [(1, 512, 1, 0), (3, 1024, 1, 1), 2],
    (3, 1024, 1, 1),
    (3, 1024, 2, 1),
    (3, 1024, 1, 1),
    (3, 1024, 1, 1),
]

class CNN_Block(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs) -> None:
        super(CNN_Block, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.batch_norm = nn.BatchNorm2d(out_channels)
        self.acti_fn = nn.LeakyReLU(0.1)

    def forward(self, x, ):
        return self.acti_fn(self.batch_norm(self.conv(x)))

class Yolo(nn.Module):
    def __init__(self, in_channels=3, split_size=7, num_boxes=2, num_classes=20) -> None:
        super(Yolo, self).__init__()
        self.in_channels = in_channels
        self.architecture_config = architecture_config
        self.main_network = self._create_network(self.architecture_config)
        self.fcs = self._create_fcs(split_size=split_size, 
                                    num_boxes=num_boxes, 
                                    num_classes=num_classes)

    def forward(self, x):
        x = self.main_network(x)
        return self.fcs(torch.flatten(x, start_dim=1))

    def _create_network(self, architecture_config):
        layers = []
        in_channels = self.in_channels
        for x in architecture_config:
            # convolution layer
            if isinstance(x, tuple):
                layers += [CNN_Block(
                    in_channels, x[1], kernel_size=x[0], stride=x[2], padding=x[3],
                )]
                in_channels = x[1]
            elif isinstance(x, str):
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            elif isinstance(x, list):
                conv1 = x[0]
                conv2 = x[1] 
                num_repeats = x[2]
                for _ in range(num_repeats):
                    layers += [
                        CNN_Block(
                            in_channels,
                            conv1[1],
                            kernel_size=conv1[0],
                            stride=conv1[2],
                            padding=conv1[3],
                        )
                    ]
                    layers += [
                        CNN_Block(
                            conv1[1],
                            conv2[1],
                            kernel_size=conv2[0],
                            stride=conv2[2],
                            padding=conv2[3],
                        )
                    ]
                    in_channels = conv2[1]

        return nn.Sequential(*layers)    

    def _create_fcs(self, split_size, num_boxes, num_classes):
        S, B, C = split_size, num_boxes, num_classes

        # In original paper this should be
        # nn.Linear(1024*S*S, 4096),
        # nn.LeakyReLU(0.1),
        # nn.Linear(4096, S*S*(B*5+C))

        return nn.Sequential(
            nn.Flatten(),
            nn.Linear(1024 * S * S, 2048),
            nn.Dropout(0.0),
            nn.LeakyReLU(0.1),
            nn.Linear(2048, S * S * (C + B * 5)),
        )

def test(S=7, B=2, C=20):
    model = Yolo(split_size=S, num_boxes=B, num_classes=C)
    x = torch.rand(2, 3, 448, 448)
    y = model(x)
    print(y.shape)

# test()