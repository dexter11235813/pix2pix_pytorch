import torch
import torch.nn as nn


class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=2):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=4,
                bias=False,
                stride=stride,
                padding_mode="reflect",
            ),
            nn.InstanceNorm2d(out_channels, affine=True),
            nn.LeakyReLU(0.2),
        )

    def forward(self, x):
        return self.conv(x)


class Discriminator(nn.Module):
    def __init__(self, in_channels=3, features=[64, 128, 256, 512]):
        super().__init__()
        # we send in both the generated patch as well as the target patch concatenated as 1 in_channels * 2 image to the first layer
        self.initial = nn.Sequential(
            nn.Conv2d(
                in_channels * 2,
                features[0],
                kernel_size=4,
                stride=2,
                padding=1,
                padding_mode="reflect",
            ),
            nn.LeakyReLU(0.2),
        )

        layers = []
        in_channels = features[0]
        for feature in features[1:]:
            layers.append(
                CNNBlock(
                    in_channels, feature, stride=1 if feature == features[-1] else 2
                )
            )
            in_channels = feature

        layers.append(
            nn.Conv2d(in_channels, 1, kernel_size=4, padding=1, padding_mode="reflect")
        )
        self.model = nn.Sequential(*layers)
        # print(self.model)

    # x is the input from the generator, y is a patch from either a real or a fake image
    def forward(self, x, y):
        x = torch.cat([x, y], dim=1)
        x = self.initial(x)
        return self.model(x)


if __name__ == "__main__":
    x = torch.randn((1, 3, 286, 286))
    y = torch.randn((1, 3, 286, 286))
    model = Discriminator()
    preds = model(x, y)
    assert preds.shape == (1, 1, 30, 30)

