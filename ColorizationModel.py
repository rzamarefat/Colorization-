import torch
from ResNet import ResNet, Bottleneck
import os
from torch.nn import ConvTranspose2d, Conv2d

class ColorizationModel(torch.nn.Module):
    def __init__(self, encoder_name="resnet101", 
                    use_pretrained_encoder=True,
                    path_to_imagenet_pretrained_weights=None):    
        super().__init__()
        self.encoder_name = encoder_name
        self.use_pretrained_encoder = True
        self.path_to_imagenet_pretrained_weights = path_to_imagenet_pretrained_weights


        assert self.encoder_name in ["resnet101"]

        if self.encoder_name == "resnet101":
            self.encoder =  ResNet(Bottleneck, [3, 4, 23, 3])

            for name, param in self.encoder.named_parameters():
                param.requres_grad = False
                if name.__contains__("layer4"):
                    param.requres_grad = True

        if self.use_pretrained_encoder:
            assert os.path.isfile(self.path_to_imagenet_pretrained_weights)
            try:
                self.encoder.load_state_dict(torch.load(self.path_to_imagenet_pretrained_weights))
                print("=====> The pretrained ImageNet ckpt is loaded successfully...")
            except Exception as e:
                print(e)
                print("=====> The pretrained ImageNet ckpt is not loaded...")


        # Unet decoder part
        self.decoder = torch.nn.Sequential( 
                ConvTranspose2d(2048, 2048, kernel_size=(2, 2), stride=(2, 2)),
                Conv2d(2048, 1024, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                Conv2d(1024, 1024, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),

                ConvTranspose2d(1024, 1024, kernel_size=(2, 2), stride=(2, 2)),
                Conv2d(1024, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                
                ConvTranspose2d(512, 256, kernel_size=(2, 2), stride=(2, 2)),
                Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                
                ConvTranspose2d(256, 128, kernel_size=(2, 2), stride=(2, 2)),
                Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                
                ConvTranspose2d(128, 64, kernel_size=(2, 2), stride=(2, 2)),
                Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),

                ConvTranspose2d(64, 32, kernel_size=(2, 2), stride=(2, 2)),
                Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),

                ConvTranspose2d(32, 16, kernel_size=(2, 2), stride=(2, 2)),
                Conv2d(16, 8, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                Conv2d(8, 8, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),

                ConvTranspose2d(8, 8, kernel_size=(2, 2), stride=(2, 2)),
                Conv2d(8, 3, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)

        return decoded

        
        
        
if __name__ == "__main__":
    path_to_imagenet_pretrained_weights = "/home/marefat/projects/github/colorization/resnet101-cd907fc2.pth"
    model = ColorizationModel(path_to_imagenet_pretrained_weights=path_to_imagenet_pretrained_weights)

    model(torch.rand(22, 3, 256, 256))