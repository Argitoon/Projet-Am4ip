import torch
from torch.nn import Module
import torch.nn as nn

class Autoencoder(nn.Module):
    def __init__(self, output_channels = 3):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(8, 16, 
                               kernel_size=3, 
                               stride=2, 
                               padding=1, 
                               output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, output_channels, 
                               kernel_size=3, 
                               stride=2, 
                               padding=1, 
                               output_padding=1),
            nn.Tanh()
        )
         
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

class CBDNetwork(Module):
    def __init__(self):
        super(CBDNetwork, self).__init__()
        self.noise_estimation = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 3, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )
        self.non_blind_denoising = nn.Sequential(
            nn.Conv2d(6, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=3, 
                               stride=2, padding=1, 
                               output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=3, 
                               stride=2, padding=1, 
                               output_padding=1),
            nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1),  # Output: denoised image
            #nn.ReLU(),
            nn.Tanh()
        )

    def forward(self, x):
        # Noise estimation
        estimated_noise = self.noise_estimation(x)
        
        # Concatenate input image and noise estimate
        concatenated_input = torch.cat((x, estimated_noise), dim=1)
        
        # Non-blind denoising
        denoised_image = self.non_blind_denoising(concatenated_input)
        
        return denoised_image

################################################################################################
class Unet(nn.Module):
    def __init__(self, num_classes):
        super(Unet, self).__init__()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)

        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1)

        self.conv5 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.conv6 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1)

        self.conv7 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.conv8 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1)

        self.conv9 = nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, stride=1, padding=1)
        self.conv10 = nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, stride=1, padding=1)

        self.upsample1 = nn.ConvTranspose2d(in_channels=1024, out_channels=512, kernel_size=2, stride=2)

        self.conv11 = nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.conv12 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1)

        self.upsample2 = nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=2, stride=2)

        self.conv13 = nn.Conv2d(in_channels=512, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.conv14 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1)

        self.upsample3 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=2, stride=2)

        self.conv15 = nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.conv16 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1)

        self.upsample4 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=2, stride=2)

        self.conv17 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv18 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv19 = nn.Conv2d(in_channels=64, out_channels=34, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        #print(x.shape)
        x1 = torch.relu(self.conv1(x))
        x1 = torch.relu(self.conv2(x1))
        
        x2 = self.pool(x1)
        x2 = torch.relu(self.conv3(x2))
        x2 = torch.relu(self.conv4(x2))

        x3 = self.pool(x2)
        x3 = torch.relu(self.conv5(x3))
        x3 = torch.relu(self.conv6(x3))

        x4 = self.pool(x3)
        x4 = torch.relu(self.conv7(x4))
        x4 = torch.relu(self.conv8(x4))

        y = self.pool(x4)
        y = torch.relu(self.conv9(y))
        y = torch.relu(self.conv10(y))

        y = self.upsample1(y)
        y = torch.cat((x4, y), dim=1)
        y = torch.relu(self.conv11(y))
        y = torch.relu(self.conv12(y))

        y = self.upsample2(y)
        y = torch.cat((x3, y), dim=1)
        y = torch.relu(self.conv13(y))
        y = torch.relu(self.conv14(y))

        y = self.upsample3(y)
        y = torch.cat((x2, y), dim=1)
        y = torch.relu(self.conv15(y))
        y = torch.relu(self.conv16(y))

        y = self.upsample4(y)
        #print(y.shape)
        y = torch.cat((x1, y), dim=1)
        y = torch.relu(self.conv17(y))
        #print(y.shape)
        y = torch.relu(self.conv18(y))
        #print(y.shape)
        y = self.conv19(y)
        #print(y.shape)
        y = torch.argmax(y, dim=1).unsqueeze(1)/255
        #print(y.shape)

        return y

class SimplifiedUnet(nn.Module):
    def __init__(self, num_classes):
        super(SimplifiedUnet, self).__init__()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )

        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )
        self.conv6 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )

        self.upsample1 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=2, stride=2)

        self.conv7 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        self.conv8 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )

        self.upsample2 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=2, stride=2)

        self.conv9 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.conv10 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

        self.conv11 = nn.Conv2d(in_channels=64, out_channels=num_classes, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x1 = self.conv1(x)
        x1 = self.conv2(x1)
        
        x2 = self.pool(x1)
        x2 = self.conv3(x2)
        x2 = self.conv4(x2)

        x3 = self.pool(x2)
        x3 = self.conv5(x3)
        x3 = self.conv6(x3)

        y = self.upsample1(x3)
        y = torch.cat((x2, y), dim=1)
        y = self.conv7(y)
        y = self.conv8(y)

        y = self.upsample2(y)
        y = torch.cat((x1, y), dim=1)
        y = self.conv9(y)
        y = self.conv10(y)
        y = self.conv11(y)
        #y = torch.argmax(y, dim=1).unsqueeze(1)/255#.to(torch.float32)

        return y

################################################################################################

class PSPNet(nn.Module):
    def __init__(self, num_classes):
        super(PSPNet, self).__init__()

        # Feature Extraction (Encoder / CNN)
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )

        # Pyramid Pooling Module (PPM)
        self.ppm_pool1 = nn.AdaptiveAvgPool2d(1)
        self.ppm_pool2 = nn.AdaptiveAvgPool2d(2)
        self.ppm_pool3 = nn.AdaptiveAvgPool2d(3)
        self.ppm_pool6 = nn.AdaptiveAvgPool2d(6)

        self.ppm_conv1 = nn.Conv2d(256, 64, kernel_size=1, bias=False)
        self.ppm_conv2 = nn.Conv2d(256, 64, kernel_size=1, bias=False)
        self.ppm_conv3 = nn.Conv2d(256, 64, kernel_size=1, bias=False)
        self.ppm_conv6 = nn.Conv2d(256, 64, kernel_size=1, bias=False)

        self.ppm_bn1 = nn.BatchNorm2d(64)
        self.ppm_bn2 = nn.BatchNorm2d(64)
        self.ppm_bn3 = nn.BatchNorm2d(64)
        self.ppm_bn6 = nn.BatchNorm2d(64)

        self.ppm_relu = nn.ReLU()

        # Final Convolution
        self.final_conv = nn.Sequential(
            nn.Conv2d(256 + 4 * 64, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, num_classes, kernel_size=1)
        )

    def forward(self, x):
        # Encoder
        x = self.encoder(x)

        # Pyramid Pooling Module
        ppm1 = self.ppm_pool1(x)
        ppm1 = self.ppm_relu(self.ppm_bn1(self.ppm_conv1(ppm1)))
        ppm1 = nn.functional.interpolate(ppm1, size=x.size()[2:], mode='bilinear', align_corners=False)

        ppm2 = self.ppm_pool2(x)
        ppm2 = self.ppm_relu(self.ppm_bn2(self.ppm_conv2(ppm2)))
        ppm2 = nn.functional.interpolate(ppm2, size=x.size()[2:], mode='bilinear', align_corners=False)

        ppm3 = self.ppm_pool3(x)
        ppm3 = self.ppm_relu(self.ppm_bn3(self.ppm_conv3(ppm3)))
        ppm3 = nn.functional.interpolate(ppm3, size=x.size()[2:], mode='bilinear', align_corners=False)

        ppm6 = self.ppm_pool6(x)
        ppm6 = self.ppm_relu(self.ppm_bn6(self.ppm_conv6(ppm6)))
        ppm6 = nn.functional.interpolate(ppm6, size=x.size()[2:], mode='bilinear', align_corners=False)

        # Concatenate PPM outputs with original feature map
        x = torch.cat([x, ppm1, ppm2, ppm3, ppm6], dim=1)

        # Final Convolution
        x = self.final_conv(x)
        
        # TODO : This is weird ?? Maybe caus some problem
        x = nn.functional.interpolate(x, size=(128, 128), mode='bilinear', align_corners=False)

        return x
    
################################################################################################

class DenoiserUnet(nn.Module):
    def __init__(self):
        super(DenoiserUnet, self).__init__()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )

        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )
        self.conv6 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )

        self.upsample1 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=2, stride=2)

        self.conv7 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        self.conv8 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )

        self.upsample2 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=2, stride=2)

        self.conv9 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.conv10 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

        self.conv11 = nn.Conv2d(in_channels=64, out_channels=3, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x1 = self.conv1(x)
        x1 = self.conv2(x1)
        
        x2 = self.pool(x1)
        x2 = self.conv3(x2)
        x2 = self.conv4(x2)

        x3 = self.pool(x2)
        x3 = self.conv5(x3)
        x3 = self.conv6(x3)

        y = self.upsample1(x3)
        y = torch.cat((x2, y), dim=1)
        y = self.conv7(y)
        y = self.conv8(y)

        y = self.upsample2(y)
        y = torch.cat((x1, y), dim=1)
        y = self.conv9(y)
        y = self.conv10(y)
        y = self.conv11(y)
        y = torch.clamp(y, 0, 1)

        return y