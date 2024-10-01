import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from torchvision.models.resnet import ResNet34_Weights
import torch.nn.functional as F

# Define SelfAttention class
# The self_attention layer is added to the encoder part of the generator. It is applied after the final encoding layer (e6), before the decoder starts to enhance the model's ability to capture global dependencies within the input images
class SelfAttention(nn.Module):
    def __init__(self, in_dim):
        super(SelfAttention, self).__init__()
        self.query_conv = nn.Conv2d(in_dim, in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_dim, in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_dim, in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        batch_size, C, width, height = x.size()
        proj_query = self.query_conv(x).view(batch_size, -1, width * height).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(batch_size, -1, width * height)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        proj_value = self.value_conv(x).view(batch_size, -1, width * height)
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(batch_size, C, width, height)
        out = self.gamma * out + x
        return out

# Define UNetResNet34 model
class UNetResNet34(nn.Module):
    def __init__(self, in_channels=4, out_channels=3, dropout_prob=0.1):
        super(UNetResNet34, self).__init__()
        resnet = models.resnet34(weights=ResNet34_Weights.IMAGENET1K_V1)

        # Modify the first layer to accept in_channels (4 instead of 3)
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(inplace=True),
            resnet.layer1,  # First residual block
            resnet.layer2,  # Second residual block
            resnet.layer3,  # Third residual block
            resnet.layer4   # Fourth residual block
        )

        # Initialize weights
        self.encoder[0].weight.data[:, :3, :, :] = resnet.conv1.weight.data
        self.encoder[0].weight.data[:, 3, :, :] = resnet.conv1.weight.data.mean(dim=1)

        # Add Self-Attention after the last encoding layer
        self.self_attention = SelfAttention(in_dim=512)

        # Decoder layers with batch normalization
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2),  # Adjusted: From 512 (e5) to 256 channels
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=False)
        )
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(256 + 256, 128, kernel_size=2, stride=2),  # Adjusted: From 256 + 256 (d1 + e4) to 128 channels
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=False)
        )
        self.up3 = nn.Sequential(
            nn.ConvTranspose2d(128 + 128, 64, kernel_size=2, stride=2),  # Adjusted: From 128 + 128 (d2 + e3) to 64 channels
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=False)
        )
        self.up4 = nn.Sequential(
            nn.ConvTranspose2d(64 + 64, 64, kernel_size=2, stride=2),  # Adjusted: From 64 + 64 (d3 + e2) to 64 channels
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=False)
        )
        self.conv_final = nn.Conv2d(64 + 64, out_channels, kernel_size=1)

        self.dropout = nn.Dropout(p=dropout_prob)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

    def forward(self, x, mask):
        if x.size() != mask.size():
            mask = F.interpolate(mask, size=(x.size(2), x.size(3)), mode='bilinear', align_corners=True)

        # Concatenate input image and mask along the channel dimension
        x_concat = torch.cat((x, mask), dim=1)

        # Encoder
        e1 = self.encoder[0](x_concat)
        e2 = self.encoder[1](e1)
        e3 = self.encoder[4](e2)
        e4 = self.encoder[5](e3)
        e5 = self.encoder[6](e4)

        # Apply Self-Attention
        e5 = self.self_attention(e5)

        # Prepare masks for element-wise multiplication in the decoder
        mask_up1 = F.interpolate(mask, size=(e5.size(2), e5.size(3)), mode='bilinear', align_corners=True)
        mask_up2 = F.interpolate(mask_up1, size=(e4.size(2), e4.size(3)), mode='bilinear', align_corners=True)
        mask_up3 = F.interpolate(mask_up2, size=(e3.size(2), e3.size(3)), mode='bilinear', align_corners=True)
        mask_up4 = F.interpolate(mask_up3, size=(e2.size(2), e2.size(3)), mode='bilinear', align_corners=True)

        # Upsample e1 to match d4's size for concatenation
        e1_upsampled = self.upsample(e1)

        # Decoder with skip connections and mask element-wise multiplication
        d1 = self.dropout(self.up1(e5 * mask_up1))
        d2 = self.dropout(self.up2(torch.cat([d1, e4], dim=1) * mask_up2))
        d3 = self.dropout(self.up3(torch.cat([d2, e3], dim=1) * mask_up3))
        d4 = self.dropout(self.up4(torch.cat([d3, e2], dim=1) * mask_up4))
        output = self.conv_final(torch.cat([d4, e1_upsampled], dim=1) * mask)

        return output

# Define weight initialization function to  initialize convolutional layers using Xavier normal initialization and initialize BatchNorm layers with weights set to 1.0 and biases to 0.0
def weights_init_normal(m):
    classname = m.__class__.__name__
    if hasattr(m, 'weight') and (classname.find('Conv') != -1):
        nn.init.xavier_normal_(m.weight.data)
    elif hasattr(m, 'weight') and (classname.find('BatchNorm') != -1 or classname.find('InstanceNorm') != -1):
        if m.weight is not None:
            nn.init.constant_(m.weight.data, 1.0)
        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0.0)

# Define PatchGANDiscriminator class
class PatchGANDiscriminator(nn.Module):
    def __init__(self, in_channels=3, dropout_prob=0.1):
        super(PatchGANDiscriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=False),
            nn.Dropout(p=dropout_prob),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.2, inplace=False),
            nn.Dropout(p=dropout_prob),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(256),
            nn.LeakyReLU(0.2, inplace=False),
            nn.Dropout(p=dropout_prob),
            nn.Conv2d(256, 512, kernel_size=4, stride=1, padding=1),
            nn.InstanceNorm2d(512),
            nn.LeakyReLU(0.2, inplace=False),
            nn.Dropout(p=dropout_prob),
            nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=1)
        )

    def forward(self, x):
        return self.model(x)

# Define Wasserstein Loss with Gradient Penalty
class WassersteinLossGP:
    def __init__(self, lambda_gp=10):
        self.lambda_gp = lambda_gp

    def __call__(self, D, real_samples, fake_samples, smooth_real_label=None, smooth_fake_label=None, apply_label_smoothing=False):
        real_validity = D(real_samples)
        fake_validity = D(fake_samples)
 
        # Conditionally apply label smoothing only when calculating the discriminator loss
        if apply_label_smoothing:
            if smooth_real_label is not None:
                real_validity = real_validity * smooth_real_label
            if smooth_fake_label is not None:
                fake_validity = fake_validity * smooth_fake_label

        gradient_penalty = self.compute_gradient_penalty(D, real_samples.data, fake_samples.data)
        return -torch.mean(real_validity) + torch.mean(fake_validity) + self.lambda_gp * gradient_penalty

    def compute_gradient_penalty(self, D, real_samples, fake_samples):
        alpha = torch.rand(real_samples.size(0), 1, 1, 1, device=real_samples.device)
        interpolates = alpha * real_samples + (1 - alpha) * fake_samples
        interpolates.requires_grad_(True)

        d_interpolates = D(interpolates)
        gradients = torch.autograd.grad(
            outputs=d_interpolates,
            inputs=interpolates,
            grad_outputs=torch.ones(d_interpolates.size(), device=real_samples.device),
            create_graph=True,
            retain_graph=True,
            only_inputs=True
        )[0]

        gradients = gradients.view(gradients.size(0), -1)
        gradient_penalty = torch.mean((gradients.norm(2, dim=1) - 1) ** 2)
        return gradient_penalty

# Custom loss function that combines L1 and L2 losses to be used for the Cycle Consistency and Identity Losses
class CombinedL1L2Loss(nn.Module):
    def __init__(self, lambda_l1=1.0, lambda_l2=1.0):
        super(CombinedL1L2Loss, self).__init__()
        self.l1_loss = nn.L1Loss()
        self.l2_loss = nn.MSELoss()
        self.lambda_l1 = lambda_l1
        self.lambda_l2 = lambda_l2

    def forward(self, input, target):
        l1 = self.l1_loss(input, target)
        l2 = self.l2_loss(input, target)
        return self.lambda_l1 * l1 + self.lambda_l2 * l2

# Abnormality Mask Loss applies the mask to both input and target images before calculating the combined L1L2 loss, ensuring changes are restricted to pathological regions
class AbnormalityMaskLoss(nn.Module):
    def __init__(self, lambda_l1=1.0, lambda_l2=1.0):
        super(AbnormalityMaskLoss, self).__init__()
        self.l1_loss = nn.L1Loss()
        self.l2_loss = nn.MSELoss()
        self.lambda_l1 = lambda_l1
        self.lambda_l2 = lambda_l2

    def forward(self, input, target, mask):
        # Apply the mask to both input and target images
        masked_input = input * mask
        masked_target = target * mask
        # Calculate L1 and L2 losses only in masked regions
        l1 = self.l1_loss(masked_input, masked_target)
        l2 = self.l2_loss(masked_input, masked_target)
        return self.lambda_l1 * l1 + self.lambda_l2 * l2
