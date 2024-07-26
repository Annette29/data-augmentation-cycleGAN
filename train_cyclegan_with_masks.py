# Initialize the 4 models for use - generator_H2P, generator_p2H, discriminator_H, and disciminator_P & the 4 loss functions to train them - WGAN-GP as the adversarial loss. identity loss, cycle consistency loss, and abnormality loss
generator_H2P = UNetResNet34(in_channels=4, out_channels=3).to(device)  # Perform element-wise multiplication of RGB images with binary masks selected randomly, hence in_channels=4
generator_P2H = UNetResNet34(in_channels=4, out_channels=3).to(device)  # Perform element-wise multiplication of RGB images with their corresponding binary masks, hence in_channels=4
discriminator_H = PatchGANDiscriminator(in_channels=3).to(device)
discriminator_P = PatchGANDiscriminator(in_channels=3).to(device)

generator_H2P.apply(weights_init_normal)
generator_P2H.apply(weights_init_normal)
discriminator_H.apply(weights_init_normal)
discriminator_P.apply(weights_init_normal)

wgan_gp_loss = WassersteinLossGP(lambda_gp=10)
smooth_real_label = 0.9
smooth_fake_label = 0.1
criterion_cycle = CombinedL1L2Loss(lambda_l1=1.0, lambda_l2=1.0).to(device)
criterion_identity = CombinedL1L2Loss(lambda_l1=1.0, lambda_l2=1.0).to(device)
criterion_abnormality = AbnormalityMaskLoss(lambda_l1=1.0, lambda_l2=1.0).to(device)
