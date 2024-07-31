import os
import torch
from torchvision import transforms

from with_masks.train_cyclegan_with_masks import initialize_components

(
    generator_H2P, generator_P2H, train_loader_healthy, train_loader_pathological, 
    validation_loader_healthy, validation_loader_pathological, test_loader_healthy, test_loader_pathological
) = initialize_components(device)

# Load previously-saved model checkpoint
checkpoint_path = '/your/model checkpoints/folder'
epoch_to_load = # latest_model_checkpoint_saved e.g. 1000 if training stopped at epoch1000
generator_H2P.load_state_dict(torch.load(os.path.join(checkpoint_path, f'generator_H2P_epoch{epoch_to_load}.pth')))
generator_P2H.load_state_dict(torch.load(os.path.join(checkpoint_path, f'generator_P2H_epoch{epoch_to_load}.pth')))
generator_H2P.eval()
generator_P2H.eval()

def setup_directories(base_dir, categories, sub_dirs):
    """
    Create directories if they don't exist.
    
    Parameters:
    - base_dir: str, base directory where the folders will be created.
    - categories: list of str, category names for subdirectories.
    - sub_dirs: list of str, names of subdirectories to create within each category.
    """
    for category in categories:
        for sub_dir in sub_dirs:
            dir_path = os.path.join(base_dir, category, sub_dir)
            os.makedirs(dir_path, exist_ok=True)

def generate_fake_samples(data_loader, generator, output_dir, device, suffix='_fake'):
    """
    Generate fake samples using the generator and save them to the specified directory.
    
    Parameters:
    - data_loader: DataLoader, data loader for the images.
    - generator: nn.Module, generator model to create fake images.
    - output_dir: str, directory to save the fake images.
    - device: torch.device, device to perform computations on.
    - suffix: str, suffix to add to the saved fake image filenames.
    """
    generator.eval()
    with torch.no_grad():
        for batch_idx, (real_images, masks, image_names) in enumerate(data_loader):
            real_images = real_images.to(device)
            masks = masks.to(device)

            # Generate fake images
            fake_images = generator(real_images, masks)

            # Save fake images
            for idx in range(fake_images.size(0)):
                fake_image = fake_images[idx].detach().cpu()
                fake_image = (fake_image + 1) / 2.0  # Denormalize to [0, 1]
                fake_image = transforms.ToPILImage()(fake_image)

                # Create the new filename with the suffix
                fake_image_name = f"{os.path.splitext(image_names[idx])[0]}{suffix}{os.path.splitext(image_names[idx])[1]}"

                # Save the fake image
                fake_image.save(os.path.join(output_dir, fake_image_name))
