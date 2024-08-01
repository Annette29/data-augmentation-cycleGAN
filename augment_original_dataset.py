import os
import torch
from torchvision import transforms

from with_masks.train_cyclegan_with_masks import initialize_components as initialize_components_masks
from without_masks.train_cyclegan_without_masks import initialize_components as initialize_components_without_masks

def initialize_and_load_generators(device, checkpoint_path, checkpoint_path_no_masks, epoch_to_load):
    # Initialize generators with masks
    generator_H2P_masks, generator_P2H_masks = initialize_components_masks(device)
    generator_H2P_masks.load_state_dict(
        torch.load(os.path.join(checkpoint_path, f'generator_H2P_masks_epoch{epoch_to_load}.pth'))
    )
    generator_P2H_masks.load_state_dict(
        torch.load(os.path.join(checkpoint_path, f'generator_P2H_masks_epoch{epoch_to_load}.pth'))
    )
    
    # Initialize generators without masks
    generator_H2P_without, generator_P2H_without = initialize_components_without_masks(device)
    generator_H2P_without.load_state_dict(
        torch.load(os.path.join(checkpoint_path_no_masks, f'generator_H2P_without_epoch{epoch_to_load}.pth'))
    )
    generator_P2H_without.load_state_dict(
        torch.load(os.path.join(checkpoint_path_no_masks, f'generator_P2H_without_epoch{epoch_to_load}.pth'))
    )
    
    return {
        'with_masks': (generator_H2P_masks, generator_P2H_masks),
        'without_masks': (generator_H2P_without, generator_P2H_without)
    }

# Load previously-saved model checkpoint
checkpoint_path = '/your/model with masks checkpoints/folder'
checkpoint_path_no_masks = '/your/model without masks checkpoints/folder'
epoch_to_load = # latest_model_checkpoint_saved e.g. 1000 if training stopped at epoch1000

generators = initialize_and_load_generators(device, checkpoint_path, checkpoint_path_no_masks, epoch_to_load)
generator_H2P_masks, generator_P2H_masks = generators['with_masks']
generator_H2P_without, generator_P2H_without = generators['without_masks']

def setup_directories(base_dir, categories, sub_dirs):
    for category in categories:
        for sub_dir in sub_dirs:
            dir_path = os.path.join(base_dir, category, sub_dir)
            os.makedirs(dir_path, exist_ok=True)

def generate_fake_samples_masks(generator, data_loader, output_dir, suffix='_fake'):
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

def generate_fake_samples_without_masks(generator, data_loader, output_dir, suffix='_fake'):
    generator.eval()
    with torch.no_grad():
        for batch_idx, (real_images, image_names) in enumerate(data_loader):
            real_images = real_images.to(device)

            # Generate fake images
            fake_images = generator(real_images)

            # Save fake images
            for idx in range(fake_images.size(0)):
                fake_image = fake_images[idx].detach().cpu()
                fake_image = (fake_image + 1) / 2.0  # Denormalize to [0, 1]
                fake_image = transforms.ToPILImage()(fake_image)
                fake_image_name = f"{os.path.splitext(image_names[idx])[0]}{suffix}{os.path.splitext(image_names[idx])[1]}"
                fake_image.save(os.path.join(output_dir, fake_image_name))

# Function that selects random images and their corresponding fakes from each dataset and then plots them
def plot_random_pairs(real_dir, fake_dir, suffix='_fake', num_pairs=5):
    # Get a list of all image filenames
    real_images = os.listdir(real_dir)
    fake_images = os.listdir(fake_dir)

    # Ensure the same number of images and matching filenames
    real_images_set = set(os.path.splitext(f)[0] for f in real_images)
    fake_images_set = set(os.path.splitext(f)[0].replace(suffix, '') for f in fake_images)
    common_images = list(real_images_set & fake_images_set)

    if len(common_images) < num_pairs:
        raise ValueError("Not enough matching images in both directories to plot pairs.")

    # Randomly select num_pairs images
    selected_basenames = random.sample(common_images, num_pairs)

    # Plotting
    fig, axes = plt.subplots(2, num_pairs, figsize=(10, 5), gridspec_kw={'hspace': 0.3})

    for i in range(num_pairs):
        # Load and plot real image
        real_image_name = selected_basenames[i] + ".png"  # or ".jpg" depending on your file extension
        real_image = Image.open(os.path.join(real_dir, real_image_name))
        axes[0, i].imshow(real_image)
        axes[0, 2].set_title('Original Images')
        axes[0, i].axis('off')

        # Load and plot fake image
        fake_image_name = selected_basenames[i] + suffix + ".png"  # or ".jpg" depending on your file extension
        fake_image = Image.open(os.path.join(fake_dir, fake_image_name))
        axes[1, i].imshow(fake_image)
        axes[1, 2].set_title('Synthetic Images')
        axes[1, i].axis('off')

    fig.subplots_adjust(wspace=0.1, hspace=0.3, top=0.85)  # Fine-tune spacing between subplots and adjust top margin
    plt.show()

