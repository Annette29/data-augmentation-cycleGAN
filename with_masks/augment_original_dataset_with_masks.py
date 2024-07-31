import os

from with_masks.train_cyclegan_with_masks import initialize_components, limit_samples

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
