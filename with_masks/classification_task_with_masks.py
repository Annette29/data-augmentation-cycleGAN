import torch
import torch.nn as nn
from torchvision import models
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, ConcatDataset, Subset
from sklearn.model_selection import KFold
from copy import deepcopy
import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline

class CustomDenseNet121(nn.Module):
    def __init__(self, num_classes=2, weights=None):
        super(CustomDenseNet121, self).__init__()
        if weights:
            self.model = models.densenet121(weights=weights)
        else:
            self.model = models.densenet121(pretrained=False)

        self._freeze_layers()

        num_features = self.model.classifier.in_features
        self.model.classifier = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
        self._initialize_weights()

    def forward(self, x):
        x = self.model(x)
        return x

    def _initialize_weights(self):
        for m in self.model.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _freeze_layers(self):
        for param in self.model.parameters():
            param.requires_grad = False

class FocalLoss(nn.Module):
    def __init__(self, alpha=2, gamma=3, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        BCE_loss = nn.CrossEntropyLoss(reduction='none')(inputs, targets)
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss

        if self.reduction == 'mean':
            return F_loss.mean()
        elif self.reduction == 'sum':
            return F_loss.sum()
        else:
            return F_loss

def evaluate_model(model, loader, device, threshold=0.7):
    model.eval()
    all_labels = []
    all_preds = []
    all_probs = []

    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)
            preds = (probs[:, 1] >= threshold).long()

            all_labels.append(labels.cpu().numpy())
            all_preds.append(preds.cpu().numpy())
            all_probs.append(probs[:, 1].cpu().numpy())

    all_labels = np.concatenate(all_labels)
    all_preds = np.concatenate(all_preds)
    all_probs = np.concatenate(all_probs)

    if np.array_equal(np.unique(all_labels), np.array([0, 1])):
        tn, fp, fn, tp = confusion_matrix(all_labels, all_preds).ravel()
    else:
        tn, fp, fn, tp = None, None, None, None

    return all_labels, all_probs, tn, fp, fn, tp

def calculate_sensitivity_at_fp(labels, probs, thresholds=np.linspace(0, 1, 300)):
    sensitivity = []
    false_positives = []

    for threshold in thresholds:
        binary_preds = (probs >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(labels, binary_preds).ravel()
        sens = tp / (tp + fn)
        sensitivity.append(sens)
        false_positives.append(fp)

    return sensitivity, false_positives

def smooth_curve(x, y, smooth_factor=300):
    x_unique, index = np.unique(x, return_index=True)
    y_unique = np.array(y)[index]

    x_new = np.linspace(np.min(x), np.max(x), smooth_factor)
    spline = make_interp_spline(x_unique, y_unique, k=3)
    y_smooth = spline(x_new)
    return x_new, y_smooth

def plot_sensitivity_vs_fp_comparison(sensitivity_real, false_positives_real, sensitivity_synthetic, false_positives_synthetic, sensitivity_combined, false_positives_combined):
    plt.figure(figsize=(10, 6))

    x_smooth_real, y_smooth_real = smooth_curve(false_positives_real, sensitivity_real)
    x_smooth_synthetic, y_smooth_synthetic = smooth_curve(false_positives_synthetic, sensitivity_synthetic)
    x_smooth_combined, y_smooth_combined = smooth_curve(false_positives_combined, sensitivity_combined)

    plt.plot(x_smooth_real, y_smooth_real, label='Real Data', color='blue')
    plt.plot(x_smooth_synthetic, y_smooth_synthetic, label='Synthetic Data', color='green')
    plt.plot(x_smooth_combined, y_smooth_combined, label='Combined Data', color='red')

    plt.xlabel('Number of false positives')
    plt.ylabel('Sensitivity [%]')
    plt.legend(loc='lower right', fontsize='small', ncol=2)
    plt.grid(True)
    plt.xlim(0, 300)
    plt.show()

def initialize_model(weights):
    model = CustomDenseNet121(num_classes=2, weights=weights).to(device)
    optimizer = optim.Adam(model.model.classifier.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = StepLR(optimizer, step_size=30, gamma=0.1)
    return model, optimizer, scheduler

def train_model(model, train_data, val_data, criterion, optimizer, scheduler, num_epochs=100, batch_size=32, threshold=0.7):
    best_model_wts = deepcopy(model.state_dict())
    best_acc = 0.0

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)

    sensitivity_progression = []
    false_positives_progression = []

    for epoch in range(num_epochs):
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
                loader = train_loader
            else:
                model.eval()
                loader = val_loader

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in loader:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(loader.dataset)
            epoch_acc = running_corrects.double() / len(loader.dataset)

            if epoch % 50 == 0:
                print(f'Epoch {epoch}/{num_epochs - 1}')
                print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = deepcopy(model.state_dict())

        if phase == 'train':
            scheduler.step()

        if phase == 'val':
            val_labels, val_probs, _, _, _, _ = evaluate_model(model, val_loader, device, threshold=threshold)
            sensitivity, false_positives = calculate_sensitivity_at_fp(val_labels, val_probs)
            sensitivity_progression.append(sensitivity)
            false_positives_progression.append(false_positives)

    print('Best val Acc: {:4f}'.format(best_acc))
    model.load_state_dict(best_model_wts)
    return model, sensitivity_progression, false_positives_progression
