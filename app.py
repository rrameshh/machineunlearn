import streamlit as st
import subprocess

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torchvision.models import resnet18
from torch.utils.data import DataLoader, random_split
from torch.utils.data import ConcatDataset
from torch.utils.data import Dataset
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

import random
import torch.nn.functional as F
import numpy as np
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

class CustomModel(nn.Module):
    def __init__(self, num_classes):
        super(CustomModel, self).__init__()
        # Replace this with your own model architecture
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(256 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
    

def kl_loss(model_logits, teacher_logits, temperature=1.):
     # apply softmax to the (scaled) logits of the teacher model
    teacher_output_softmax = F.softmax(teacher_logits / temperature, dim=1) 
    # apply log softmax to the (scaled) logits of the student model
    output_log_softmax = F.log_softmax(model_logits / temperature, dim=1)
    
    # calculate the KL divergence between the student and teacher outputs
    kl_div = F.kl_div(output_log_softmax, teacher_output_softmax, reduction='batchmean')
    return kl_div

def soft_cross_entropy(preds, soft_targets):
    loss = torch.sum(-soft_targets * torch.log_softmax(preds, dim=1), dim=1)
    return torch.mean(loss)


def unlearning(model, teacher_model, retain_loader, forget_loader, val_loader, temperature=1.,
               weight=None, device='cuda'):
    model.conv1.reset_parameters()
    model.fc.reset_parameters()
    
    # set the teacher model to evaluation mode
    teacher_model.eval()
    
    # define the number of epochs for warm-up and retain phases
    retain_epochs = 3
    warmup_epochs = 3
    
    # standard cross-entropy loss for fine-tuning
    criterion = nn.CrossEntropyLoss(weight=weight)
    
    # Warm-up phase: Adjust the student model closer to the teacher model using knowledge distillation
    optimizer = torch.optim.SGD(model.parameters(), lr=9e-4, weight_decay=5e-4, momentum=0.9)
    for epoch in range(warmup_epochs):

        model.train()
        for sample in val_loader:
            x = sample["image"]
            y = sample["age_group"]
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            student_out = model(x)
            
            with torch.no_grad():
                teacher_out = teacher_model(x)
                
            loss = kl_loss(model_logits=student_out, 
                           teacher_logits=teacher_out, 
                           temperature=temperature)
            loss.backward()
            optimizer.step()

    # Fine-tuning phase: Train the model on the retain set using standard cross-entropy along with knowledge distillation
    optimizer = torch.optim.SGD(model.parameters(),
                                lr=1e-3, weight_decay=5e-4, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=retain_epochs
    )
    for epoch in range(retain_epochs):
        model.train()
        for sample in retain_loader:
            x = sample["image"]
            y = sample["age_group"]
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            with torch.no_grad():
                teacher_out = teacher_model(x)
            soft_labels = torch.softmax(teacher_out / temperature, dim=1)
            soft_predictions = torch.log_softmax(out / temperature, dim=1)
            loss = soft_cross_entropy(soft_predictions, soft_labels)
            loss += criterion(out, y)
            loss += kl_loss(model_logits=out, 
                           teacher_logits=teacher_out, 
                           temperature=temperature)
            loss.backward()
            optimizer.step()
        scheduler.step()

    model.eval()

def calculate_mia_scores(model, dataloader, criterion, optimizer, num_epochs=5):
    """
    Calculate MIA scores for the model with and without unlearning.

    Args:
        model: The machine learning model.
        dataloader: DataLoader for the dataset.
        criterion: Loss criterion for training.
        optimizer: Optimizer for training.
        num_epochs: Number of training epochs.

    Returns:
        The MIA scores for the two models.
    """
    # Clone the original model
    original_model = CustomModel(num_classes=10)
    original_model.load_state_dict(model.state_dict())

    # Train the model without unlearning
    train_model(original_model, dataloader, criterion, optimizer, num_epochs=num_epochs)

    # Train the model with unlearning
    unlearning_model = CustomModel(num_classes=num_classes)
    unlearning_model.load_state_dict(model.state_dict())
    unlearning(unlearning_model, dataloader, criterion, optimizer, num_epochs=num_epochs)

    # Generate synthetic samples for inversion attack
    synthetic_samples = torch.randn_like(next(iter(dataloader))[0], requires_grad=True)

    # Invert the original model
    original_model.eval()
    original_outputs = original_model(synthetic_samples)
    original_scores = torch.softmax(original_outputs, dim=1)[:, 1].detach().numpy()

    # Invert the model with unlearning
    unlearning_model.eval()
    unlearning_outputs = unlearning_model(synthetic_samples)
    unlearning_scores = torch.softmax(unlearning_outputs, dim=1)[:, 1].detach().numpy()

    return original_scores, unlearning_scores

def plot_mia_curve(original_scores, unlearning_scores):
    """
    Plot the MIA curve.

    Args:
        original_scores: MIA scores for the model without unlearning.
        unlearning_scores: MIA scores for the model with unlearning.
    """
    fpr_original, tpr_original, _ = roc_curve(np.ones_like(original_scores), original_scores)
    fpr_unlearning, tpr_unlearning, _ = roc_curve(np.zeros_like(unlearning_scores), unlearning_scores)

    roc_auc_original = auc(fpr_original, tpr_original)
    roc_auc_unlearning = auc(fpr_unlearning, tpr_unlearning)

    plt.figure()
    plt.plot(fpr_original, tpr_original, color='darkorange', lw=2, label=f'Original Model (AUC = {roc_auc_original:.2f})')
    plt.plot(fpr_unlearning, tpr_unlearning, color='green', lw=2, label=f'Unlearning Model (AUC = {roc_auc_unlearning:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Model Inversion Attack Curve')
    plt.legend(loc="lower right")
    st.pyplot()

def main():
    st.title("Machine Unlearning")

    # Upload pretrained model weights
    model_weights = st.file_uploader("Upload pretrained model weights (must be a .pth file)", type=["pth"])

    # Upload retained and forget datasets
    retained_set = st.file_uploader("Upload retained dataset", type=["zip", "tar.gz", "tar"])
    forget_set = st.file_uploader("Upload forget dataset", type=["zip", "tar.gz", "tar"])

    # Number of classes in the dataset (modify based on your dataset)
    num_classes = 10

    if model_weights is not None and retained_set is not None and forget_set is not None:
        # Load the user-uploaded pretrained model
        try:
            loaded_model = CustomModel(num_classes=num_classes)
            checkpoint = torch.load(model_weights, map_location=torch.device('cpu'))
            loaded_model.load_state_dict(checkpoint['model_state_dict'])
            st.success("User-uploaded pretrained model loaded successfully!")
        except Exception as e:
            st.error(f"Error loading the pretrained model: {e}")
            return

        # Define transformations and load datasets
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        retained_dataset = ImageFolder(root="retain_data", transform=transform)
        forget_dataset = ImageFolder(root="forget_data", transform=transform)

        # Use user-uploaded datasets if provided
        if retained_set:
            retained_dataset = ImageFolder(root="retain_data", transform=transform)
            retained_dataset.data = torch.load(retained_set)["data"]
            retained_dataset.targets = torch.load(retained_set)["targets"]

        if forget_set:
            forget_dataset = ImageFolder(root="forget_data", transform=transform)
            forget_dataset.data = torch.load(forget_set)["data"]
            forget_dataset.targets = torch.load(forget_set)["targets"]

        # Combine retained and forget datasets into a single DataLoader for training
        combined_dataset = ConcatDataset([retained_dataset, forget_dataset])
        combined_dataloader = DataLoader(combined_dataset, batch_size=64, shuffle=True)

        teacher_model = models.resnet18(pretrained=false)
        teacher_model.load_state_dict(torch.load(loaded_model))
        teacher_model.to(DEVICE)

        model = resnet18(pretrained=false)
        model.to(DEVICE)
        for i in range(512):
            model.load_state_dict(torch.load(loaded_model)) # student model
            unlearning(model, teacher_model, retained_dataset, forget_dataset, combined_dataset, temperature=5.,
                   weight=None, device=DEVICE) # perform unlearning
            state = model.state_dict()
            torch.save(state, f'/tmp/unlearned_checkpoint_{i}.pth')
        #
        
        original_scores, unlearning_scores = calculate_mia_scores(
            loaded_model, combined_dataloader, criterion_kl, optimizer_kl)

        plot_mia_curve(original_scores, unlearning_scores)

        
        updated_model_path = "updated_model.pth"
        torch.save(loaded_model.state_dict(), updated_model_path)

        # Provide a link for the user to download the updated model
        st.subheader("Download Updated Model:")
        st.markdown(f"[Download Updated Model]({updated_model_path})")

        # Optionally, you can remove the saved model file after providing the link
        os.remove(updated_model_path)

        st.success("Machine unlearning applied successfully!")
        

if __name__ == "__main__":
    main()
