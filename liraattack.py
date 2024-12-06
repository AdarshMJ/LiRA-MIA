import warnings
warnings.filterwarnings('ignore')
import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import roc_curve, auc
import pandas as pd
from torchvision.models import resnet18
import torch.nn as nn
import torch.optim as optim
import scipy.stats
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader, Subset
from typing import Tuple

class TaskDataset(Dataset):
    def __init__(self, transform=None):
        self.ids = []
        self.imgs = []
        self.labels = []
        self.transform = transform

    def __getitem__(self, index) -> Tuple[int, torch.Tensor, int]:
        id_ = self.ids[index]
        img = self.imgs[index]
        if not self.transform is None:
            img = self.transform(img)
        label = self.labels[index]
        return id_, img, label

    def __len__(self):
        return len(self.ids)

class MembershipDataset(TaskDataset):
    def __init__(self, transform=None):
        super().__init__(transform)
        self.membership = []

    def __getitem__(self, index) -> Tuple[int, torch.Tensor, int, int]:
        id_, img, label = super().__getitem__(index)
        return id_, img, label, self.membership[index]

class LiRAAttack:
    def __init__(self, target_model, n_shadow_models, n_epochs, batch_size):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.target_model = target_model.to(self.device)
        self.n_shadow_models = n_shadow_models
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.shadow_models = []

    def train_shadow_models(self, dataset):
        """Train shadow models with class-balanced sampling"""
        print(f"Training {self.n_shadow_models} shadow models...")
        
        # Calculate class weights for balanced sampling
        label_counts = np.bincount([label for _, _, label, _ in dataset])
        class_weights = 1. / label_counts
        class_weights = class_weights / class_weights.sum()
        
        for i in tqdm(range(self.n_shadow_models)):
            shadow_model = resnet18(pretrained=False)
            shadow_model.fc = nn.Linear(512, 44)
            shadow_model = shadow_model.to(self.device)
            
            # Weighted sampling for better class balance
            weights = [class_weights[label] for _, _, label, _ in dataset]
            weights = np.array(weights) / sum(weights)
            indices = np.random.choice(len(dataset), size=len(dataset)//2, replace=False, p=weights)
            self._train_single_shadow(shadow_model, dataset, indices)
            shadow_model.eval()
            self.shadow_models.append(shadow_model)

    def _train_single_shadow(self, model, dataset, indices):
        """Train a single shadow model with weighted loss"""
        model.train()
        
        # Calculate class weights for loss function
        label_counts = np.bincount([dataset.labels[i] for i in indices])
        class_weights = torch.FloatTensor(1. / label_counts).to(self.device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        optimizer = optim.Adam(model.parameters(), lr=0.01)
        
        subset = Subset(dataset, indices)
        dataloader = DataLoader(subset, batch_size=self.batch_size, shuffle=True)
        
        for epoch in range(self.n_epochs):
            for batch in dataloader:
                _, img, label, _ = batch
                img = img.to(self.device)
                label = label.to(self.device)
                
                optimizer.zero_grad()
                outputs = model(img)
                loss = criterion(outputs, label)
                loss.backward()
                optimizer.step()

    def compute_attack_scores(self, dataset):
        scores = []
        memberships = []
        
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        print("Computing attack scores...")
        with torch.no_grad():
            for batch in tqdm(dataloader):
                ids, imgs, labels, batch_memberships = batch
                imgs = imgs.to(self.device)
                labels = labels.to(self.device)
                
                for img, label, membership in zip(imgs, labels, batch_memberships):
                    shadow_preds = self._get_shadow_predictions(img, label)
                    test_score = self._calculate_test_score(shadow_preds, img, label)
                    
                    scores.append(test_score)
                    memberships.append(membership.item())
        
        return np.array(scores), np.array(memberships)

    def predict_membership_scores(self, dataset, output_path="predictions.csv"):
        """Predict membership scores for a dataset in batches and save to CSV"""
        scores = []
        ids = []
        
        # Use a collate_fn to handle None values in membership
        def collate_fn(batch):
            ids = torch.tensor([item[0] for item in batch])
            imgs = torch.stack([item[1] for item in batch])
            labels = torch.tensor([item[2] for item in batch])
            # Ignore membership values as they might be None
            return ids, imgs, labels
        
        dataloader = DataLoader(
            dataset, 
            batch_size=self.batch_size, 
            shuffle=False,
            collate_fn=collate_fn
        )
        
        print("Computing membership scores...")
        with torch.no_grad():
            for batch_ids, imgs, labels in tqdm(dataloader):
                imgs = imgs.to(self.device)
                labels = labels.to(self.device)
                
                for id_, img, label in zip(batch_ids, imgs, labels):
                    shadow_preds = self._get_shadow_predictions(img, label)
                    test_score = self._calculate_test_score(shadow_preds, img, label)
                    
                    scores.append(test_score)
                    ids.append(id_.item())
        
        # Save predictions to CSV
        df = pd.DataFrame({
            'id': ids,
            'membership_score': scores
        })
        df.to_csv(output_path, index=False)       
        return scores

    def _get_shadow_predictions(self, img, label):
        """Get predictions from all shadow models"""
        shadow_preds = []
        for shadow_model in self.shadow_models:
            outputs = shadow_model(img.unsqueeze(0))
            probs = F.softmax(outputs, dim=1)
            score = probs[0, label]
            shadow_preds.append(score.item())
        return shadow_preds

    def _calculate_test_score(self, shadow_preds, img, label):
        """Calculate test score using shadow predictions and target model"""
        # Calculate shadow statistics
        shadow_logits = list(map(lambda x: np.log(x / (1 - x + 1e-30)), shadow_preds))
        mean_out = np.mean(shadow_logits)
        std_out = np.std(shadow_logits)

        # Get target model prediction
        outputs = self.target_model(img.unsqueeze(0))
        probs = F.softmax(outputs, dim=1)
        target_score = probs[0, label].item()
        target_logit = np.log(target_score / (1 - target_score + 1e-30))

        # Calculate test score using CDF
        return scipy.stats.norm.cdf(target_logit, mean_out, std_out + 1e-30)

def calculate_metrics(scores, memberships):
    fpr, tpr, _ = roc_curve(memberships, scores)
    
    # Calculate TPR@FPR=0.05
    idx = np.argmin(np.abs(fpr - 0.05))
    tpr_at_fpr = tpr[idx]
    actual_fpr = fpr[idx]
    
    # Calculate AUC
    roc_auc = auc(fpr, tpr)
    
    print(f"TPR@FPR=0.05: {tpr_at_fpr:.4f} (actual FPR: {actual_fpr:.4f})")
    print(f"AUC: {roc_auc:.4f}")
    
    return tpr_at_fpr, roc_auc

if __name__ == "__main__":
    # Load the model
    model = resnet18(pretrained=False)
    model.fc = nn.Linear(512, 44)
    ckpt = torch.load("/content/drive/MyDrive/Code/01_MIA_67.pt", map_location="cuda")
    model.load_state_dict(ckpt)
    model.eval()

    # Load the dataset
    pub_data: MembershipDataset = torch.load("pub.pt")
    
    # Create and train attack model
    attack = LiRAAttack(model, n_shadow_models=100, n_epochs=12,batch_size=64)
    attack.train_shadow_models(pub_data)
    
    # Evaluate on public data
    print("Evaluating on Public data...")
    pub_scores, pub_memberships = attack.compute_attack_scores(pub_data)
    tpr, roc_auc = calculate_metrics(pub_scores, pub_memberships)
    
    ###Log public dataset metrics
    metrics_df = pd.DataFrame({
        'n_shadow_models': [attack.n_shadow_models],
        'n_epochs': [attack.n_epochs],
        'tpr_at_fpr': [tpr],
        'auc': [roc_auc]
    })
    metrics_df.to_csv("public_metrics.csv", mode='a', header=not pd.io.common.file_exists("public_metrics.csv"), index=False)
    print("Public dataset metrics saved to public_metrics.csv")
    
    # Predict on private data
    print("Evaluating Private Data...")
    priv_data: MembershipDataset = torch.load("priv_out.pt")
    priv_scores = attack.predict_membership_scores(priv_data, "private_predictions.csv")
