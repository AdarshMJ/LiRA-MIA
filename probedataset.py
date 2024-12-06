import torch
import numpy as np
import pandas as pd
from typing import Any, Tuple
from torch.utils.data import Dataset

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

def probe_dataset(dataset: Any) -> None:

    print("\nDataset Statistics:")
    print("-" * 50)
    
    # Overall dataset size
    print(f"Total samples: {len(dataset)}")
    print(f"Number of classes: {len(np.unique(dataset.labels))}")
    
    # Class distribution
    label_counts = np.bincount(dataset.labels)
    df_stats = pd.DataFrame({
        'Class': range(len(label_counts)),
        'Count': label_counts,
        'Percentage': (label_counts / len(dataset.labels) * 100).round(2)
    })
    
    print("How the dataset looks like :")
    print(f"Min samples per class: {label_counts.min()} (Class {label_counts.argmin()})")
    print(f"Max samples per class: {label_counts.max()} (Class {label_counts.argmax()})")
 
    # Membership distribution
    member_counts = np.bincount(dataset.membership)
    print("\nMembership Distribution:")
    print(f"Non-members (0): {member_counts[0]} ({member_counts[0]/len(dataset)*100:.2f}%)")
    print(f"Members (1): {member_counts[1]} ({member_counts[1]/len(dataset)*100:.2f}%)")
    
    # Save detailed statistics to CSV
    df_stats.to_csv("PUB_dataset_stats.csv", index=False)

if __name__ == "__main__":
    # Example usage
    pub_data: MembershipDataset = torch.load("pub.pt")
    probe_dataset(pub_data)