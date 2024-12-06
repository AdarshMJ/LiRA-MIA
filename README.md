# LiRA-MIA

This is a PyTorch implementation of Membership Inference Attack proposed in the paper - Membership Inference Attacks From First Principles (https://arxiv.org/abs/2112.03570).
The implementation is based on Algorithm 1, the MIA attack is called LiRA - Likelihood Ratio Test Attack, I implemented here the offline version.

## Task at hand

1. We are interested in discerning if a certain data point was used in the training of a target model. In our current task we have been given a ResNet-18 model checkpoint. We do not have access to information on how the model was trained, that is, the learning rate, dropout, regularization etc.
2. We are given 2 datasets a PUB dataset which has membership labels (1 for member), the goal here is to achieve the highest TPR@FPR=0.05 by predicting if the data point was a member(1) or non-member (0). The other dataset is called PRIV which do not have the membership labels and the goal is to output continuous membership values.

## Analysing the dataset
I first initially probed the dataset to see how the dataset looks like, if there is some kind of data imbalance. We can check this by running the file 
```python
python probedataset.py
```
We get the following stats -

```python
Dataset Statistics:
--------------------------------------------------
Total samples: 20000
Number of classes: 44

Class Distribution Summary:
Min samples per class: 4 (Class 35)
Max samples per class: 3281 (Class 17)

Membership Distribution:
Non-members (0): 10000 (50.00%)
Members (1): 10000 (50.00%)
```
Notice how there are certain data points belonging to Class 35 that have only 4 samples per class. This would constitute a minor class and if we want our MIA to be successful we need to ensure we sample the datapoints in a way that would circumvent this imbalance.
