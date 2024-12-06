# LiRA-MIA

This is a PyTorch implementation of Membership Inference Attack proposed in the paper [1]. The implementation is based on Algorithm 1, the MIA attack is called LiRA - Likelihood Ratio Test Attack, I implemented here the offline version.

## Task at hand

1. We are interested in discerning if a certain data point was used in the training of a target model. In our current task we have been given a ResNet-18 model checkpoint. We do not have access to information on how the model was trained, that is, the learning rate, dropout, regularization etc.
2. We are given 2 datasets a PUB dataset which has membership labels (1 for member), the goal here is to achieve the highest TPR@FPR=0.05 by predicting if the data point was a member(1) or non-member (0). The other dataset is called PRIV which do not have the membership labels and the goal is to output continuous membership values.

## Analysing the dataset
I first initially probed the dataset to see how the dataset looks like, if there is some kind of data imbalance. We can check this by running the file 
```python
python probedataset.py
```
We get the following stats for PUB dataset -

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
and for PRIV dataset

```python 
Dataset Statistics:
--------------------------------------------------
Total samples: 20000
Number of classes: 44
How the dataset looks like :
Min samples per class: 4 (Class 35)
Max samples per class: 3305 (Class 17)
```

Notice how there are certain data points belonging to Class 35 that have only 4 samples per class. This would constitute a minor class and if we want our MIA to be successful we need to ensure we sample the datapoints in a way that would circumvent this imbalance. 

## How it works?
Below here I detail how the algorithm works -

1. This the LiRA offline attack which is computationally friendly since we do not train shadow models on the target data points. We train $N = k$ shadow models on a random disjoint sample of the dataset, where $k = {2,10,50,100}$.
2. Note that, since we observe the given datasets are class imbalanced, we account for this by weighted sampling. That is, we give higher weight to an under-represented class,
```python
label_counts = np.bincount([label for _, _, label, _ in dataset])
class_weights = 1. / label_counts
class_weights = class_weights / class_weights.sum()           
# Weighted sampling for better class balance
weights = [class_weights[label] for _, _, label, _ in dataset]
weights = np.array(weights) / sum(weights)
indices = np.random.choice(len(dataset), size=len(dataset)//2, replace=False, p=weights)
```
3. We also take care of this in the loss function by passing ```class_weights``` as the parameter to the loss function

```python
label_counts = np.bincount([dataset.labels[i] for i in indices])
class_weights = torch.FloatTensor(1. / label_counts).to(self.device)
criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = optim.Adam(model.parameters(), lr=0.01)
```
4. The LiRA method works by constructing empirical distributions $\mathbb{\tilde{Q}}{in}(x,y)$ where the data point was included and $\mathbb{\tilde{Q}}{out}(x,y)$ where the data point was excluded by approximating them as Gaussians. The likelihood is calculated as follows $p(l(f(x),y)) | \mathbb{\tilde{Q}}_{in/out}(x,y)$. To ensure the cross-entropy loss is accurately captured by the Normal distribution we apply a logit scaling and calculate the score, the higher the score is the more likely the data point is a member (Equation 4 in the paper).

```Python
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
```

5. Finally we calculate the TPR@FPR = 0.05 for the PUB dataset and output continuous membership values for the PRIV_OUT dataset.

## References 
[1]  Membership Inference Attacks From First Principles (https://arxiv.org/abs/2112.03570)

[2]  My implementation is different from this but it helped me get an idea on how the logit scaling is performed. https://github.com/DevPranjal/mico-first-principles/tree/master

[3] The original implementation by the authors - https://github.com/tensorflow/privacy/tree/master which is in Tensorflow.
