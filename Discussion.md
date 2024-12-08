# Results and Discussion

### Training details
1. Shadow models = 2,5,10,20,50,100
2. LR = 0.001
3. Number of epochs tested = 10,20,50
4. ReduceLORonPlateau with patience = 7
5. Adam optimizer

### PUB Dataset
The LiRA method needs greater number of reference/shadow models to be trained for the empirical distributions to make sense and the $mu_{out}$ and $\sigma^2_{out}$ to be meaningful. I have trained shadow models = 2,5,10,20,100 and below I report the TPR@FPR for the PUB dataset in the table below.

| n_shadow_models | n_epochs | tpr_at_fpr | auc          |
|------------------|----------|------------|--------------|
| 2               | 10       | 0.0567     | 0.498278175  |
| 5               | 10       | 0.0543     | 0.50317402   |
| 10              | 10       | 0.0499     | 0.499587235  |
| 20              | 10       | 0.0508     | 0.50074731   |
| 100             | 12       | 0.0514     | 0.500463805  |

### PRIV Dataset

For this dataset, the only way I could evaluate if my method works is by posting it on the server and waiting for an hour. The results are given in the table below. I have also included a ```prediction_logs.csv``` file which shows the responses I have received.

| Timestamp           | Input_File             | TPR@FPR=0.05                     | AUC                        |
|---------------------|------------------------|----------------------------------|----------------------------|
| 2024-12-07 06:50:59 | private_predictions_50.csv | 0.044                            | 0.5013378333333334         |
| 2024-12-07 08:07:56 | private_predictions_100.csv | 0.044                            | 0.4986865                  |
| 2024-12-07 09:30:56 | private_predictions_20.csv | 0.04766666666666667             | 0.4995845                  |
| 2024-12-07 11:21:48 | private_predictions_10.csv | 0.04666666666666667             | 0.5020166111111111         |
| 2024-12-07 12:28:23 | PrivateUpdated_10.csv    | 0.036333333333333336            | 0.49953266666666674        |
| 2024-12-07 14:04:12 | private_preds_updated_3.csv    | 0.049666666666666665            | 0.49227183333333335  |
|2024-12-08 06:59:30 |private_predictions_5.csv| 0.05266666666666667|0.5026543888888889|




