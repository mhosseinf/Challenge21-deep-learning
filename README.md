# Challenge21-Alphabet Soup Charity - Deep Learning Model Analysis Report

## Overview
This analysis aims to create a deep learning model predicting the success of funding applications for Alphabet Soup Charity. The dataset is sourced from the `charity_data.csv` file, with "**IS_SUCCESSFUL**" as the target variable indicating funding success.

### Data Preprocessing

#### Target and Features
- **Targets:**
  - "IS_SUCCESSFUL"
- **Features:**
  - All columns excluding "EIN" and "NAME."

#### Binning and Encoding
- **APPLICATION_TYPE Binning:**
  - Applications with counts < 500 grouped into "**Other**."
- **CLASSIFICATION Binning:**
  - Classifications with counts < 100 grouped into "**Other**."
- **Categorical Encoding:**
  - One-hot encoding to convert categorical columns into numbers using `pd.get_dummies`.

### Changes Made

1. **Neurons:** Increased Number of neurons on the 2nd optimisation


|   Changes        | Before | 1st Optimisation | 2nd Optimization |3rd Optimization |
|------------------|--------|------------------|------------------|-----------------|
| 1st Layer Neurons| 8      | 64               | 64               |64               |
| 2nd Layer Neurons| 5      | 64               | 64               |64               |
| 3rd Layer Neurons| NA     | NA               | 32               |32               |
                


2. **Activation Functions:**Added additional hidden layers after the 1st optimisation.


|   Changes        |Before   | 1st Optimisation | 2nd Optimisation |3rd Optimisation |
|------------------|---------|------------------|------------------|-----------------|
| 1st Act Func     | ReLU    | ReLU             | ReLU             |ReLU             |
| 2nd Act Func     | ReLU    | Tanh             | Tanh             |Tanh             |
| 3rd Act Func     | NA      | NA               | Sigmoid          |Sigmoid          |
|Output Act Func   | Sigmoid | Sigmoid          | Sigmoid          |Sigmoid          |

 


  
3. **Epochs:**Trained for a higher number of epochs after 1st optimization

|   Changes        | Before | 1st Optimisation | 2nd Optimisation |3rd Optimisation |
|------------------|--------|------------------|------------------|-----------------|
| No of epochs     | 100    | 120              | 120              |120              |
|
  
 


4. **ModelCheckpoint Callback:**
   - Weights saved every 5 epochs.


5. **Data adjustment:**
   - Detected and handled outliers in the data by dropping the "ASK_AMT" column on the 3rd optimisation.

![handled outliers using boxplot](https://github.com/mhosseinf/Challenge21-deep-learning/assets/139053922/f1a5f06d-deae-4618-bbd1-24774fcf0595)


### Model summary 

1. **Before**

![model summary before opt](https://github.com/mhosseinf/Challenge21-deep-learning/assets/139053922/b9d380a4-45cd-45a1-b189-925271c9cc9a)


2-**1st Optimisation**

![1st Optimisation](https://github.com/mhosseinf/Challenge21-deep-learning/assets/139053922/a0008171-5877-41e6-ba2c-7aa07c782707) 



3-**2nd Optimisation**

![2nd Optimisation](https://github.com/mhosseinf/Challenge21-deep-learning/assets/139053922/a3e741b0-bf09-427f-aa5c-f2e9fe5b80a7)


3-**3rd Optimisation**

![3rd Optimisation](https://github.com/mhosseinf/Challenge21-deep-learning/assets/139053922/75e9e000-4a4b-4c13-98d5-b8a10ec6ea0e)


#### Results


| Metric           | Before | 1st Optimisation | 2nd Optimisation |3rd Optimisation |
|------------------|--------|------------------|------------------|-----------------|
| Training Loss    | 0.5528 | 0.5546           | 0.5548           |0.5883           |
| Training Accuracy| 72.54% | 72.77%           | 72.47%           |69.99%           |
| Eval. Loss       | 0.5528 | 0.5546           | 0.5548           |0.5883           |
| Eval. Accuracy   | 72.54% | 72.77%           | 72.47%           |69.99%           |






### Summarise the overall results of the deep learning optimisation
While there were slight improvements in training accuracy after the 1st optimization, the 2nd optimization showed a similar accuracy level but with more complex changes in the model architecture. The 3rd optimization, involving outlier handling, resulted in a decrease in accuracy, indicating a potential negative impact. 


### Recommendation for Further Model Refinement

Exploring alternative model architectures, such as Hyperparameter Tuning, may offer insights into achieving the desired accuracy target.


# Hyperparameter Tuning Results and Model Analysis

### Proposed Model Architecture
The hyperparameter tuning process has resulted in the following neural network architecture:

plaintext
Copy code
Model: "sequential"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 dense (Dense)               (None, 7)                 350       
                                                                 
 dense_1 (Dense)             (None, 9)                 72        
                                                                 
 dense_2 (Dense)             (None, 7)                 70        
                                                                 
 dense_3 (Dense)             (None, 1)                 8         
                                                                 
 dense_4 (Dense)             (None, 1)                 2         
                                                                 
=================================================================
Total params: 502 (1.96 KB)
Trainable params: 502 (1.96 KB)
Non-trainable params: 0 (0.00 Byte)

![otimised using hyperparameter options](https://github.com/mhosseinf/Challenge21-deep-learning/assets/139053922/f77d7ec0-e6c6-4eac-b520-83cad451ee33)


The proposed model consists of multiple dense layers with varying neuron counts.
The output layer utilises a sigmoid activation function, suitable for binary classification tasks.
The model has a total of 502 parameters, making it a relatively lightweight model.

### Model Performance
After training the model with the optimized hyperparameters, the evaluation results are as follows:

Loss: 0.5525
Accuracy: 0.7202




### Random Forest Model Results and Model Analysis 
The Random Forest model achieved an accuracy of approximately 70.94%. 


#### Decision Tree Model Results and Model Analysis
The Decision Tree model achieved an accuracy of approximately 70.54%. 


### Conclusion
In conclusion, while all three models showed reasonable accuracy, further experimentation and refinement are recommended to push accuracy beyond the current levels. The proposed model, after hyperparameter tuning, demonstrates better accuracy compared to the other models.

### Recommendation for Further Model Refinement
Ensemble Approaches: Explore more advanced ensemble methods like Gradient Boosting or XGBoost, which may further enhance model performance.

Feature Engineering: Investigate feature importance and potentially engineer or select features more effectively for model training.

Hyperparameter Tuning: Conduct more extensive hyperparameter tuning for both Random Forest and Decision Tree models to discover optimal settings.

Cross-Validation: Implement cross-validation techniques to ensure the robustness of the models.

Alternative Architectures: Continue exploring different deep learning architectures or even hybrid models combining deep learning with ensemble methods.



**Links:**
   - Reference to save and export your results to an HDF5 file [TensorFlow Documentation](https://www.tensorflow.org/tutorials/keras/save_and_load).