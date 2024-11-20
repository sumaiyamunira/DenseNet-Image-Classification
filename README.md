# DenseNet Image Classification with Big Data Analytics

## Project Overview

This project explores image classification in the realm of **Big Data Analytics**, using advanced **deep learning techniques**. The primary focus is on enhancing the performance of a deep learning model trained on the **ImageNet** and **ImageNetV2** datasets by analyzing pre-trained deep features. These features are extracted from an image recognition model, and the task is to understand and improve the model's performance on two different test sets.

The analysis aims to identify and address any performance gaps between two test sets by evaluating the model’s performance on both. The project also hypothesizes that specific **pre-processing techniques** and the **distribution of features** across the datasets could impact model accuracy and generalization.

## Dataset Overview
The project leverages three key CSV files, which contain the deep features extracted from an image recognition model:

- **train_eva02_large_patch14_448.mim_m38m_ft_in22k_in1k.csv** (Training data, 13 GB)
- **val_eva02_large_patch14_448.mim_m38m_ft_in22k_in1k.csv** (Test Set 1, 100 MB)
- **v2_eva02_large_patch14_448.mim_m38m_ft_in22k_in1k.csv** (Test Set 2, 550 MB)

**Note**: Due to the large size of these datasets, they are **not included** in this repository. You will need to download these files separately and place them in the same directory as the project or update the code paths accordingly.

## Objectives

The main goals of the project are:

1. **Feature Extraction**: Extract deep features from both the training and testing datasets.
2. **Classifier Training**: Train a classifier using the extracted deep features.
3. **Performance Evaluation**: Evaluate model performance on both test sets, comparing the results to identify performance gaps.
4. **Gap Analysis**: Investigate reasons behind any differences in performance between the two test sets.
5. **Model Optimization**: Explore techniques to enhance the model’s generalization, especially on **Test Set 2**.

## Methodology

1. **Pre-trained Feature Extraction**: The pre-trained model features will be extracted from the provided datasets, which represent images processed through a pre-trained deep learning model.
   
2. **Data Preprocessing**: A set of pre-processing techniques will be applied to the extracted features, including normalization, dimensionality reduction, and potential augmentation.

3. **Model Training**: A classifier will be trained on the pre-processed features. The model will be evaluated on both Test Set 1 and Test Set 2.

4. **Evaluation**: The trained model’s performance will be assessed using metrics such as accuracy, precision, recall, F1-score, and confusion matrix.

5. **Analysis**: We will compare the feature distributions between the test sets and perform in-depth analysis to determine why there may be performance discrepancies.

## Requirements

### Python Libraries

This project requires Python 3.6+ and several Python libraries for deep learning, data processing, and analysis. Install the dependencies with:

```bash
pip install tensorflow pandas numpy scikit-learn dask matplotlib seaborn
```

## Hardware Requirements
Given the large dataset sizes (13 GB, 100 MB, and 550 MB), the following hardware requirements are recommended:
- **Substantial RAM**: At least 16 GB (preferably more).
- **GPU** (for training the model) is highly recommended.
- **Storage**: Enough space to store the large CSV files and any intermediate files generated.

## How to Use
1. **Download the Data**: Download the following CSV files (not included in the repository):
    - `train_eva02_large_patch14_448.mim_m38m_ft_in22k_in1k.csv`
    - `val_eva02_large_patch14_448.mim_m38m_ft_in22k_in1k.csv`
    - `v2_eva02_large_patch14_448.mim_m38m_ft_in22k_in1k.csv`

2. **Setup the Environment**: Ensure that the datasets are stored in the project directory or specify the paths to these files in the provided Python scripts.

3. **Run the Code**: Once the data is ready, you can train the model and evaluate its performance.

    Example command to start training the model:

    ```bash
    python main_DenseNet.py
    ```

    This will extract features, preprocess them, train the classifier, and evaluate the performance on both test sets.

4. **Results**: The model’s evaluation metrics will be displayed, including:
    - **Accuracy**
    - **Precision, Recall, F1-Score**
    - **Confusion Matrix**

5. **Analysis**: Detailed analysis will be provided in the results, including comparisons between Test Set 1 and Test Set 2.



