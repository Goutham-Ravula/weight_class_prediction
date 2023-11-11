# Weight Classification ML Model

![License](https://img.shields.io/badge/license-MIT-blue)

This repository contains the implementation of a machine learning model for weight classification. The model leverages diverse algorithms, effective data processing techniques, and ensemble methods to optimize accuracy. The project also includes a user-friendly prediction system for practical application.

## Table of Contents
1. [Introduction](#introduction)
2. [Dataset](#dataset)
3. [Project Objectives](#project-objectives)
4. [Data Processing](#data-processing)
5. [Algorithms Used](#algorithms-used)
6. [Ensemble Method](#ensemble-method)
7. [Prediction System](#prediction-system)
8. [Project Conclusion](#project-conclusion)
9. [Disclaimer](#disclaimer)
10. [License](#license)

## Introduction
TThe Weight Class Prediction Model is a machine learning project designed to predict an individual's weight status based on a set of input features. The primary goal of this project is to provide a user-friendly tool that allows users to understand and assess their weight based on various lifestyle and demographic factors.

The project leverages a machine learning ensemble approach, combining multiple predictive models to enhance the accuracy and robustness of the predictions. Each model has been trained on a dataset of individuals with known weight statuses and corresponding features.

## Dataset
The dataset used for this project is sourced from [[UC Irvine Machine Learning Repository]](https://archive.ics.uci.edu/dataset/544/estimation+of+obesity+levels+based+on+eating+habits+and+physical+condition). It includes essential features such as age, height, and gender, contributing to the classification of weight classes. The dataset plays a pivotal role in training and evaluating the machine learning models implemented in this project.

## Project Objectives
The primary objectives of this project are as follows:

1. **Data Encoding:** Implement encoding techniques, including ordinal and label encoding, to transform categorical features into a numerical format suitable for machine learning.

2. **Algorithm Implementation:** Utilize a diverse set of algorithms to train the model, including:
   - Support Vector Machine (SVM)
   - Light Gradient Boosting
   - Logistic Regression
   - Gradient Boosting
   - Decision Tree
   - Random Forest Classifier
   - Cat Boost
   - XG Boost

3. **Hyperparameter Tuning:** Optimize the performance of the algorithms through hyperparameter tuning. This step involves fine-tuning the parameters to achieve better accuracy and generalization.

4. **Ensemble Method:** Implement an ensemble method using majority voting. Combine predictions from individual models to enhance overall accuracy and robustness.

5. **Prediction System:** Conclude the project with a user-friendly prediction system. Allow users to input relevant features (age, height, gender) and obtain a predicted weight class. This system enhances practicality and user interaction.

By achieving these objectives, the project aims to deliver an accurate, versatile, and user-friendly weight classification system.

## Data Processing

### Features (X Variables)
The dataset consists of the following features:

1. **Age**
2. **Gender**
3. **Height**
4. **Weight**
5. **family history with overweight**
6. **Frequent consumption of high caloric food**
7. **Frequency of consumption of vegetables**
8. **Number of main meals**
9. **Consumption of food between meals**
10. **Smoking habit**
11. **Consumption of water daily**
12. **Calories consumption monitoring**
13. **Physical activity frequency**
14. **Time using technology devices**
15. **Consumption of alcohol**
16. **Transportation used**

### Target Variable (Y Variable)
The target variable, obesity class, represents the weight categories that individuals fall into. To establish a systematic representation of classes, the Label Encoding technique is applied to the obesity class. This ensures a numerical association with each class, making it compatible with machine learning algorithms.

### Label Encoding
Label Encoding is employed for the target variable (Y) to convert categorical classes into numerical values. This encoding establishes a meaningful order between different obesity classes, facilitating model training and evaluation.

### Ordinal Encoding
Ordinal Encoding is applied to categorical features in the X variables to represent their order or rank. This encoding technique ensures that the model understands the relationship between different categories within these features.

By implementing both Label Encoding and Ordinal Encoding, the data is transformed into a suitable format for training diverse machine learning models.

## Algorithms Used
The project leverages a diverse set of machine learning algorithms to train and optimize the model for accurate weight classification. Here's a brief overview of the eight algorithms employed:

1. **Support Vector Machine (SVM):**
   - SVM is a powerful algorithm used for classification and regression tasks. It works by finding the hyperplane that best separates data points into different classes.

2. **Light Gradient Boosting (LGB):**
   - LGB is a gradient boosting framework that uses tree-based learning. It is known for its efficiency, speed, and accuracy, making it suitable for large datasets.

3. **Logistic Regression (LR):**
   - Logistic Regression is a widely used algorithm for binary and multiclass classification. It models the probability of a certain class and makes predictions based on a threshold.

4. **Gradient Boosting (GBC):**
   - Gradient Boosting is an ensemble learning method that builds a series of weak learners and combines their predictions to improve accuracy.

5. **Decision Tree (DTC):**
   - Decision Tree is a simple yet effective algorithm that makes decisions based on a set of rules. It recursively splits the dataset to create a tree-like structure.

6. **Random Forest Classifier (RFC):**
   - Random Forest is an ensemble algorithm that constructs multiple decision trees during training and outputs the mode of the classes as the prediction.

7. **Cat Boost:**
   - Cat Boost is a gradient boosting library that handles categorical features efficiently. It provides high performance with default hyperparameters.

8. **XG Boost:**
   - XG Boost is another popular gradient boosting library known for its speed and performance. It implements parallel processing and regularization techniques for better results.

Each algorithm brings its unique strengths to the ensemble, contributing to a robust and accurate weight classification model.

## Ensemble Method
The ensemble method is a key component of this project, enhancing predictive performance by combining the strengths of multiple individual models. After training the eight algorithms mentioned above, the ensemble method aggregates their predictions using a majority vote.

### Majority Voting
The majority voting approach considers the class predicted by the majority of individual models as the final prediction. In other words, each model "votes" for a particular class, and the class with the most votes becomes the ensemble's prediction. This technique is particularly effective in reducing overfitting and enhancing overall accuracy.

The combination of different algorithms through ensemble learning helps mitigate the weaknesses of individual models and improves the model's robustness and generalization to new data. The use of majority voting ensures a more reliable final prediction, making the weight classification model more robust and accurate.

## Prediction System
The project concludes with a user-friendly prediction system that allows users to input data for all 16 features and receive a weight class prediction. This user-driven prediction system serves as an intuitive interface, making the model accessible and practical for real-world applications.

### User Input
Users can input values for the 16 features, representing various physiological and lifestyle attributes. 

### Weight Class Prediction
The model processes the user input through the trained ensemble of algorithms, leveraging the majority voting approach. By considering the combined predictions of diverse algorithms, the system delivers an accurate weight class prediction based on the provided feature values.

## Project Conclusion
In conclusion, this project undertakes the task of developing a comprehensive machine learning model for weight classification. By employing advanced data processing techniques, including label and ordinal encoding, we enhance the model's ability to interpret and learn from diverse datasets. The use of sophisticated algorithms, such as Support Vector Machines, Gradient Boosting, and Random Forest, ensures a robust and accurate weight classification system.

The hyperparameter tuning of the selected algorithms optimizes their performance, leading to a more refined and precise model. The ensemble method, implemented through majority voting, further strengthens the model's predictive capabilities by leveraging the diverse strengths of individual algorithms.

The culmination of this project is a user-friendly prediction system, allowing individuals to input their physiological and lifestyle data and receive personalized weight class predictions. This holistic approach, combining advanced machine learning techniques with user-centric features, establishes a versatile tool for weight status assessment.

Please refer to the [License](#license) section for the usage terms and conditions of this project.

## Disclaimer
This machine learning model for weight classification is developed for educational and experimental purposes. The predictions and assessments made by the model should not substitute professional medical advice or diagnosis. Users are encouraged to consult with healthcare professionals for accurate and personalized weight-related guidance.

The developers disclaim any responsibility for the misuse or misinterpretation of the model's results. The accuracy and reliability of predictions may vary, and the model is not intended as a substitute for professional healthcare services.

Use this tool responsibly, and be aware that its predictions are based solely on input data and statistical patterns. Always consult with qualified healthcare professionals for personalized advice and assessments.

## License
![License](https://img.shields.io/badge/license-MIT-blue)

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
