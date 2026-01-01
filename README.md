# Simple Multi Linear Regression
# Insurance Charges Prediction (Project 1)
## Dataset
This project uses the *Health Insurance Charges Dataset* from Kaggle. 

- Source: https://www.kaggle.com/datasets/nalisha/health-insurance-charges-dataset
- License: CC0 (Public Domain) — allowed for reuse and redistribution.

### Project Overview 
This project demonstrates the use of Multiple Linear Regression to estimate insurance charges of an individual based on :

- Age
- Body Mass Index (BMI)
- Smoking Status

The following are the steps I applied : 

- Extracted only the relevant features age, bmi and smoker and the label column charges from the original dataset
- Changed the categorical smoker table data from yes or no value to numerical 1 or 0 respectively
- Split the data into training and testing data
- Scaled the bmi column
- Used LinearRegression() function and trained the model using training data
- Calculated accuracy scores
- Plotted the predicted charges against acutal charges

### Conclusion

This project demonstrates a foundational implementation of Multiple Linear Regression. 

Thank you for reading!
