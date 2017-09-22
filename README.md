# LogisticRegression
Goal of this project is to implement logistic regression classification using Logistic Regression without using Machine Learning Libraries.

# Dataset
I used the breast cancer dataset available from UCI machine learning library.

Number of Instances: 699
Number of missing data points: 16
Number of features: 10

Features Information:

   Attribute                     Domain
   1. Sample code number            id number
   2. Clump Thickness               1 - 10
   3. Uniformity of Cell Size       1 - 10
   4. Uniformity of Cell Shape      1 - 10
   5. Marginal Adhesion             1 - 10
   6. Single Epithelial Cell Size   1 - 10
   7. Bare Nuclei                   1 - 10
   8. Bland Chromatin               1 - 10
   9. Normal Nucleoli               1 - 10
  10. Mitoses                       1 - 10
  11. Class:                        (2 for benign, 4 for malignant)
  
# Dataset Processing
1. Remove the rows that are missing.
2. Remove the ID column.
3. Process all the features.
   x = (x - mean)/range
4. Convert y to use 0 and 1.

# Cross-validation
I use 10 fold cross-validation.

# Gradient descent
I use:
learning rate (alpha) = 0.01
number of iterations = 500

Hypothesis quations:
![H](/images/hypothesis.PNG?raw=true)
Gradient descent equation:
![GD](/images/gradient_descent.PNG?raw=true)

Cost function equation:
![C](/Images/cost.PNG?raw=true)

Gradient descent plot:
![GD](/images/gradient_descent_plot.PNG?raw=true)


