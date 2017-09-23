# LogisticRegression
Goal of this project is to implement binary classification using Logistic Regression without using any Machine Learning Libraries.

# Dataset
I implemented on two datasets.
  
* Breast cancer dataset  
  Number of Instances: 699 <br />
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

* Spambase dataset
  
  Number of Instances: 4600  
  Number of missing data points: None    
  Number of features: 57  

  The last column of 'spambase.data' denotes whether the e-mail was 
  considered spam (1) or not (0), i.e. unsolicited commercial e-mail.  
  Most of the attributes indicate whether a particular word or
  character was frequently occuring in the e-mail.  The run-length
  attributes (55-57) measure the length of sequences of consecutive 
  capital letters.  For the statistical measures of each attribute, 
  see the end of this file.  Here are the definitions of the attributes:

  1. 48 continuous real [0,100] attributes of type word_freq_WORD 
  = percentage of words in the e-mail that match WORD,
  i.e. 100 * (number of times the WORD appears in the e-mail) / 
  total number of words in e-mail.  A "word" in this case is any 
  string of alphanumeric characters bounded by non-alphanumeric 
  characters or end-of-string.

  2. 6 continuous real [0,100] attributes of type char_freq_CHAR
  = percentage of characters in the e-mail that match CHAR,
  i.e. 100 * (number of CHAR occurences) / total characters in e-mail

  3. 1 continuous real [1,...] attribute of type capital_run_length_average
  = average length of uninterrupted sequences of capital letters

  4. 1 continuous integer [1,...] attribute of type capital_run_length_longest
  = length of longest uninterrupted sequence of capital letters

  5. 1 continuous integer [1,...] attribute of type capital_run_length_total
  = sum of length of uninterrupted sequences of capital letters
  = total number of capital letters in the e-mail

  6. 1 nominal {0,1} class attribute of type spam
  = denotes whether the e-mail was considered spam (1) or not (0), 
  i.e. unsolicited commercial e-mail.  


# Dataset Processing
For breast cancer dataset:  
1. Remove the rows that are missing.
2. Remove the ID column.
3. Process all the features.
   x = (x - mean)/range
4. Convert y to use 0 and 1.
  
For spambase dataset:  
I used z-score normalization to normalize features.  

# Cross-validation
I use 10 fold cross-validation.

# Gradient descent
For breast cancer dataset:  
learning rate (alpha) = 0.01  
number of iterations = 500  
  
For spambase dataset:  
learning rate (alpha) = 0.00004  
number of iterations = 100   

Hypothesis quations:
![H](/images/hypothesis.png?raw=true)
Gradient descent equation:
![GD](/images/gradient_descent.png?raw=true)

Cost function equation:
![C](/images/cost.png?raw=true)



Gradient descent plot for breast cancer:
![GDP](/images/gradient_descent_plot_breast_cancer.png?raw=true)

Gradient descent plot for spambase:
![GDPS](/images/gd_plot_spam.png?raw=true)

Results:  
  
Breast Cancer dataset:  
Without Regularization  
Mean accuracy: 96.76  
With Regularization  
Mean accuracy: 97.05  
  
SpamBase dataset:  
Without Regularization  
Mean accuracy: 89.5  
With Regularization <br />
Mean accuracy: 89.52  

