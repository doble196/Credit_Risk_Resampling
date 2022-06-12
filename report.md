# Module 12 Report Template

## Overview of the Analysis

In this section, describe the analysis you completed for the machine learning models used in this Challenge. This might include:

* Explain the purpose of the analysis.
    * train a Machine Learning Machine to classify healthy loas from high-risk loans.  
* Explain what financial information the data was on, and what you needed to predict.  
    * From the lending_data.csv file with:  `6 dimensions`  
          * loan_size,  
          * interest_rate,  
          * borrower_income,  
          * debt_to_income,num_of_accounts,  
          * derogatory_marks,total_debt,  
          * loan_status.
* Provide basic information about the variables you were trying to predict (e.g., `value_counts`).  
    * Four variables:  
          * balanced_accuracy_score  
          * RandomOverSampler  
          * LogisticRegression  
          * classification_report_imbalanced  
* Describe the stages of the machine learning process you went through as part of this analysis.  
      * Split the Data into Training and Testing Sets
      * Create a Logistic Regression Model with the Original Data
      * Predict a Logistic Regression Model with Resampled Training Data 

      ### Split the Data into Training and Testing Sets
      
      1. Read the `lending_data.csv` data from the `Resources` folder into a Pandas DataFrame.
      2. Create the labels set (`y`)  from the “loan_status” column, and then create the features (`X`) DataFrame from the remaining columns.

      > **Note** A value of `0` in the “loan_status” column means that the loan is healthy. A value of `1` means that the loan has  a high risk of defaulting.  
      3. Check the balance of the labels variable (`y`) by using the `value_counts` function.
      4. Split the data into training and testing datasets by using `train_test_split`.

      ### Create a Logistic Regression Model with the Original Data

* Employ your knowledge of logistic regression to complete the following steps:

      1. Fit a logistic regression model by using the training data (`X_train` and `y_train`).
      2. Save the predictions on the testing data labels by using the testing feature data (`X_test`) and the fitted model.
      3. Evaluate the model’s performance by doing the following:
          * Calculate the accuracy score of the model.
          * Generate a confusion matrix.
          * Print the classification report.
      4. Answer the following question: How well does the logistic regression model predict both the `0` (healthy loan) and `1` (high-risk loan) labels?

      ### Predict a Logistic Regression Model with Resampled Training Data

* Did you notice the small number of high-risk loan labels? Perhaps, a model that uses resampled data will perform better. You’ll   thus resample the training data and then reevaluate the model. Specifically, you’ll use `RandomOverSampler`.
To do so, complete the following steps:   
  1. Use the `RandomOverSampler` module from the imbalanced-learn library to resample the data. Be sure to confirm that the labels have an equal number of data points. 
  2. Use the `LogisticRegression` classifier and the resampled data to fit the model and make predictions.
  3. Evaluate the model’s performance by doing the following:
      * Calculate the accuracy score of the model.
      * Generate a confusion matrix.
      * Print the classification report.
  4. Answer the following question: How well does the logistic regression model, fit with oversampled data, predict both the `0` (healthy loan) and `1` (high-risk loan) labels?  

* Briefly touch on any methods you used (e.g., `LogisticRegression`, or any resampling method).  
          * balanced_accuracy_score  
          * RandomOverSampler  
          * LogisticRegression  
          * classification_report_imbalanced  
      ## Results
* Balanced accuracy scores and the precision and recall scores of all machine learning models.

* Machine Learning Model 1:
      * Description of Model 1 Accuracy, Precision, and Recall scores.

Classification Report
                   pre       rec       spe        f1       geo       iba       sup

          0       1.00      0.99      0.91      1.00      0.95      0.91     18765
          1       0.85      0.91      0.99      0.88      0.95      0.90       619

avg / total       0.99      0.99      0.91      0.99      0.95      0.91     19384


* `Accuracy` score is `%0.5093333333333334` for healthy loans.
  
    * `0` (`healthy loan`)  

          * `f1` score is `%99` harmonic mean of precision and recall.     
          * `Precision` score was perfect `%100` for transactions predicted as frauduent and were actually fraudulent.  
          * `Recall` score is `%99` for transactions labeled as not fraudulent but the transaction really was fraudulent.  

    * `1` (`high-risk loan`)  

          * `f1` score of `%88` harmonic mean of precision and recall.   
          * `precision` score is `%85` for transactions predicted as fraudulent but was not really fraudulent.  
          * `recall` score is `%91` for transactions predicted as fraudulent and really were fraudulent.  



* Machine Learning Model 2:
  * Description of Model 2 Accuracy, Precision, and Recall scores.

Classification Report on Resampled Data
                   pre       rec       spe        f1       geo       iba       sup

          0       1.00      0.99      0.99      1.00      0.99      0.99     18765
          1       0.84      0.99      0.99      0.91      0.99      0.99       619

avg / total       0.99      0.99      0.99      0.99      0.99      0.99     19384


* `Accuracy` score is `%52.09424083769633` 

    * `0` (`healthy loan`)  

          * `f1` score of `%99` harmonic mean of precision and recall. 
          * `Precision` score is perfect `%100` for transactions predicted as frauduent and were actually fraudulent.
          * `Recall` score is `%99` for transactions labeled as not fraudulent but the transaction really was fraudulent.  


  * `1` (`high-risk loan`) 
          * `f1` score of `%91` harmonic mean of precision and recall.  
          * `precision` score is `%84` for transactions predicted as fraudulent but was not really fraudulent.
          * `recall` score is `%99` for transactions predicted as fraudulent and really were fraudulent.
          

## Summary

Summarize the results of the machine learning models, and include a recommendation on the model to use, if any. For example:
* Which one seems to perform best? How do you know it performs best?
    * Answer: Machine Learniing Model 2 perfored %2 better than the 1st model. Only lost %1 percent to the 1st model.

* Does performance depend on the problem we are trying to solve? (For example, is it more important to predict the `1`'s, or predict the `0`'s? )  
    * Answer: yes the problem solving is dependent on getting the predictions that were false to being a true prediction therefore it is more important to predict the `1`s. these are the loans were the company is predicting incorrectly.  

* If you do not recommend any of the models, please justify your reasoning.  
    * Out of the two models yes the 2nd one with the resampled data is performing better but the accuracy score is low on both calculated at around `%50` and `%52`. These scores are `unnaceptible`. My recommendation is continue training until better results or create a new machine learning model with better performance.
