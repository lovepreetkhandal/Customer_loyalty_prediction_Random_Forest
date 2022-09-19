
# Predicting Customer Loyalty Using Regression Models
![image](https://img.favpng.com/23/8/1/shopping-cart-png-favpng-s0swYX7XaNDeEwnY8wKnSzvF1.jpg)


Our client, a grocery retailer, hired a market research consultancy to append market level customer loyalty information to the database. However, only around 50% of the client’s customer base could be tagged, thus the other half did not have this information present. Let’s use ML to solve this!
# Project Overview
## Context

The overall aim of this work is to accurately predict the loyalty score for those customers who could not be tagged, enabling our client a clear understanding of true customer loyalty, regardless of total spend volume - and allowing for more accurate and relevant customer tracking, targeting, and comms.

To achieve this, we looked to build out a predictive model that will find relationships between customer metrics and loyalty score for those customers who were tagged, and use this to predict the loyalty score metric for those who were not.

## Actions
We firstly needed to compile the necessary data from tables in the database, gathering key customer metrics that may help predict loyalty score, appending on the dependent variable, and separating out those who did and did not have this dependent variable present.

As we are predicting a numeric output, we tested three regression modelling approaches, namely:

* Linear Regression
* Decision Tree
* Random Forest

## Results
Our testing found that the Random Forest had the highest predictive accuracy.

#### Metric 1: Adjusted R-Squared (Test Set)

* Random Forest = 0.955
* Decision Tree = 0.886
* Linear Regression = 0.754
#### Metric 2: R-Squared (K-Fold Cross Validation, k = 4)

* Random Forest = 0.925
* Decision Tree = 0.871
* Linear Regression = 0.853
As the most important outcome for this project was predictive accuracy, rather than explicitly understanding weighted drivers of prediction, we chose the Random Forest as the model to use for making predictions on the customers who were missing the loyalty score metric.

## Growth/Next Steps
While predictive accuracy was relatively high - other modelling approaches could be tested, especially those somewhat similar to Random Forest, for example XGBoost, LightGBM to see if even more accuracy could be gained.

From a data point of view, further variables could be collected, and further feature engineering could be undertaken to ensure that we have as much useful information available for predicting customer loyalty.

## Key Definition
The loyalty score metric measures the % of grocery spend (market level) that each customer allocates to the client vs. all of the competitors.

Example 1: Customer X has a total grocery spend of $100 and all of this is spent with our client. Customer X has a loyalty score of 1.0

Example 2: Customer Y has a total grocery spend of $200 but only 20% is spent with our client. The remaining 80% is spend with competitors. Customer Y has a customer loyalty score of 0.2.

# Data Overview
We will be predicting the loyalty_score metric. This metric exists (for half of the customer base) in the loyalty_scores table of the client database.

The key variables hypothesised to predict the missing loyalty scores will come from the client database, namely the transactions table, the customer_details table, and the product_areas table.

Using pandas in Python, we merged these tables together for all customers, creating a single dataset that we can use for modelling.

![9](https://user-images.githubusercontent.com/100878908/190929216-41cfe65d-4534-43c2-98ec-3ebbb7a7e29d.png)

After this data pre-processing in Python, we have a dataset for modelling that contains the following fields…

![10](https://user-images.githubusercontent.com/100878908/190929269-cbc5395e-2b87-4d71-aae3-e92d5b95eb90.png)

# Modelling Overview
We will build a model that looks to accurately predict the “loyalty_score” metric for those customers that were able to be tagged, based upon the customer metrics listed above.

If that can be achieved, we can use this model to predict the customer loyalty score for the customers that were unable to be tagged by the agency.

As we are predicting a numeric output, we tested three regression modelling approaches, namely:

* Linear Regression
* Decision Tree
* Random Forest

# Linear Regression
We utlise the scikit-learn library within Python to model our data using Linear Regression. The code sections below are broken up into 4 key sections:

* Data Import
* Data Preprocessing
* Model Training
* Performance Assessment
## Data Import
Since we saved our modelling data as a pickle file, we import it. We ensure we remove the id column, and we also ensure our data is shuffled.

![11](https://user-images.githubusercontent.com/100878908/190929369-5486d50a-dcd0-4b3b-98e2-47cc265ef503.png)

## Data Preprocessing
For Linear Regression we have certain data preprocessing steps that need to be addressed, including:

* Missing values in the data
* The effect of outliers
* Encoding categorical variables to numeric form
* Multicollinearity & Feature Selection

### Missing Values
The number of missing values in the data was extremely low, so instead of applying any imputation (i.e. mean, most common value) we will just remove those rows

![12](https://user-images.githubusercontent.com/100878908/190929514-8bfe2406-33c2-4b59-bc65-8535aecb38cc.png)

### Outliers

The ability for a Linear Regression model to generalise well across all data can be hampered if there are outliers present. There is no right or wrong way to deal with outliers, but it is always something worth very careful consideration - just because a value is high or low, does not necessarily mean it should not be there!

In this code section, we use .describe() from Pandas to investigate the spread of values for each of our predictors. The results of this can be seen in the table below.

![13](https://user-images.githubusercontent.com/100878908/190929603-05e865bf-d031-45c3-b173-a470094f5682.png)

Based on this investigation, we see some max column values for several variables to be much higher than the median value.

This is for columns distance_from_store, total_sales, and total_items

For example, the median distance_to_store is 1.645 miles, but the maximum is over 44 miles!

Because of this, we apply some outlier removal in order to facilitate generalisation across the full dataset.

We do this using the “boxplot approach” where we remove any rows where the values within those columns are outside of the interquartile range multiplied by 2.

![14](https://user-images.githubusercontent.com/100878908/190929604-0332b898-7705-4c6e-bc85-ea3a286e55cc.png)

## Split Out Data For Modelling
In the next code block we do two things, we firstly split our data into an X object which contains only the predictor variables, and a y object that contains only our dependent variable.

Once we have done this, we split our data into training and test sets to ensure we can fairly validate the accuracy of the predictions on data that was not used in training. In this case, we have allocated 80% of the data for training, and the remaining 20% for validation.

![15](https://user-images.githubusercontent.com/100878908/190929732-378b35a4-ed96-4b0e-8ec4-da4897e2719f.png)

## Categorical Predictor Variables
In our dataset, we have one categorical variable gender which has values of “M” for Male, “F” for Female, and “U” for Unknown.
The Linear Regression algorithm can’t deal with data in this format as it can’t assign any numerical meaning to it when looking to assess the relationship between the variable and the dependent variable.

As gender doesn’t have any explicit order to it, in other words, Male isn’t higher or lower than Female and vice versa - one appropriate approach is to apply One Hot Encoding to the categorical column.

For ease, after we have applied One Hot Encoding, we turn our training and test objects back into Pandas Dataframes, with the column names applied.

![16](https://user-images.githubusercontent.com/100878908/190929733-86b2e092-4ec5-4bc1-9c98-3afe13902e85.png)

## Feature Selection
Feature Selection is the process used to select the input variables that are most important to your Machine Learning task. It can be a very important addition or at least, consideration, in certain scenarios. The potential benefits of Feature Selection are:

* Improved Model Accuracy - eliminating noise can help true relationships stand out
* Lower Computational Cost - our model becomes faster to train, and faster to make predictions
* Explainability - understanding & explaining outputs for stakeholder & customers becomes much easier

For our task we applied a variation of Reursive Feature Elimination called Recursive Feature Elimination With Cross Validation (RFECV) where we split the data into many “chunks” and iteratively trains & validates models on each “chunk” seperately. This means that each time we assess different models with different variables included, or eliminated, the algorithm also knows how accurate each of those models was. From the suite of model scenarios that are created, the algorithm can determine which provided the best accuracy, and thus can infer the best set of input variables to use!

![17](https://user-images.githubusercontent.com/100878908/190929899-b00f8837-7c10-48a8-8a54-a200d5de05f2.png)

The plot below shows us that the highest cross-validated accuracy (0.8635) is actually when we include all eight of our original input variables. This is marginally higher than 6 included variables, and 7 included variables. We will continue on with all 8!

![18](https://user-images.githubusercontent.com/100878908/190929902-8be1f614-9a35-46a4-bdc1-2d0725e7695e.png)

## Model Training
Instantiating and training our Linear Regression model is done using the below code:

![19](https://user-images.githubusercontent.com/100878908/190930255-137cb89e-78a1-4f2d-9131-81ce471140d4.png)

## Model Performance Assessment
### Predict On The Test Set
To assess how well our model is predicting on new data - we use the trained model object (here called regressor) and ask it to predict the loyalty_score variable for the test set
![20](https://user-images.githubusercontent.com/100878908/190930260-ec87ad2c-874f-42fb-8dba-d28b26325f0d.png)

### Calculate R-Squared
R-Squared is a metric that shows the percentage of variance in our output variable y that is being explained by our input variable(s) x. It is a value that ranges between 0 and 1, with a higher value showing a higher level of explained variance. Another way of explaining this would be to say that, if we had an r-squared score of 0.8 it would suggest that 80% of the variation of our output variable is being explained by our input variables - and something else, or some other variables must account for the other 20%

To calculate r-squared, we use the following code where we pass in our predicted outputs for the test set (y_pred), as well as the actual outputs for the test set (y_test)

![21](https://user-images.githubusercontent.com/100878908/190930264-3021d84d-3c3a-40c5-911c-52e567368960.png)
The resulting r-squared score from this is 0.78

### Calculate Cross Validated R-Squared
An even more powerful and reliable way to assess model performance is to utilise Cross Validation.

![22](https://user-images.githubusercontent.com/100878908/190930265-f1f2ab46-42bf-4355-8f69-3ddbcea9909c.png)
The mean cross-validated r-squared score from this is 0.853

### Calculate Adjusted R-Squared
Adjusted R-Squared is a metric that compensates for the addition of input variables, and only increases if the variable improves the model above what would be obtained by probability. It is best practice to use Adjusted R-Squared when assessing the results of a Linear Regression with multiple input variables, as it gives a fairer perception the fit of the data.

![23](https://user-images.githubusercontent.com/100878908/190930270-730026b5-06d6-47e6-922e-15f162021eec.png)

The resulting adjusted r-squared score from this is 0.754 which as expected, is slightly lower than the score we got for r-squared on it’s own.

## Model Summary Statistics
Although our overall goal for this project is predictive accuracy, rather than an explcit understanding of the relationships of each of the input variables and the output variable, it is always interesting to look at the summary statistics for these.

![24](https://user-images.githubusercontent.com/100878908/190930275-dc76f4be-d682-481d-aacf-24338ca7c70b.png)

The information from that code block can be found in the table below:

![25](https://user-images.githubusercontent.com/100878908/190930278-4518cc9e-3c8a-42a3-9e0f-82482d584966.png)

For each input variable, the coefficient value we see above tells us, with everything else staying constant how many units the output variable (loyalty score) would change with a one unit change in this particular input variable.

To provide an example of this - in the table above, we can see that the distance_from_store input variable has a coefficient value of -0.201. This is saying that loyalty_score decreases by 0.201 (or 20% as loyalty score is a percentage, or at least a decimal value between 0 and 1) for every additional mile that a customer lives from the store. This makes intuitive sense, as customers who live a long way from this store, most likely live near another store where they might do some of their shopping as well, whereas customers who live near this store, probably do a greater proportion of their shopping at this store…and hence have a higher loyalty score!

# Decision Tree'
We will again utlise the scikit-learn library within Python to model our data using a Decision Tree. The code sections below are broken up into 4 key sections:

* Data Import
* Data Preprocessing
* Model Training
* Performance Assessment

## Data Import
Since we saved our modelling data as a pickle file, we import it. We ensure we remove the id column, and we also ensure our data is shuffled.

![26](https://user-images.githubusercontent.com/100878908/190930536-e3727c96-2dd3-430f-8ba3-79b2dc2e0742.png)

## Data Preprocessing
While Linear Regression is susceptible to the effects of outliers, and highly correlated input variables - Decision Trees are not, so the required preprocessing here is lighter. We still however will put in place logic for:

* Missing values in the data
* Encoding categorical variables to numeric form

For missing values and encoding categorical variables to numeric form, we use the same code as mentioned above in the Linear Regression section. Please refer to that section if you are interested.


## Model Training
Instantiating and training our Decision Tree model is done using the below code. We use the random_state parameter to ensure we get reproducible results, and this helps us understand any improvements in performance with changes to model hyperparameters.


## Model Performance Assessment
### Predict On The Test Set
To assess how well our model is predicting on new data - we use the trained model object (here called regressor) and ask it to predict the loyalty_score variable for the test set
* R-square: 0.87
* Adjusted R Square: 0.85
* Cross-validated R Square: 0.84

## Decision Tree Regularisation
Decision Tree’s can be prone to over-fitting, in other words, without any limits on their splitting, they will end up learning the training data perfectly. We would much prefer our model to have a more generalised set of rules, as this will be more robust & reliable when making predictions on new data.

One effective method of avoiding this over-fitting, is to apply a max depth to the Decision Tree, meaning we only allow it to split the data a certain number of times before it is required to stop.

Unfortunately, we don’t necessarily know the best number of splits to use for this - so below we will loop over a variety of values and assess which gives us the best predictive performance!

![29](https://user-images.githubusercontent.com/100878908/190932490-7eead8e8-944e-4ce3-b97f-2271c77d1fc1.png)

That code gives us the below plot - which visualises the results!

![30](https://user-images.githubusercontent.com/100878908/190932492-6400b52e-db16-4deb-9267-997c6bd24ec6.png)


## Model Performance Assessment after using max depth (7)

![31](https://user-images.githubusercontent.com/100878908/190932606-3b7ba16f-c792-4677-a2b7-1557c134692e.png)

One interesting thing to note is that the very first split appears to be using the variable distance from store so it would seem that this is a very important variable when it comes to predicting loyalty!

# Random Forest
We will again utlise the scikit-learn library within Python to model our data using a Random Forest. The code sections below are broken up into 4 key sections:

* Data Import
* Data Preprocessing
* Model Training
* Performance Assessment

## Data Import
Again, since we saved our modelling data as a pickle file, we import it. We ensure we remove the id column, and we also ensure our data is shuffled.

![32](https://user-images.githubusercontent.com/100878908/190933004-a34c078e-fb54-466d-b017-73e8037cd965.png)

## Data Preprocessing

While Linear Regression is susceptible to the effects of outliers, and highly correlated input variables - Random Forests, just like Decision Trees, are not, so the required preprocessing here is lighter. We still however will put in place logic for:

* Missing values in the data
* Encoding categorical variables to numeric form

#### Note: 
The code and description for treating missing values and categorical variables are same as 
Linear regression as we are using the same dataset. So, please check out Linear regression part above if 
are interested in Data Preprocessing for this project.

## Hyperparmeters Tuning and Model Training

The code for hyperparameters tuning, instantiating and training the model is given below.
Based on tuning, we used maximum depth of 10 and 500 Decision Tress in this Random Forest Model.

![33](https://user-images.githubusercontent.com/100878908/190933176-63148864-f180-4251-ad1b-eaeb2f98a3d0.png)

## Model Performance Assessment
## Predict On The Test Set
To assess how well our model is predicting on new data - we use the trained model object (here called regressor) and ask it to predict the loyalty_score variable for the test set.


## Calculate R-square, Adjusted R-square and Cross-Validated R-square
![34](https://user-images.githubusercontent.com/100878908/190933270-258367ac-4338-41f7-b971-e0b68e93030c.png)

* Cross-validation score: 0.92
* R-square: 0.95
* Adjusted R-Square: 0.95

The Random Forest Regressor model has outperformed both Decision Tree and Linear Regression Model.

## Feature Importance
In our Linear Regression model, to understand the relationships between input variables and our ouput variable, loyalty score, we examined the coefficients. With our Decision Tree we looked at what the earlier splits were. These allowed us some insight into which input variables were having the most impact.

Random Forests are an ensemble model, made up of many, many Decision Trees, each of which is different due to the randomness of the data being provided, and the random selection of input variables available at each potential split point.

Because of this, we end up with a powerful and robust model, but because of the random or different nature of all these Decision trees - the model gives us a unique insight into how important each of our input variables are to the overall model.

As we’re using random samples of data, and input variables for each Decision Tree - there are many scenarios where certain input variables are being held back and this enables us a way to compare how accurate the models predictions are if that variable is or isn’t present.

So, at a high level, in a Random Forest we can measure importance by asking How much would accuracy decrease if a specific input variable was removed or randomised?

If this decrease in performance, or accuracy, is large, then we’d deem that input variable to be quite important, and if we see only a small decrease in accuracy, then we’d conclude that the variable is of less importance.

At a high level, there are two common ways to tackle this. The first, often just called Feature Importance is where we find all nodes in the Decision Trees of the forest where a particular input variable is used to split the data and assess what the Mean Squared Error (for a Regression problem) was before the split was made, and compare this to the Mean Squared Error after the split was made. We can take the average of these improvements across all Decision Trees in the Random Forest to get a score that tells us how much better we’re making the model by using that input variable.

If we do this for each of our input variables, we can compare these scores and understand which is adding the most value to the predictive power of the model!

The other approach, often called Permutation Importance cleverly uses some data that has gone unused at when random samples are selected for each Decision Tree (this stage is called “bootstrap sampling” or “bootstrapping”)

These observations that were not randomly selected for each Decision Tree are known as Out of Bag observations and these can be used for testing the accuracy of each particular Decision Tree.

For each Decision Tree, all of the Out of Bag observations are gathered and then passed through. Once all of these observations have been run through the Decision Tree, we obtain an accuracy score for these predictions, which in the case of a regression problem could be Mean Squared Error or r-squared.

In order to understand the importance, we randomise the values within one of the input variables - a process that essentially destroys any relationship that might exist between that input variable and the output variable - and run that updated data through the Decision Tree again, obtaining a second accuracy score. The difference between the original accuracy and the new accuracy gives us a view on how important that particular variable is for predicting the output.

Permutation Importance is often preferred over Feature Importance which can at times inflate the importance of numerical features. Both are useful, and in most cases will give fairly similar results.

Let’s put them both in place, and plot the results…

![35](https://user-images.githubusercontent.com/100878908/190933479-57348bff-0493-4784-aee0-b7bc39a3e04f.png)

![36](https://user-images.githubusercontent.com/100878908/190933481-44f1cb0b-11c2-4607-98b6-022bb4560d47.png)

The overall story from both approaches is very similar, in that by far, the most important or impactful input variable is distance_from_store which is the same insights we derived when assessing our Linear Regression & Decision Tree models.

There are slight differences in the order or “importance” for the remaining variables but overall they have provided similar findings.

# Modelling Summary
The most important outcome for this project was predictive accuracy, rather than explicitly understanding the drivers of prediction. Based upon this, we chose the model that performed the best when predicted on the test set - the Random Forest.


#### Metric 1: Adjusted R-Squared (Test Set)

* Random Forest = 0.95
* Decision Tree = 0.87
* Linear Regression = 0.78

#### Metric 2: R-Squared (K-Fold Cross Validation, k = 4)

* Random Forest = 0.92
* Decision Tree = 0.88
* Linear Regression = 0.85

Even though we were not specifically interested in the drivers of prediction, it was interesting to see across all three modelling approaches, that the input variable with the biggest impact on the prediction was distance_from_store rather than variables such as total sales. This is interesting information for the business, so discovering this as we went was worthwhile.

## Predicting Missing Loyalty Scores
We have selected the model to use (Random Forest) and now we need to make the loyalty_score predictions for those customers that the market research consultancy were unable to tag.

We cannot just pass the data for these customers into the model, as is - we need to ensure the data is in exactly the same format as what was used when training the model.

### Steps:

* Import the required packages for preprocessing
* Import the data for those customers who are missing a loyalty_score value
* Import our model object & any preprocessing artifacts
* Drop columns that were not used when training the model (customer_id)
* Drop rows with missing values
* Apply One Hot Encoding to the gender column (using transform)
* Make the predictions using .predict()

## Growth & Next Steps
While predictive accuracy was relatively high - other modelling approaches could be tested, especially those somewhat similar to Random Forest, for example XGBoost, LightGBM to see if even more accuracy could be gained.

We could even look to tune the hyperparameters of the Random Forest, notably regularisation parameters such as tree depth, as well as potentially training on a higher number of Decision Trees in the Random Forest.

From a data point of view, further variables could be collected, and further feature engineering could be undertaken to ensure that we have as much useful information available for predicting customer loyalty