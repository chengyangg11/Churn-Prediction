# Predicting Churn for Google Merchandise Store
Google merchandise store is an eCommerce site that sells Google branded products such as T-shirts, mugs, and umbrellas. Although the site has experienced steady traffic, the majority of visitors are new to the site. Repeat visitors tend to be more devoted to the brand, more inclined to make purchases, and more likely to recommend the store to their acquaintances. This project involves constructing a model that forecasts which visitors are at risk of churning, allowing us to distribute coupons to encourage their return. 
<br>
Business objectives:
<li>Reduce churn rate from the current 82.6% to 50% in 12 months</li>
<li>Positive ROI on retention marketing campaign, given the cutomer life-time values</li>

# 1.Data
The data set comprises daily records spanning from July 2016 to July 2017, containing over 900K entries and 16 columns of data. The dataset includes information on traffic sources, such as organic, paid search, and display traffic, as well as data on user behavior on the site, including page URLs, content interactions, and more. Additionally, the dataset includes information on transactions that have taken place on the Google Merchandise Store website.<br><br>
[Google Cloud bucket](https://console.cloud.google.com/storage/browser?project=big-query-test-350401&prefix=)<br>
The data is available on Google Bigquery. I pulled the data from Bigquery and saved it in 100 slices (.parquet files) in Google Cloud bucket<br>
[Google Colab - Data Import to Google Drive](https://colab.research.google.com/drive/1r0nej4vJNAQIXLZBpit8qkLBnPtyZpXC#scrollTo=uO9mZs7nxl8K)<br>
I strung all the slices together into a single file in Google Drive, 10 at a time due to Googble Colab RAM limit 
# 2.Data Cleaning
[Google Colab - Data Cleaning](https://colab.research.google.com/drive/1YyAqxCwZUnUltECHXGx8W-G6fWLYvG9l)
<li>The dataset is nested as both struct and array types. This means that a single column can contain a group of related values, such as a list of items, or a set of nested fields representing a hierarchy of relationships. This is because BigQuery is designed to handle massive amounts of data. Nested data allows for more efficient data storage and processing, as it enables data to be organized in a logical and structured manner. Since no machine learning models can handled nested data, I created a function to flatten the dataframe. The resulting dataframe has 307 columns in total 
<li>Among 307 columns, 158 don't contain any info (sensitive customer data was redacted) and hence should be removed
<li>4 columns contain information that are used as samples to calculate other metrics, hence not important to our objective and can be removed 
<li>Because classification models cannot handle NaN, I decided to replace NaN with 0 for all numeric columns (transactions, visits, etc.) 
<li>Define churn: As I plot the percentages of visitors who donot come back in 7/15/30 day windows, I realize that they follow a very similar trend. However, since Google Merchandise store sells nonessential products, very few visitors would check back within 7 or 15 days. Therefore, I defined churn as the percentage of visitors not returning to the store in 30 day period. This definition can change during peak sales periods when we expect customers to visit more often.</li> 
  
<img width="690" alt="Picture 1" src="https://user-images.githubusercontent.com/70331438/224564632-2e6a8af4-6c82-4bde-8a5a-b23752d25536.png">
  
# 3.EDA & Feature Engineering
[Google Colab - EDA & Feature Engineering](https://colab.research.google.com/drive/115PsjYHYLRoCiQlLR7BgsTPEP5lKaCzo#scrollTo=YCY1k_QALRYf)
## 3.1. Initial observations
Observation 1: The data set is highly skewed, more than 80% of the samples are churned 
  
<img width="356" alt="churn rate" src="https://user-images.githubusercontent.com/70331438/224564675-88048f99-c5bc-4e3d-b637-8f4740dff161.png">
  
Observation 2: Display, referral, and paid have the highest not-churned to churned ratios (60%, 47.6%, 32%), whereas social has the lowest (5.9%) 
  
<img width="490" alt="churn rate by channel" src="https://user-images.githubusercontent.com/70331438/224564691-320a0b30-6310-4eee-bde3-0b1c829fd4b8.png">
  

## 3.2. Feature Engineering
Feature engineering is the process of selecting, transforming, and creating features from raw data that can be used to train machine learning models. This process can significantly improve the predictive power and performance of machine learning models. I created 27 new features from the original dataset. These features either give summaries of visitors' activities in the past 7/15/30 days or summaries of channels in the same time frames. For example: 
<li>Total visits by a customer in the past 7/15/30 days</li>
<li>Total number of page views by a customer in the past 7/15/30 days</li>
<li>Bounce rate by a customer in the past 7/15/30 days</li>
<li>Gap between the latest 2 visits by customer</li>
<li>Gap between first and last visits by customer</li>
<li>Average visits by channel in the past 15 days</li>
  
## 3.3. Feature Importance Analysis
I will evaluate feature importance for categorical and continuous variables separately. I define categorical variables as those fulfilling one of these conditions: 
<li>The data type is object OR </li>
<li>There are only 2-20 unique values (more than 20 are too granular, hence not meaningful to classify) OR </li>

For categorical variables, I calculated and compared their information value (IV) scores (the sum of WOE * difference between percentages of churned and not-churned in each group. <br>

![image](https://user-images.githubusercontent.com/70331438/224569822-929da139-4d71-4b79-a659-f1010d472acb.png) <br>

Figure: Categorical variables' IV scores<br>
![image](https://user-images.githubusercontent.com/70331438/224565131-b2905416-e477-45cb-a78d-17447dbce730.png)


Then, I selected the categorical variables with highest IV scores, combined them with continuous variables, and used random forest model to get the final feature importance scores<br>

Figure: All variables' importance scores<br>
![image](https://user-images.githubusercontent.com/70331438/224565045-50e1ce71-e39e-4d73-ae7e-0995f0df6430.png)
I finalized top 13 features for model building 
# 4.Modeling & Recommendations
[Modeling](https://colab.research.google.com/drive/1laYNlYowustBvl9DNnnd3T1YuBW_0-f6#scrollTo=nPv8OTO9eUyg)<br>
This is a binary classification problem, where the target variable can take one of two possible outcomes: churned or not churned.
There are several types of models that can be used for churn prediction, including logistic regression, decision trees, ensemble methods(random forests, GBM, etc.), and support vector machines (SVM). <br>
I fit and compared the overall performance of 3 classification models (logistics regression, random forest, and GBM) using ROC-AUC curves. 
The AUC can be interpreted as the probability that the model will correctly identify a churned customer as churned and a non-churned customer as non-churned. A higher AUC indicates that the model is better at distinguishing between churned and non-churned customers. 

GBM had the best performance with 0.81 AUC score. I then tuned the hyperparameters for GBM model, and used the best parameters to predict churn probability in the test set. 

![image](https://user-images.githubusercontent.com/70331438/224565187-ab230c86-b105-4c09-9f1f-b738d0eca5d1.png)<br>
Assuming life-time value of a customer is 100 dollars if he or she does not churn and we will be spending 10 dollars per customer in our retention marketing campaign. Not churned customers tend to be more engaged and have higher chance of adopting our promo. I tested 3 scenarios:
<li>10% promo adoption rate among both "not churned" and "churned" groups</li>
<li>10% adoption rate among the churned, 20% among the not churned group</li>
  <li>10% adoption rate among the churned group and 40% among the not churned group</li>
<br>I also assume that 20% of the high churn propensity customers who adopt our promo will end up returning to the site <br>

Scenario 2 and 3 are more realistic: Not churned customers are inherently interested in our products and more receptive to our marketing effort, hence we can expect them to be more likely to use the promo. 
<li>According to the simulation, profit peaks at 80th percentile for scenario 2 and 60% for scenario 3. This means in scenario 3, we can distribute coupons to 217k visitors scoring 0.6 and above to maximize profit return, if budget is not a constrainst.  
<li>In reality, marketing team is likely to have budget constrainst. Therefore, depending on the projected coupon redemption rate, we can decide how many people to target in the campaign.</li>   
Assuming we are leaning toward scenario 3 and the marketing budget is $1.5M, we'll then decide to target the 40th percentile of population (153k people with highest probability to churn)
  
# 5.Next Steps
<li>Train model for peak sales seasons with different churn definition (shorter timeframe)</li>
<li>Develop a web application that enables the marketing manager to input the total budget, budget per customer, and expected customer lifetime value. The application should then conduct a scenario simulation to estimate the return on investment (ROI) and provide the option to download a list of target customers</li>
