# Fair & Flavorful: Predicting Recipe Ratings of beef dishes with Machine Learning

by: Ganesh Kumarappan

## Introduction

Food is an essential part of daily life, and recipe ratings provide valuable insights into what people enjoy cooking and eating. This project explores how the presence of "beef" or beef-related words in a recipe’s name affects its overall rating.

The central question of this project is:
Does having the word "beef" or beef-related terms in a recipe’s name influence its average rating compared to other recipes?

Understanding how specific keywords in recipe names impact ratings can be useful for food bloggers, chefs, and recipe developers who want to optimize their content for audience engagement. If certain words consistently correlate with higher or lower ratings, it could inform how recipes are named and marketed on food-related platforms.

Through data analysis and machine learning, this project investigates whether recipes with beef-related terms receive significantly different ratings than other recipes, tests the statistical significance of this difference, and develops predictive models to estimate ratings based on recipe characteristics. Additionally, a fairness analysis ensures that our model does not introduce unintended biases against certain types of recipes.

This project provides valuable insights into the relationship between recipe names, ingredient perception, and user ratings, contributing to a better understanding of how people interact with and evaluate online recipes.


## Data Cleaning and Exploratory Data Analysis

### Data cleaning

First I merged the recipes and the interactions data frame on their relative IDs
Then I replaced every 0 with a null value because the scale is only from 1-5, so if someone gave a 0, it means that they just didn't fill it out, not necessarily that they thought it was that bad
Found the average of each recipe by grouping by the ID and then calculating the mean, then added it to another column in the merged dataframe

Converted tags, steps, ingredients to lists
Expanded the nutrition column to different columns per string

Created a boolean column to see if the word beef or beef-related key words exist in the recipe name

### Univariate Analysis

The distribution of average recipe ratings is left-skewed, with most recipes having a rating close to 5.0, indicating a generally positive 
reception. The visible clusters around integers (1, 2, 3, 4, and 5) suggest that many recipes may have received only a few ratings, 
keeping their mean rating at whole numbers.

The box plot shows that most recipes take under 60 minutes, but there are a few extreme outliers with significantly longer preparation times.
The presence of these outliers suggests that some recipes, likely elaborate ones, require much more effort than the average dish

### Bivariate Analysis

1st plot
The box plot shows that recipes containing "beef" in their name tend to have slightly lower ratings compared to recipes without "beef."
The lower whisker and first quartile for beef recipes suggest that these recipes might be more polarizing among users, with some lower 
ratings pulling down the median.

2nd plot
The scatter plot indicates that most recipes fall under 100 minutes of preparation time, with no clear correlation between cooking time 
and rating. However, there are a few longer-duration recipes (over 200 minutes) that tend to have high ratings, suggesting that some long, 
elaborate recipes may be highly appreciated by users.


### Interesting Aggregates

The pivot tables reveal how the presence of "chocolate" in a recipe's name, ingredients, or description affects its average rating. 
Recipes with "chocolate" in the name may indicate user preference for chocolate-themed dishes, while chocolate in the ingredients 
shows whether its actual inclusion impacts ratings. Lastly, mentioning chocolate in the description could suggest that appealing 
descriptions influence user perception, highlighting the role of marketing and expectations in recipe ratings.

## Assessment of Missingness

My initial merged dataset (the recipes dataset after step 1 of the data cleaning process) contained three columns with missing data: name, description, and rating. Since the chocolate_in_name, chocolate_in_ingredients, and chocolate_in_description columns were derived from these, they also contain missing values.

### NMAR Analysis
I believe that the missing data in the rating column is likely Not Missing At Random (NMAR). Users who feel indifferent about a recipe may choose not to leave a review, while those who either love or strongly dislike a recipe may be more motivated to submit a rating. Since the decision to rate is influenced by the user's experience, rather than randomness, it is reasonable to suspect that rating is not missing at random.

## Missingness Dependency
To explore further relationships in the data, I analyzed whether the missingness of the rating column depends on two other variables:

n_ingredients (the number of ingredients in a recipe)
calories (the calorie content of the recipe)
Missingness of Rating Based on n_ingredients
Null Hypothesis (H₀): The missingness of ratings does not depend on the number of ingredients in a recipe.
Alternative Hypothesis (H₁): The missingness of ratings does depend on the number of ingredients in a recipe.
Statistic: The absolute difference between the mean n_ingredients of recipes with missing ratings and recipes with non-missing ratings.
Significance Level: α = 0.05
The observed difference in the number of ingredients between recipes with and without missing ratings is 0.25. The p-value is 0.002, which is less than 0.05, meaning that this result is statistically significant. Since none of the permuted differences were as extreme as the observed difference, we reject the null hypothesis and conclude that the missingness of ratings is dependent on the number of ingredients.

Missingness of Rating Based on Calories
Null Hypothesis (H₀): The missingness of ratings does not depend on the calorie content of the recipe.
Alternative Hypothesis (H₁): The missingness of ratings does depend on the calorie content of the recipe.
Statistic: The absolute difference between the mean calorie count of recipes with missing ratings and recipes with non-missing ratings.
Significance Level: α = 0.05
The observed difference in calories between recipes with and without missing ratings is 87.86. However, the p-value is 1.0 (rounded from 0.00e+00), which is greater than 0.05. This means that the difference could have happened by random chance, and we fail to reject the null hypothesis. Therefore, our results suggest that the missingness of ratings is not dependent on calorie content.

Conclusion
The missingness of ratings is dependent on the number of ingredients, meaning that recipes with fewer or more ingredients may be more likely to have missing ratings.
However, the missingness of ratings is not dependent on calorie content, indicating that caloric value does not influence whether a recipe receives a rating.
This analysis helps identify whether missing ratings are structured or random, which is useful for further modeling and imputation strategies

# Hypothesis Testing

Hypothesis Testing
Null Hypothesis: The mean rating of recipes with "beef" or beef-related words in the recipe name is equal to the mean rating of all recipes.

I chose this null hypothesis because the presence of beef-related terms in the name directly indicates that the recipe primarily features beef, making it a reasonable way to categorize beef-based recipes.

Alternate Hypothesis: The mean rating of recipes with "beef" or beef-related words in the recipe name is not equal to the mean rating of all recipes.

I chose this alternate hypothesis because we are interested in whether beef-based recipes deviate from the overall population in terms of ratings, without assuming a specific direction.

Statistic: Absolute difference between the mean rating of recipes with beef-related terms in the name and the mean rating of all recipes.

Because this is a two-sided test, I chose to use the absolute difference in means to capture any deviation.

Significance Level: α = 0.05

I chose 0.05 as the significance level because a Type-1 error (rejecting the null hypothesis when it is actually true) is not particularly harmful in this context.

To conduct this hypothesis test, I first calculated the observed difference in mean ratings between beef-based recipes and all recipes. I then created 1000 permuted samples by randomly shuffling the ratings and computing the absolute difference between the sample mean and the overall mean. The resulting permutation distribution was visualized in a histogram.

Following these steps, 0 out of 1000 observations were as extreme as the observed test statistic.

Thus, the p-value is 0.0. We reject the null hypothesis. This test suggests that the mean rating of recipes with "beef" or beef-related terms in their name is significantly different from the mean rating of all recipes.


# Framing a Prediction Problem

Prediction Problem: I will attempt to predict the average rating of a recipe using regression. Ratings serve as a quantifiable measure of how well-received a recipe is, making it a valuable prediction target. Unlike my previous analysis on specific recipe categories like beef-related recipes, this model aims to generalize predictions across all recipes.

Type of Prediction Model:
This is a regression problem, as the response variable (rating) is continuous rather than categorical.

Response Variable (Target):
The response variable is rating, as it represents the overall quality and reception of a recipe.

Choice of Evaluation Metric:
I will use Mean Squared Error (MSE) as the primary evaluation metric because:

MSE penalizes large deviations more than small ones, making it ideal for capturing prediction errors in a continuous target like ratings.
Since ratings are skewed toward higher values (closer to 5), I will also consider R-squared (R²) to assess the proportion of variance explained by the model.

# Baseline Model

Baseline Model
For my baseline model, I utilized a linear regression model with default hyperparameters.

First, I removed any rows containing missing values in the selected columns to ensure the model trained on complete data. I then split the data into a training set (80%) and a test set (20%).

For the baseline model, I utilized two features:

n_steps (quantitative) - Represents the number of steps in a recipe.
n_ingredients (quantitative) - Represents the number of ingredients in a recipe.
Since both features are already numerical, I did not need to apply any categorical encoding. However, I applied feature scaling using StandardScaler to normalize the numerical values and prevent potential issues caused by differences in scale.

The mean squared error (MSE) for the test data is 0.4045, and the R² (coefficient of determination) score is -0.0004.
An R² value close to zero (or negative) suggests that the model performs poorly—essentially, it fails to explain the variance in ratings better than a simple mean predictor.

Overall, this result suggests that n_steps and n_ingredients alone are not strong predictors of recipe ratings. To improve model performance, I will consider incorporating additional features such as categorical variables (e.g., recipe tags, cuisine type) and experiment with more complex models in the next iteration.

# Final Model

You said:

also here are the results from the code of the 2 final models 
ChatGPT said:
Here’s the revised final model description incorporating your results:

Final Model: Feature Selection and Justification
In my final model, I utilized the following features:

contains_beef_name
This feature indicates whether the word "beef" appears in the name of a recipe. Based on my previous analysis, I observed slight differences in ratings between recipes with and without "beef" in the name. This could reflect consumer preferences for beef-based recipes, making it a useful feature. I one-hot encoded this column to differentiate between these categories.

contains_beef_ingredients
This feature identifies if "beef" or beef-related products are listed in the recipe's ingredients. Since ingredient lists provide a stronger indication of a dish’s actual composition than just the name, this feature may capture variations in ratings influenced by beef-based ingredients. I applied one-hot encoding to this feature.

sugar
This feature represents the percentage daily value of sugar in a recipe. During my analysis, I noticed that recipes with extreme amounts of sugar may have different ratings. To mitigate the effects of extreme outliers, I applied a QuantileTransformer, which helps normalize the distribution.

protein
Protein content is another nutritional aspect that may influence ratings, as some users may favor high-protein recipes for health reasons. Similar to the sugar column, I used a QuantileTransformer to standardize the values and handle outliers.

n_steps
This feature represents the number of steps in a recipe. Recipes with more steps could be perceived as too complex or, alternatively, as more refined and detailed, which could influence ratings. I normalized this feature using a QuantileTransformer.

n_ingredients
The number of ingredients in a recipe is an essential measure of complexity. Simpler recipes may be preferred due to convenience, while highly ingredient-heavy recipes may be perceived as more flavorful or gourmet. I also applied a QuantileTransformer to normalize the distribution of this feature.

Model Selection and Hyperparameter Tuning
For the modeling algorithm, I explored two options:

Final Model 1: Linear Regression
For this model, I retained the linear regression approach from my baseline model but incorporated additional transformed features. Linear regression is useful for understanding how each feature contributes to the predicted rating, but its predictive power is limited due to the complexity of interactions between different features.

Results for Final Model 1 (Linear Regression):
Mean Squared Error (MSE): 0.4040
R² Score: 0.0007
While the performance was similar to the baseline model, the additional features did not significantly improve predictive power, suggesting a more complex model might be needed.

Final Model 2: Random Forest Regressor with Hyperparameter Tuning
To address the limitations of linear regression, I implemented a Random Forest Regressor, which is better suited for capturing non-linear interactions. I tuned the following hyperparameters using GridSearchCV:

max_depth: The depth of trees in the random forest (values tested: 5, 10)
min_samples_split: The minimum number of samples required to split an internal node (10, 20)
n_estimators: The number of trees in the forest (100, 200)
After tuning, the best combination of hyperparameters was:

max_depth = 5

min_samples_split = 10

n_estimators = 200

Results for Final Model 2 (Random Forest Regressor):

Mean Squared Error (MSE): 0.4037
R² Score: 0.0015

# Fairness Analysis

For my fairness analysis, I investigated model parity between two groups: high sodium recipes and low sodium recipes. To categorize recipes into these groups, I compared their sodium PDV (percent daily value) to the median sodium PDV. If a recipe’s sodium PDV was greater than or equal to the median, I classified it as a high sodium recipe. Otherwise, I classified it as a low sodium recipe.

I chose Mean Squared Error (MSE) as my evaluation metric because large deviations between actual ratings and predicted ratings are significantly worse than minor deviations. If the model consistently performs worse for one group, it could suggest bias in the model’s predictions.

Hypotheses:
Null Hypothesis: The MSE of the model across recipes with high sodium and low sodium is roughly the same. The model achieves MSE parity across these two groups.
Alternative Hypothesis: The MSE of the model across recipes with high sodium and low sodium is not the same. The model does not achieve MSE parity across these two groups.
Test Statistic:
Absolute difference between the MSE of our final model for recipes with high sodium and recipes with low sodium.
Significance Level:
α = 0.05
I chose 0.05 as the significance level because a Type-1 error (rejecting the null hypothesis when it is actually true) is not particularly harmful in our case.
Permutation Test for Fairness Analysis
To conduct this fairness analysis, I:

Computed the MSE for both high sodium and low sodium recipes separately.
Randomly shuffled the sodium labels (high vs. low sodium) to break any potential relationship between sodium level and prediction error.
Repeated this permutation process 1000 times to create a distribution of differences in MSE under the null hypothesis.
Compared the observed difference in MSE to the permuted differences to compute a p-value.


Results and Conclusion
My fairness analysis yielded a p-value of 0.008. Thus, we reject the null hypothesis. This result suggests that the MSE of our model across recipes with high sodium and low sodium is not the same.

This indicates that our final model may be unfair across these two groups, as it predicts ratings less accurately for one group compared to the other. This bias could be due to an underlying relationship between sodium content and recipe ratings that the model has not captured effectively.

Next Steps
To mitigate this bias, further improvements could be made, such as:

Introducing more features related to sodium intake (e.g., total fat, sugar) to better understand how nutritional values affect ratings.
Testing different models (e.g., Gradient Boosting, XGBoost) to see if other regressors provide a more balanced performance across groups.
Investigating interactions between sodium and other variables to determine whether certain subgroups (e.g., high sodium & high protein) experience more bias than others.
By addressing these issues, we can work toward building a more equitable prediction model.


