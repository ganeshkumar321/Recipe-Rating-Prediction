# Fair & Flavorful: Predicting Recipe Ratings of beef dishes with Machine Learning

by: Ganesh Kumarappan

## Introduction

Food is an essential part of life, and analyzing recipe ratings provides valuable insight to what people are eating and enjoying these days. This project explores how the having the word "beef" or beef-related words in a recipe’s name affects its overall rating.

The central question of this project is:
Does having the word "beef" or beef-related terms in a recipe’s name influence its average rating compared to other recipes?

Understanding how specific keywords in recipe names impact ratings can be useful for food bloggers, chefs, and recipe developers who want to optimize their content for audiences. If certain words consistently correlate with higher or lower ratings, it could affect how recipes are named and marketed.

Through data analysis and machine learning, this project investigates whether recipes with beef-related terms receive significantly different ratings than other recipes. Additionally, it also tests the statistical significance of this difference, and develops predictive models to estimate ratings based on recipe characteristics. Additionally, a fairness analysis ensures that our model does not introduce  biases against certain types of recipes.

This project provides valuable insights into the relationship between recipe names, ingredient perception, and user ratings, contributing to a better understanding of how people interact with and evaluate online recipes.


## Data Cleaning and Exploratory Data Analysis

### Data cleaning

First I merged the recipes and the interactions data frame on their relative IDs. Then I replaced every 0 with a null value because the scale is only from 1-5, so if someone gave a 0, it means that they just didn't fill it out, not necessarily that they thought it was that bad. Then I found the average of each recipe by grouping by the ID and then calculating the mean, then added it to another column in the recipes_df dataframe. In addition to this, I changed the types of some of the columns such as tags, steps, and ingredients from strings to lists because it was easier to digest. The nutrition column contained many different information such as calories (#), total fat (PDV), sugar (PDV), sodium (PDV), protein (PDV), saturated fat (PDV), and carbohydrates, so I expanded the nutrition column into many different columns. Then I created 3 boolean columns if the word 'beef' or beef related words such as "steak", "hamburger", "meatloaf", "sloppy joe", "brisket" existed in the recipe_name, ingredients, and the description.

### Univariate Analysis

The distribution of average recipe ratings is skewed towards the upper end with most recipes having a rating close to 5, which. As we can see, the clusters are more visible at whole integers, which may suggest that recipes only got a few ratings, thus keeping their averages at or near whole numbers. 


### Bivariate Analysis

The box plot shows that recipes containing "beef" in their name tend to have  lower ratings compared to recipes without "beef."
The lower whisker and first quartile for beef recipes suggest that these recipes might be more polarizing, with some lower 
ratings pulling down the median.


### Interesting Aggregates

The pivot tables show how having "chocolate" in a recipe's name, ingredients, or description affects its average rating. 
Recipes with "chocolate" in the name can indicate user preference for chocolate-themed dishes, while chocolate in the ingredients 
shows whether its actual inclusion impacts ratings. Lastly, mentioning chocolate in the description could suggest that having appealing descriptions can influence users' decisions.

## Assessment of Missingness

### NMAR Analysis
I think that the missing data in the rating column is most likely Not Missing At Random (NMAR). Users who don't have any strong feelings about a recipe may choose not to leave a review, while those who have strong views of love or strongly dislike a recipe may be more likely to submit a rating. Since the decision to rate is influenced by the user's experience, rather than randomness and other columns, it is reasonable conclude that the data is not missing at random.

## Missingness Dependency
To explore relationships in the data, I analyzed whether the missingness of the rating column depends on two other variables: n_ingredients (the number of ingredients in a recipe) and calories (the calorie content of the recipe)

Missingness of Rating Based on n_ingredients:
Null Hypothesis (H₀): The missingness of ratings does not depend on the number of ingredients in a recipe.
Alternative Hypothesis (H₁): The missingness of ratings does depend on the number of ingredients in a recipe.
Statistic: The absolute difference between the mean n_ingredients of recipes with missing ratings and recipes with non-missing ratings.
Significance Level: α = 0.05
The observed difference in the number of ingredients between recipes with and without missing ratings is 0.25. The p-value is 0.002, which is less than 0.05, meaning that this result is statistically significant. Since none of the permuted differences were as extreme as the observed difference, we reject the null hypothesis and conclude that the missingness of ratings is dependent on the number of ingredients.

Missingness of Rating Based on Calories:
Null Hypothesis (H₀): The missingness of ratings does not depend on the calorie content of the recipe.
Alternative Hypothesis (H₁): The missingness of ratings does depend on the calorie content of the recipe.
Statistic: The absolute difference between the mean calorie count of recipes with missing ratings and recipes with non-missing ratings.
Significance Level: α = 0.05
The observed difference in calories between recipes with and without missing ratings is 87.86. However, the p-value is 1.0 (rounded from 0.00e+00), which is greater than 0.05. This means that the difference could have happened by random chance, and we fail to reject the null hypothesis. Therefore, our results suggest that the missingness of ratings is not dependent on calorie content.

Conclusion
The likelihood of missing ratings depends on the number of ingredients, suggesting that recipes with fewer ingredients may be more prone to missing ratings, possibly because simpler meals require less effort and engagement. However, missing ratings do not appear to be influenced by the calories, indicating that calorie count does not impact whether a recipe receives a rating. This analysis helps determine whether missing ratings follow a pattern or occur randomly, providing valuable insights for modeling and imputation strategies.

# Hypothesis Testing

Hypothesis Testing
Null Hypothesis: The mean rating of recipes with "beef" or beef-related words in the recipe name is equal to the mean rating of all recipes.

I chose this null hypothesis because the presence of beef-related terms in the name directly indicates that the recipe primarily features beef, making it a reasonable way to categorize beef-based recipes.

Alternate Hypothesis: The mean rating of recipes with "beef" or beef-related words in the recipe name is not equal to the mean rating of all recipes.

I chose this alternate hypothesis because we are interested in whether beef-based recipes deviate from the overall population in terms of ratings, without assuming a specific direction.

Statistic: Absolute difference between the mean rating of recipes with beef-related terms in the name and the mean rating of all recipes.

Significance Level: α = 0.05

To test this hypothesis, I first calculated the observed difference in mean ratings between beef-based recipes and all recipes. Next, I generated 1,000 permuted samples by randomly shuffling the ratings and computing the absolute difference between the sample mean and the overall mean. I then visualized the resulting permutation distribution using a histogram.

After conducting the test, none of the 1,000 permuted samples had a test statistic as extreme as the observed value.

As a result, the p-value is 0.0, leading us to reject the null hypothesis. This suggests that recipes with "beef" or beef-related terms in their name have a significantly different mean rating compared to all recipes.


# Framing a Prediction Problem

I will build a regression model to predict the average rating of a recipe. Since ratings reflect how well a recipe is received, they serve as a meaningful prediction target. Unlike my previous analysis focused on specific recipe categories, such as beef-based recipes, this model aims to make predictions across all recipes.

Type of Model:
This is a regression problem, as the target variable (rating) is continuous rather than categorical.

Target Variable:
The response variable is the recipe rating, which represents its overall quality and reception.

Evaluation Metric:
I will use Mean Squared Error (MSE) as the primary evaluation metric because: MSE penalizes larger errors more heavily, making it effective for capturing prediction inaccuracies in a continuous variable like ratings. Since ratings are skewed toward higher values (closer to 5), I will also consider R-squared (R²) to measure how well the model explains the variance in ratings.

# Baseline Model

Baseline Model
For my baseline model, I utilized a linear regression model with default hyperparameters.

First, I removed any rows containing missing values in the selected columns to ensure the model trained on complete data. I then split the data into a training set (80%) and a test set (20%).

For the baseline model, I utilized two features:

n_steps (quantitative) - Represents the number of steps in a recipe.
n_ingredients (quantitative) - Represents the number of ingredients in a recipe.
Since both features are already numerical, I did not need to apply any categorical encoding. However, I applied feature scaling using StandardScaler to normalize the numerical values and prevent potential issues caused by differences in scale.

The mean squared error (MSE) for the test data is 0.4045, and the R² (coefficient of determination) score is -0.0004.
An R² value close to zero (or negative) suggests that the model performs poorly and it fails to explain the variance in ratings better than a simple mean predictor.

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


