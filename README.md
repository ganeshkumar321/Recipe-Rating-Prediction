# Fair & Flavorful: Predicting Recipe Ratings of beef dishes with Machine Learning

by: Ganesh Kumarappan

## Introduction

Food is an essential part of life, and analyzing recipe ratings provides valuable insight to what people are eating and enjoying these days. This project explores how the having the word "beef" or beef-related words in a recipe’s name affects its overall rating.

The central question of this project is:
Does having the word "beef" or beef-related terms in a recipe’s name influence its average rating compared to other recipes?

Understanding how specific keywords in recipe names impact ratings can be useful for food bloggers, chefs, and recipe developers who want to optimize their content for audiences. If certain words consistently correlate with higher or lower ratings, it could affect how recipes are named and marketed.

Through data analysis and machine learning, this project investigates whether recipes with beef-related terms receive significantly different ratings than other recipes. Additionally, it also tests the statistical significance of this difference, and develops predictive models to estimate ratings based on recipe characteristics. Additionally, a fairness analysis ensures that our model does not introduce  biases against certain types of recipes.

This project provides valuable insights into the relationship between recipe names, ingredient perception, and user ratings, contributing to a better understanding of how people interact with and evaluate online recipes.

The number of rows in the recipes dataset is 83782, with the columns names shown below:

| Column          | Description                                                                                           |
|------------------|-------------------------------------------------------------------------------------------------------|
| `name`          | Recipe name                                                                                          |
| `id`            | Recipe ID                                                                                            |
| `minutes`       | Minutes to prepare recipe                                                                            |
| `contributor_id`| User ID who submitted this recipe                                                                     |
| `submitted`     | Date the recipe was submitted                                                                         |
| `tags`          | Food.com tags for the recipe                                                                          |
| `nutrition`     | Nutrition information in the form `[calories (#), total fat (PDV), sugar (PDV), sodium (PDV), protein (PDV), saturated fat (PDV), carbohydrates (PDV)]`; PDV stands for “percentage of daily value” |
| `n_steps`       | Number of steps in the recipe                                                                         |
| `steps`         | Text for recipe steps, in order                                                                       |
| `description`   | User-provided description                                                                             |
| `ingredients`   | Text for recipe ingredients                                                                           |
| `n_ingredients` | Number of ingredients in the recipe                                                                   |

The number of rows in the interactions dataset is 731927 with the column names shown below:

| Column     | Description          |
|------------|----------------------|
| `user_id`  | User ID              |
| `recipe_id`| Recipe ID            |
| `date`     | Interaction Date     |
| `rating`   | Rating               |
| `review`   | Review text          |

The columns most relevant to my question are "name", "description", "description", and "ingredients" in recipes, along with "rating" in interactions.

## Data Cleaning and Exploratory Data Analysis


### Data cleaning

First I merged the recipes and the interactions data frame on their relative IDs. Then I replaced every 0 with a null value because the scale is only from 1-5, so if someone gave a 0, it means that they just didn't fill it out, not necessarily that they thought it was that bad. Then I found the average of each recipe by grouping by the ID and then calculating the mean, then added it to another column in the recipes_df dataframe. In addition to this, I changed the types of some of the columns such as tags, steps, and ingredients from strings to lists because it was easier to digest. The nutrition column contained many different information such as calories (#), total fat (PDV), sugar (PDV), sodium (PDV), protein (PDV), saturated fat (PDV), and carbohydrates, so I expanded the nutrition column into many different columns. Then I created 3 boolean columns if the word 'beef' or beef related words such as "steak", "hamburger", "meatloaf", "sloppy joe", "brisket" existed in the recipe_name, ingredients, and the description.

My cleaned dataset contained 83,782 rows and 22 columns

Here is a preview of the first 5 rows, but some of the columns have been removed because there are too many to fit.

| name                                | id      | minutes | contributor_id | sodium | protein | saturated_fat | carbohydrates |
|-------------------------------------|---------|---------|----------------|--------|---------|---------------|---------------|
| 1 brownies in the world best ever  | 333281  |  40      | 985201         | 3.0    | 3.0     | 19.0          | 6.0           |
| 1 in canada chocolate chip cookies | 453467  |  45      | 1848091        | 22.0   | 13.0    | 51.0          | 26.0          |
| 412 broccoli casserole             | 306168  |  40      | 50969          | 32.0   | 22.0    | 36.0          | 3.0           |
| millionaire pound cake             | 286009  |  120     | 461724         | 13.0   | 20.0    | 123.0         | 39.0          |
| 2000 meatloaf                      | 475785  |  90      | 2202916        | 12.0   | 29.0    | 48.0          | 2.0           |


### Univariate Analysis

<iframe
  src="assets/recipe_ratings_histogram.html"
  width="800"
  height="600"
  frameborder="0"
></iframe>

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
For my baseline model, I implemented a linear regression model with default hyperparameters. To ensure the model was trained on complete data, I first removed any rows with missing values in the selected columns. Then, I split the dataset into an 80% training set and a 20% test set.

For the baseline model, I utilized two features:
n_steps (quantitative) - Represents the number of steps in a recipe.
n_ingredients (quantitative) - Represents the number of ingredients in a recipe.
Since both features are already numerical, I did not need to apply any categorical encoding. However, I applied feature scaling using StandardScaler to normalize the numerical values and prevent potential issues caused by differences in scale.

The mean squared error (MSE) for the test data is 0.4045, and the R² (coefficient of determination) score is -0.0004.
An R² value near zero (or negative) indicates that the model performs poorly, offering little to no improvement over simply predicting the average rating for all recipes.

Overall, this result indicates that n_steps and n_ingredients alone are weak predictors of recipe ratings. To enhance model performance, I plan to incorporate additional features, such as categorical variables and explore more complex models in the next iteration.

# Final Model

In my final model, I utilized the following features:

contains_beef_name
This feature indicates whether the word "beef" appears in the recipe’s name. My previous analysis showed slight differences in ratings between recipes with and without "beef" in the name, potentially reflecting consumer preferences for beef-based dishes. To distinguish between these categories, I applied one-hot encoding.

contains_beef_ingredients
This feature identifies whether "beef" or beef-related products appear in the ingredient list. Since ingredient lists provide a more accurate representation of a dish’s actual composition than just the name, this feature may better capture rating variations influenced by beef-based ingredients. I also used one-hot encoding for this feature.

sugar
This feature represents the percentage daily value of sugar in a recipe. My analysis suggested that recipes with extremely high or low sugar content might receive different ratings. To mitigate the impact of outliers and normalize the distribution, I applied a QuantileTransformer.

protein
Protein content is another nutritional factor that may influence ratings, as some users might favor high-protein recipes for health reasons. Similar to the sugar feature, I used a QuantileTransformer to standardize values and handle potential outliers.

n_steps
This feature captures the number of steps in a recipe. Recipes with more steps might be perceived as too complex (discouraging users) or more refined and detailed (appealing to certain users), both of which could influence ratings. I applied a QuantileTransformer to normalize this feature.

n_ingredients
The number of ingredients in a recipe serves as a measure of complexity. Simpler recipes may be preferred for their convenience, while ingredient-heavy recipes may be perceived as more flavorful or gourmet. To ensure a balanced distribution, I also normalized this feature using a QuantileTransformer.

Model Selection and Hyperparameter Tuning
For the modeling algorithm, I explored two options:

Final Model 1: Linear Regression
For this model, I continued using the linear regression approach from my baseline model while incorporating additional transformed features. Linear regression is valuable for interpreting how each feature influences the predicted rating. However, its predictive power remains limited due to the complex interactions between different features, which a linear model may not fully capture.

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

For my fairness analysis, I evaluated whether my model performed equally well across two groups: high-sodium recipes and low-sodium recipes.

To categorize recipes, I compared their sodium PDV (percent daily value) to the median sodium PDV. Recipes with a sodium PDV greater than or equal to the median were classified as high-sodium, while those below the median were classified as low-sodium.

I used Mean Squared Error (MSE) as the evaluation metric since large discrepancies between actual and predicted ratings are more concerning than minor deviations. If the model systematically performs worse for one group, this could indicate bias in its predictions.

Hypotheses
Null Hypothesis: The MSE of the model across recipes with high sodium and low sodium is roughly the same. The model achieves MSE parity across these two groups.
Alternative Hypothesis: The MSE of the model across recipes with high sodium and low sodium is not the same. The model does not achieve MSE parity across these two groups.
Test Statistic
I used the absolute difference between the MSE of the final model for high sodium recipes and low sodium recipes as my test statistic.

Significance Level
I chose α = 0.05 as the significance level because a Type-1 error (rejecting the null hypothesis when it is actually true) is not particularly harmful in this case.

Permutation Test for Fairness Analysis
To conduct this fairness analysis, I followed these steps:

Computed the MSE for both high sodium and low sodium recipes separately.
Randomly shuffled the sodium labels (high vs. low sodium) to break any potential relationship between sodium level and prediction error.
Repeated this permutation process 1000 times to create a distribution of MSE differences under the null hypothesis.
Compared the observed difference in MSE to the permuted differences to compute a p-value.
Results and Conclusion
The fairness analysis yielded:

Observed MSE Difference: 0.0236
P-Value: 0.2760
Since the p-value is greater than 0.05, we fail to reject the null hypothesis. This suggests that our model does not exhibit significant bias in predicting ratings for high sodium vs. low sodium recipes.

Although a difference in MSE exists, it is not statistically significant at the α = 0.05 level, meaning any observed disparity in prediction error could be due to random chance rather than unfairness in the model.


