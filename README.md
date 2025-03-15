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





Assessment of Missingness
Hypothesis Testing
Framing a Prediction Problem
Baseline Model
Final Model
Fairness Analysis