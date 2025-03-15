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





Assessment of Missingness
Hypothesis Testing
Framing a Prediction Problem
Baseline Model
Final Model
Fairness Analysis