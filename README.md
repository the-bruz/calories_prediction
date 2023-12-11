### Framing the Problem

**Our exploratory data analysis on this dataset can be found**: [here](https://the-bruz.github.io/Recipes-and-Ratings-Analysis/)  

In this project, the calories of recipes will be predicted based on the other nutrition of recipes. The model used is the linear regression model, and it will be a regression task.  

The response variable is the amount of calories of a recipe. It was chosen because in some cases, the calories are not labeled clearly for foods, and it will be helpful if people can predict it and establish a healthy diet plan.  

To examine the model, two metrics will be used: R-square and RMSE, which are the common choices for a regression task and can reflect the performance of the model appropriately. These two values are derived with the following formulas:  

$$R^2=1−\frac{∑(y_{i}−\hat{y}_{i})^2}{∑(y{i}−\overline{y})^2}$$  

$$RMSE = \sqrt{\frac{1}{n}\Sigma_{i=1}^{n}{\Big(\frac{y_i -\hat{y}_i}{\sigma_i}\Big)^2}}$$  

MAE is not considered since it is not a standardized metric and thus cannot reflect the performance of the model well.  

There is no clearly sequence relations between calories and other nutrition, so we can use the features (nutrition info) to predict the response variable (calories).  

### Baseline Model

In the baseline model, two features are used:  

* protein: quantitative variable  
* sugar: quantitative variable  

Since all features are quantitative, there is no encoding performed for the model.  

The baseline model is a linear regression model, with the two features above as the input and the calories as the output.  

The final scores the baseline model received are:  

| metric   | value              |
| -------- | ------------------ |
| r-square | 0.6891289257542752 |
| rmse     | 321.19566487234107 |

The r-square of the model is below 0.7, so the baseline model is considered bad and will be modified further to gain better results.

### Final Model

There are two more features added to the model:  

* total_fat: quantitative variable  
* carbohydrates: quantitative variable  

Again, since all features are quantitative, there is no encoding performed for the model. These two features are added since they are considered a great implication to the amount of calories.  

The final model is a pipeline consisted of three parts:  

* 'std': a sklearn `StandardScaler()` to standardize all features, so that the final model will be less complicated.  
* 'poly': a sklearn `PolynomialFeatures()` which allows the model to predict calories based on the powers of the features.  
* 'lin_reg': a sklearn `LinearRegression()`, the main predictor which is suitable to perform a regression task with quantitative features.  

The hyperparameter tuned in this model is the degree of `PolynomialFeatures()`. Sklearn `GridSearchCV` is used to fine-tune the hyperparameter, which searches the best degree between 1 and 4. By the result of `GridSearchCV`, the best hyperparameter for this model is `degree=1`.  

The final scores the baseline model received are:  

| metric   | value              |
| -------- | ------------------ |
| r-square | 0.9951338041841551 |
| rmse     | 40.18595638511328  |

The performance of the final model is hugely improved from the baseline model, with a r-square of 0.99 (almost optimal) and a rmse of around 40 from previously 321.  

### Fairness Analysis

The two groups chosen are recipes with high sugar (more than median) and the ones with low sugar (less than or equal to median). To evaluate the fairness of the final model, a permutation test is run with the following configurations:  

* Null hypothesis: The final model performs the same for recipes with high sugar and low sugar.  

* Alt hypothesis: The final model performs differently for recipes with high sugar and low sugar.  
* Evaluation metric: RMSE

* Test statistics: The absolute difference in the rmse's for two groups.  

* Significant Value: 5%  

The p-value for the permutation test with 1000 repetitions is 0.002 (0.2%), which is less than the significance level. Therefore, the null hypothesis is rejected, and it is more likely that the final model performs differently on recipes with high sugar and low sugar.
