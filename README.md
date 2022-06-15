## Multiple Linear Regression using Gradient Descent on Seoul Bike Share Demand Data Set
### Data set information
This repository showcases multiple linear regression on the data set to predict the rented bike count. Gradient descent algorithm with the batch update is implemented to study the effects of the hyper parameters such as learning rate and threshold on the residuals, R - squared value, and nature of convergence on the test and training sets.

Gradient descent is a first-order optimization algorithm used to find local minima or maxima. In multiple linear regression,a cost function (J) is defined over the predicted values (yᶺ) and the actual values (y) from a data set as follows<br><br>
J(β0, β1) = (1/2m) x ∑(yᶺ(i) – y(i))2 <br><br>
Where β0 and β1 are the coefficients that are updated in an iteration to converge to a local minimum of the cost function. For multiple linear regression, multiple coefficients can be updated in one iteration and this is called batch update.

### Data set information
Currently, Rental bikes are introduced in many urban cities for the enhancement of mobility comfort. It is important to make the rental bike available and accessible to the public at the right time as it lessens the waiting time. Eventually, providing the city with a stable supply of rental bikes becomes a major concern. The crucial part is the prediction of the bike count required at each hour for the stable supply of rental bikes. The dataset contains weather information (Temperature, Humidity, Windspeed, Visibility, Dewpoint, Solar radiation, Snowfall, Rainfall), the number of bikes rented per hour, and date information.

### Implementation
The multi-linear regression experiments are implemented in Python and executed in a jupyter notebook. The Seoul bike share data is stored in a CSV (comma-separated value) file which can be read by Pandas, which is a fast, powerful, flexible, and easy-to-use open-source data analysis and manipulation tool, built on top of the Numpy, which is another python package for handling numerical matrix computations. Matplotlib and seaborn are python plotting libraries and provided an object-oriented API for generating plots. The following steps are executed for this study

### Exploratory Data Analysis

The dataset consists of 8760 entries and 14 columns of which the Rented Bike count is the target (dependent variable) which needs to be predicted given features such as an hour of the day, Humidity, and temperature, etc. The rented bike count data is highly skewed and might need a log transformation for better results. The dataset
has both numerical and categorical variables. The table below describes the summary of the range of values in the continuous variables.

<img src="https://github.com/karthikramx/Multiple-Linear-Regression-using-Gradient-Descent-on-Seoul-Bike-Share-Demand-Data-Set/blob/main/Plots/target_distribution.png" alt="drawing" style="width:500px;"/>

A pair plot against the target variable Rented Bike Count is plotted to check for any relationships and patterns that can help us choose or drop features. There seems to be a high correlation between temperature and dew point temperature

<img src="https://github.com/karthikramx/Multiple-Linear-Regression-using-Gradient-Descent-on-Seoul-Bike-Share-Demand-Data-Set/blob/main/Plots/y%20-%20pair%20plot.png" alt="drawing" style="width:1500px;"/>

### Preparing data - mapping, normalizing, and splitting

<img src="https://github.com/karthikramx/Multiple-Linear-Regression-using-Gradient-Descent-on-Seoul-Bike-Share-Demand-Data-Set/blob/main/Plots/log(target_distribution).png" alt="drawing" style="width:500px;"/>

First, the target variable is log-transformed. The distribution on the left side shows that the data is
somewhat better balanced as opposed to what previously existed.
Train_test_split function from the sklearn.model_selection module is used to split the data set randomly into train and test sets. It is advisable to normalize the data set before performing linear regression. Standardscaler function from sklearn is used to standardize the data. The data is transformed and fit on the train set and only transformed on the test set using the scaler object. This is done as in a real-life scenario, we do not have any idea of the newly generated data and hence we should not assume any other distribution parameters than that of the test set.


A heat map can be used to check the degree of correlation between all the features of a dataset. The helps not only helps to observe highly correlated variables with the target variable, but also checks for high correlations within the features. This allows for dropping variables and avoids the problem of multicollinearity. The weekday features show an insignificant correlation with the target variable and can be dropped during regression.

<img src="https://github.com/karthikramx/Multiple-Linear-Regression-using-Gradient-Descent-on-Seoul-Bike-Share-Demand-Data-Set/blob/main/Plots/heatmap_corr.png" alt="drawing" style="width:1500px;"/>

### Multiple Linear Regression - Experiments and results

#### Experiment 1
Various parameters for linear regression (e.g. learning rate ∝) are varied to check the error values in
test and train sets change. The R-squared metric checks how much of the variance in the data is explained by the model. Under this experiment, R-squared for test and train set is observed for varying the number of iterations and learning rate in gradient descent.

<p float="left" align="center">
<img src="https://github.com/karthikramx/Multiple-Linear-Regression-using-Gradient-Descent-on-Seoul-Bike-Share-Demand-Data-Set/blob/main/Plots/Iterations_vs_R2.png" alt="drawing"     style="width:400px;"/>

<img src="https://github.com/karthikramx/Multiple-Linear-Regression-using-Gradient-Descent-on-Seoul-Bike-Share-Demand-Data-Set/blob/main/Plots/Learning_rate_vs_R2.png" alt="drawing" style="width:400px;"/>
</p>

It is to be noted that the hyperparameters (iterations and learning rate) in the above graphs have been log scaled to observe the effect of changing them. The solution seems to converge when iterations are more than 1000 and when the learning rate is fixed to 0.001. For the second sub-experiment, the experiment was performed at 100 iterations for varying learning rates. The solution is converging when the learning rate is less than 0.01. Both iterations and learning rates play an important role to attain an optimal solution.

#### Experiment 2

<img src="https://github.com/karthikramx/Multiple-Linear-Regression-using-Gradient-Descent-on-Seoul-Bike-Share-Demand-Data-Set/blob/main/Plots/threshold_vs_iterations_R2.png" alt="drawing"/>

Various thresholds were tested for convergence for linear regression. The plot shows the R-Squared results for train and test sets as a function of the threshold. The secondary y-axis shows the number of iterations required to converge at different thresholds. The solution converges when the threshold is less than 0.001. It is also observed that the iterations required to converge increase rapidly when the threshold is less than 0.0001.

#### Experiment 3,4
In these expeiments, the errors of three differnt models were compared. The models are listed as below

- Model with all the features <br>
log( Bike Share Count ) = features = (β0 × const) + (β1 × Hour) + (β2 × Temperature(°C)) + (β3 ×
Humidity(%)) + (β4 × Wind speed (m/s)) + (β5 × Visibility (10m)) + (β6 × Dew point temperature(°C)) + (β7 ×
Solar Radiation (MJ/m2)) + (β8 × Rainfall(mm)) + (β9 × Snowfall (cm)) + (β10 × Holiday) + (β11 × Functioning
Day) + (β12 × Spring) + (β13 × Summer) + (β14 × Winter)


- Model for testing with 8 random features  <br>
log( Bike Share Count ) = features = (β0 × const) + (β1 × Wind speed (m/s)) + (β2 ×Humidity(%)) + (β3 ×
Visibility (10m)) + (β4 × Solar Radiation (MJ/m2)) + (β5 × Spring) + (β6 × Winter) + (β7 × 'Summer) + (β8 ×
Dew point temperature(°C))

Eight random features from the data set. The solution converged 1867 iterations and had a lower train and test
R-squared value of 0.23 and 0.21 respectively. The residuals are less uniformly distributed compared to the case of all the features being included.

- Model for testing with 8 features selected with intuition  <br>
log( Bike Share Count ) = features = (β0 × const) + (β1 × Hour) + (β2 × Temperature(°C)) + (β3 ×
Humidity(%)) + (β4 ×Wind speed (m/s)) + (β5 × Visibility (10m)) + (β6 × Solar Radiation (MJ/m2)) + (β7 ×
Rainfall(mm)) + (β8 × Snowfall (cm))

Eight features were intuitively selected on the basis of correlation from the heat map and general idea about how people might go about riding a bike on a given day. The model performed a little better than 8 randomly selected features but much worse in comparison with all the features. The model converges around 1089 iterations with a train and test scores of 0.265 and 0.243. The residuals distribution seems similar to randomly selected features.






