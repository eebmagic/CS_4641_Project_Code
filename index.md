# Predicting Housing Prices in US Counties from US Census Data and Other Local Features:
## Members:
- Ethan Bolton
- Carlo Casas
- Evan Chase
- Felix Pei
- Sagar Singhal

---

## Background and Problem Definition
Understanding how housing prices correlate with demographics can help to predict how property values will adjust to changes in the population, income, ethnicity, areas of employment, and other features of an area. This insight can help investors in real estate decide where to buy and sell property and could inform governments on how they can make housing more affordable. Additionally this tool could be used to predict future changes rent before rent is adjusted to correspond with local changes in an area.

This project aims to predict housing prices of counties in the United States given demographic data and discover relevant correlations using supervised machine learning with the American Community Survey of 2015 and the Zillow Rent Index, which tracks the median housing price per square foot of a given area.

## Data Collection and Preprocessing
Two different datasets from Kaggle were selected in this analysis -- the 2015 U.S Census and the Zillow Rent Index. 

The Zillow Rent Index is a dollar-valued index that captures average market rent for a given demographic and/or geographic group. Much of this data has been prescreened and cleaned by Zillow by removing outliers and weighting the homes that actually rent as higher. The data provided details the Rent Index for each month for each U.S City (and corresponding county which will be used to match with the census) from 2010 to 2017. The data has 13131 unique cities and 1820 counties. The U.S Census data on the other hand includes demographic and geographic columns by U.S County. These columns include things like income, race, poverty levels, Voting Age, and gender.

To clean the data, the first thing we did was remove the counties that don't overlap between the two data sets. The datasets were then sorted and combined into one, with each numerical column normalized.

## Methods
Our problem is a regression problem, where we attempt to approximate the relationship between independent variables (population, etc.) and a dependent variable (median rent).

Next, we need to learn a mapping from the input features to the output. There are many approaches we can choose from for this regression task. From linear approaches, we can experiment with simple linear regression, Lasso regression, and GLMs like Gamma regression. Among non-linear approaches, we can try support vector regression and feedforward neural networks, and we can also use additional input features generated from non-linear transformations of original inputs with the linear methods. Afterward, we can compare training time and performance of these various approaches, and we can probe the trained models to see what features are particularly informative of the output.

We began with ridge regression as our intitial approach. Ridge regression is often effective in problems with high correlation between features, as often seem with economic/demographic data. We studied the effect of preprocessing the data before running regression, by using forward, backward, and lasso feature selection.  

## Potential Results / Discussion
The regression analysis will yield a relationship between median rent of United States counties and the various demographics of each. Given the wide range of demographic data from the census dataset, we seek to find what parameter or set of parameters correlates to the highest or lowest rent prices. Examples of these demographic parameters include age, ethnicity, income, poverty, and unemployment, commute time, industry distribution, etc. While some parameters seem directly correlated, others may yield unexpected dependence to rent. 

From our preprocessing of the data, we identify forward feature selection as the most effective. As to the actual effectiveness of the preprocessing, we saw a slight improvement in the R2 value of our ridge regression, as illustrated below.

![](/results/Ridge_NoForward.png)
![](/results/Ridge_WithForward.png)

## Proposed Timeline
### Project Proposal (10-7)

The main idea of what our project - estimating rent based on the defining parameters of a County in the US - will consist of. This will be a good guideline for the steps we need to take in the project.

| Background and Problem Definition | Methods | Potential Results and Discussion | References | Timeline | Proposal Video |
| --------------------------------- | ------- | -------------------------------- | ---------- | -------- | -------------- |
| Sagar | Felix | Evan | Sagar | Carlo | Ethan |

### Pre-Processing (10-22)
We plan to collectively work to clean the data and manipulate it into a format that we can easily use in our models. Within this group effort, Felix and Carlo will take the numerical data and normalize it while Sagar and Ethan do similar manipulation with the empirical data. Once this is done we need to combine the data we have from two separate datasets into a form we can work with together more easily in the final model - Evan's role. 

### Project Midpoint Report (11-16)
At this point we hope to have tried various methods for predicting output from all or some of our variables. It is difficult to say how much will be completed at this point with little guidelines up to this point, but even without a fully optimum model or solution, we at least hope to have some results to show at this milestone. We each plan to try to fit our data to one model or another. Sagar and Ethan will attempt as many linear approaches as possible while Felix, Carlo, and Evan plan to do the same with non-linear methods. 

### Final Project (12-7)
With the final project, the main step from the midpoint report will hopefully be optimization and a finalized algorithm for producing more accurate predictions. With a finalized algorithm, hopefully we can not only predict an output from the others, but know which parameters have the greatest impact on our output. After the midpoint report we will collectively choose which model fits our needs best and work together to optimize the hyperparameters. 

---

## References
MuonNeutrino. (2019). US Census Demographic Data (Version 3) [Data file] Retrieved from https://www.kaggle.com/muonneutrino/us-census-demographic-data.

US Census Bureau. (2019) ACS Demographic and Housing Estimates https://data.census.gov/cedsci/table?q=demographic&tid=ACSDP1Y2019.DP05

Schuetz, Jenny. “How Can Government Make Housing More Affordable?” *Policy 2020: Voter Vials*, Brookings, 15 Oct. 2019, https://www.brookings.edu/policy2020/votervital/how-can-government-make-housing-more-affordable/.

Zillow Group. (2017). Zillow Rent Index, 2010-Present (Version 1) [Data file] Retrieved from https://www.kaggle.com/zillow/rent-index.
