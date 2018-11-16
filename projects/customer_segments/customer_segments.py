#!/usr/bin/env python
# coding: utf-8

# # Machine Learning Engineer Nanodegree
# ## Unsupervised Learning
# ## Project: Creating Customer Segments

# Welcome to the third project of the Machine Learning Engineer Nanodegree! In this notebook, some template code has already been provided for you, and it will be your job to implement the additional functionality necessary to successfully complete this project. Sections that begin with **'Implementation'** in the header indicate that the following block of code will require additional functionality which you must provide. Instructions will be provided for each section and the specifics of the implementation are marked in the code block with a `'TODO'` statement. Please be sure to read the instructions carefully!
# 
# In addition to implementing code, there will be questions that you must answer which relate to the project and your implementation. Each section where you will answer a question is preceded by a **'Question X'** header. Carefully read each question and provide thorough answers in the following text boxes that begin with **'Answer:'**. Your project submission will be evaluated based on your answers to each of the questions and the implementation you provide.  
# 
# >**Note:** Code and Markdown cells can be executed using the **Shift + Enter** keyboard shortcut. In addition, Markdown cells can be edited by typically double-clicking the cell to enter edit mode.

# ## Getting Started
# 
# In this project, you will analyze a dataset containing data on various customers' annual spending amounts (reported in *monetary units*) of diverse product categories for internal structure. One goal of this project is to best describe the variation in the different types of customers that a wholesale distributor interacts with. Doing so would equip the distributor with insight into how to best structure their delivery service to meet the needs of each customer.
# 
# The dataset for this project can be found on the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Wholesale+customers). For the purposes of this project, the features `'Channel'` and `'Region'` will be excluded in the analysis — with focus instead on the six product categories recorded for customers.
# 
# Run the code block below to load the wholesale customers dataset, along with a few of the necessary Python libraries required for this project. You will know the dataset loaded successfully if the size of the dataset is reported.

# In[55]:


# Import libraries necessary for this project
import numpy as np
import pandas as pd
from IPython.display import display # Allows the use of display() for DataFrames

# Import supplementary visualizations code visuals.py
import visuals as vs

# Pretty display for notebooks
get_ipython().run_line_magic('matplotlib', 'inline')

# Load the wholesale customers dataset
try:
    data = pd.read_csv("customers.csv")
    data.drop(['Region', 'Channel'], axis = 1, inplace = True)
    print("Wholesale customers dataset has {} samples with {} features each.".format(*data.shape))
except:
    print("Dataset could not be loaded. Is the dataset missing?")


# ## Data Exploration
# In this section, you will begin exploring the data through visualizations and code to understand how each feature is related to the others. You will observe a statistical description of the dataset, consider the relevance of each feature, and select a few sample data points from the dataset which you will track through the course of this project.
# 
# Run the code block below to observe a statistical description of the dataset. Note that the dataset is composed of six important product categories: **'Fresh'**, **'Milk'**, **'Grocery'**, **'Frozen'**, **'Detergents_Paper'**, and **'Delicatessen'**. Consider what each category represents in terms of products you could purchase.

# In[56]:


# Display a description of the dataset
display(data.describe())


# ### Implementation: Selecting Samples
# To get a better understanding of the customers and how their data will transform through the analysis, it would be best to select a few sample data points and explore them in more detail. In the code block below, add **three** indices of your choice to the `indices` list which will represent the customers to track. It is suggested to try different sets of samples until you obtain customers that vary significantly from one another.

# In[57]:


# TODO: Select three indices of your choice you wish to sample from the dataset
indices = [0,130,430]

# Create a DataFrame of the chosen samples
samples = pd.DataFrame(data.loc[indices], columns = data.keys()).reset_index(drop = True)
print("Chosen samples of wholesale customers dataset:")
display(samples)


# ### Question 1
# Consider the total purchase cost of each product category and the statistical description of the dataset above for your sample customers.  
# 
# * What kind of establishment (customer) could each of the three samples you've chosen represent?
# 
# **Hint:** Examples of establishments include places like markets, cafes, delis, wholesale retailers, among many others. Avoid using names for establishments, such as saying *"McDonalds"* when describing a sample customer as a restaurant. You can use the mean values for reference to compare your samples with. The mean values are as follows:
# 
# * Fresh: 12000.2977
# * Milk: 5796.2
# * Grocery: 7951.3
# * Frozen: 3071.9
# * Detergents_paper: 2881.4
# * Delicatessen: 1524.8
# 
# Knowing this, how do your samples compare? Does that help in driving your insight into what kind of establishments they might be? 
# 

# **Answer:** The first customer (index 0) could be a bakery as it has relatively high values in the category "milk", whereas "frozen" is low and the other categories are average.
# The second customer (index 130) could be an ice cream seller, as "frozen" has a high value and the other categories rather low values.
# The last customer (index 430) could be a super market as it has high values for "grocery", "Delicatessen" and "Detergents_paper" but rather low values in the other categories.
# 
# Overall, it may be possible to deduce what kind of establishments they might be but further understanding and clustering of the data would help a lot with that.

# In[58]:


import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
get_ipython().run_line_magic('matplotlib', 'inline')

scaler = MinMaxScaler()
df = np.round(samples, 1)
index  = df.index[:]
print(index)
categories = list(df)
#display(df)
df = scaler.fit_transform(df)*100
#display(df)
N = len(categories)
angles = [n / float(N) * 2 * np.pi for n in range(N)]
print(angles)
angles += angles[:1]
print(angles)

plt.figure(figsize=(20, 10))
def Radar(index, title, color):
    ax = plt.subplot(1, 3, index+1, polar=True)
    ax.set_theta_offset(np.pi/2)
    ax.set_theta_direction(-1)
    plt.xticks(angles[:-1], categories, color ='grey', size = 10)
    plt.yticks((25, 50, 75, 100), ("1/4", "1/2", "3/4", "Max"), color = "grey", size = 7)
    values = df[index]
    values = np.append(values, values[:1])
    ax.plot(angles, values, color = color)
    ax.fill(angles, values, color=color, alpha=0.5)
    plt.title('Sample {}'.format(title), y= 1.1)

for i, n in enumerate(index):
    Radar(index=i, title=n, color='g')


# In[59]:


import matplotlib.pyplot as plt
import seaborn as sns

samples_for_plot = samples.copy()
samples_for_plot.loc[3] = data.median()
samples_for_plot.loc[4] = data.mean()

labels = ['Sample 1', 'Sample 2', 'Sample 3', 'Median', 'Mean']
samples_for_plot.plot(kind='bar', figsize=(15, 5))
plt.xticks(range(5), labels)
plt.show()


# ### Implementation: Feature Relevance
# One interesting thought to consider is if one (or more) of the six product categories is actually relevant for understanding customer purchasing. That is to say, is it possible to determine whether customers purchasing some amount of one category of products will necessarily purchase some proportional amount of another category of products? We can make this determination quite easily by training a supervised regression learner on a subset of the data with one feature removed, and then score how well that model can predict the removed feature.
# 
# In the code block below, you will need to implement the following:
#  - Assign `new_data` a copy of the data by removing a feature of your choice using the `DataFrame.drop` function.
#  - Use `sklearn.cross_validation.train_test_split` to split the dataset into training and testing sets.
#    - Use the removed feature as your target label. Set a `test_size` of `0.25` and set a `random_state`.
#  - Import a decision tree regressor, set a `random_state`, and fit the learner to the training data.
#  - Report the prediction score of the testing set using the regressor's `score` function.

# In[60]:


# TODO: Make a copy of the DataFrame, using the 'drop' function to drop the given feature
new_data = data.drop(['Frozen'], axis=1)

# TODO: Split the data into training and testing sets(0.25) using the given feature as the target
# Set a random state.
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(new_data,data['Frozen'],test_size=0.25, random_state=1)

#print(X_test)
# TODO: Create a decision tree regressor and fit it to the training set
from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state=1)
y_pred=regressor.fit(X_train,y_train).predict(X_test)
# TODO: Report the score of the prediction using the testing set
from sklearn.metrics import accuracy_score, r2_score
accscore = accuracy_score(y_pred, y_test)
r2score = r2_score(y_pred, y_test)
#print(y_test)
#print(y_pred)
#print(score(X_train, y_pred))
print(accscore)
print(r2score)


# In[61]:


# TODO: Split the data into training and testing sets(0.25) using the given feature as the target
# Set a random state.
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(new_data,data['Grocery'],test_size=0.25, random_state=1)

#print(X_test)
# TODO: Create a decision tree regressor and fit it to the training set
from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state=1)
y_pred=regressor.fit(X_train,y_train).predict(X_test)
# TODO: Report the score of the prediction using the testing set
from sklearn.metrics import accuracy_score, r2_score
accscore = accuracy_score(y_pred, y_test)
r2score = r2_score(y_pred, y_test)
#print(y_test)
#print(y_pred)
#print(score(X_train, y_pred))
print(accscore)
print(r2score)


# ### Question 2
# 
# * Which feature did you attempt to predict? 
# * What was the reported prediction score? 
# * Is this feature necessary for identifying customers' spending habits?
# 
# **Hint:** The coefficient of determination, `R^2`, is scored between 0 and 1, with 1 being a perfect fit. A negative `R^2` implies the model fails to fit the data. If you get a low score for a particular feature, that lends us to beleive that that feature point is hard to predict using the other features, thereby making it an important feature to consider when considering relevance.

# **Answer:** I attempted to predit "Frozen". The reported accuracy score was 0.0 and the r2 score -0.9927. Therefore, "Frozen" is necessary for indentifying customers' spending habits as it is hard to predict itself.
# 
# After the review, I attempted to predict "Grocery". The reported accuracy score 0.0182 and the r2 score 0.897. Therefore, "Grocery" might be sufficiently predicted from the other features.

# In[62]:


import seaborn as sns
sns.barplot(X_train.columns, regressor.feature_importances_)


# ### Visualize Feature Distributions
# To get a better understanding of the dataset, we can construct a scatter matrix of each of the six product features present in the data. If you found that the feature you attempted to predict above is relevant for identifying a specific customer, then the scatter matrix below may not show any correlation between that feature and the others. Conversely, if you believe that feature is not relevant for identifying a specific customer, the scatter matrix might show a correlation between that feature and another feature in the data. Run the code block below to produce a scatter matrix.

# In[63]:


# Produce a scatter matrix for each pair of features in the data
pd.plotting.scatter_matrix(data, alpha = 0.3, figsize = (14,8), diagonal = 'kde');


# In[64]:


axes = pd.scatter_matrix(data, alpha = 0.3, figsize = (14,8), diagonal = 'kde')
corr = data.corr().as_matrix()
for i, j in zip(*np.triu_indices_from(axes, k=1)):
    axes[i, j].annotate("%.3f" %corr[i,j], (0.8, 0.8), xycoords='axes fraction', ha='center', va='center')


# ### Question 3
# * Using the scatter matrix as a reference, discuss the distribution of the dataset, specifically talk about the normality, outliers, large number of data points near 0 among others. If you need to sepearate out some of the plots individually to further accentuate your point, you may do so as well.
# * Are there any pairs of features which exhibit some degree of correlation? 
# * Does this confirm or deny your suspicions about the relevance of the feature you attempted to predict? 
# * How is the data for those features distributed?
# 
# **Hint:** Is the data normally distributed? Where do most of the data points lie? You can use [corr()](https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.corr.html) to get the feature correlations and then visualize them using a [heatmap](http://seaborn.pydata.org/generated/seaborn.heatmap.html) (the data that would be fed into the heatmap would be the correlation values, for eg: `data.corr()`) to gain further insight.

# In[65]:


correlation = data[['Grocery','Milk']].corr(method='pearson')
print(correlation)
correlation = data[['Grocery','Detergents_Paper']].corr(method='pearson')
print(correlation)
correlation = data[['Milk','Detergents_Paper']].corr(method='pearson')
print(correlation)
correlation = data[['Fresh','Frozen']].corr(method='pearson')
print(correlation)
correlation = data[['Delicatessen','Frozen']].corr(method='pearson')
print(correlation)


# In[66]:


from string import ascii_letters
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

sns.set(style="white")


# Compute the correlation matrix
corr = data.corr()
print(corr)
# Generate a mask for the upper triangle
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(12, 12))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(220, 10, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
            square=True, annot=True, linewidths=.5, cbar_kws={"shrink": .5})


# 
# **Answer:** The dataset does not seem to be normally distributed. Most points in the scatter matrix lie close to the origin, with the exception of some outliers, or in the case of Grocery/Detergents_Paper where a linear relation can be seen. Also for Milk/Grocery or Milk/Detergents_Paper a linear relations seems to exist by inspecting the scatter matrix.
# In the heat map , more easily other corelations are detected, e.g. between Fresh/Frozen, or Frozen/Delicatessen. The latter are rather low though as shown when compution with corr().

# ## Data Preprocessing
# In this section, you will preprocess the data to create a better representation of customers by performing a scaling on the data and detecting (and optionally removing) outliers. Preprocessing data is often times a critical step in assuring that results you obtain from your analysis are significant and meaningful.

# ### Implementation: Feature Scaling
# If data is not normally distributed, especially if the mean and median vary significantly (indicating a large skew), it is most [often appropriate](http://econbrowser.com/archives/2014/02/use-of-logarithms-in-economics) to apply a non-linear scaling — particularly for financial data. One way to achieve this scaling is by using a [Box-Cox test](http://scipy.github.io/devdocs/generated/scipy.stats.boxcox.html), which calculates the best power transformation of the data that reduces skewness. A simpler approach which can work in most cases would be applying the natural logarithm.
# 
# In the code block below, you will need to implement the following:
#  - Assign a copy of the data to `log_data` after applying logarithmic scaling. Use the `np.log` function for this.
#  - Assign a copy of the sample data to `log_samples` after applying logarithmic scaling. Again, use `np.log`.

# In[67]:


# TODO: Scale the data using the natural logarithm
log_data = np.log(data)
#print(log_data)

# TODO: Scale the sample data using the natural logarithm
log_samples =  np.log(samples)

# Produce a scatter matrix for each pair of newly-transformed features
pd.plotting.scatter_matrix(log_data, alpha = 0.3, figsize = (14,8), diagonal = 'kde');


# ### Observation
# After applying a natural logarithm scaling to the data, the distribution of each feature should appear much more normal. For any pairs of features you may have identified earlier as being correlated, observe here whether that correlation is still present (and whether it is now stronger or weaker than before).
# 
# Run the code below to see how the sample data has changed after having the natural logarithm applied to it.

# In[68]:


# Display the log-transformed sample data
display(log_samples)


# ### Implementation: Outlier Detection
# Detecting outliers in the data is extremely important in the data preprocessing step of any analysis. The presence of outliers can often skew results which take into consideration these data points. There are many "rules of thumb" for what constitutes an outlier in a dataset. Here, we will use [Tukey's Method for identfying outliers](http://datapigtechnologies.com/blog/index.php/highlighting-outliers-in-your-data-with-the-tukey-method/): An *outlier step* is calculated as 1.5 times the interquartile range (IQR). A data point with a feature that is beyond an outlier step outside of the IQR for that feature is considered abnormal.
# 
# In the code block below, you will need to implement the following:
#  - Assign the value of the 25th percentile for the given feature to `Q1`. Use `np.percentile` for this.
#  - Assign the value of the 75th percentile for the given feature to `Q3`. Again, use `np.percentile`.
#  - Assign the calculation of an outlier step for the given feature to `step`.
#  - Optionally remove data points from the dataset by adding indices to the `outliers` list.
# 
# **NOTE:** If you choose to remove any outliers, ensure that the sample data does not contain any of these points!  
# Once you have performed this implementation, the dataset will be stored in the variable `good_data`.

# In[69]:


# For each feature find the data points with extreme high or low values
outliers_number=[0 for i in range(0,len(log_data))]
for feature in log_data.keys():
    #print(feature)
    # TODO: Calculate Q1 (25th percentile of the data) for the given feature
    Q1 = np.percentile(log_data[feature], 25)
    
    # TODO: Calculate Q3 (75th percentile of the data) for the given feature
    Q3 = np.percentile(log_data[feature], 75)
    
    # TODO: Use the interquartile range to calculate an outlier step (1.5 times the interquartile range)
    step = 1.5*(Q3 - Q1)
    
    print(step)
    # Display the outliers
    print("Data points considered outliers for the feature '{}':".format(feature))
    o=log_data[~((log_data[feature] >= Q1 - step) & (log_data[feature] <= Q3 + step))]
    display(o)
    for i in range(0,len(o)):
        outliers_number[o.iloc[i].name]=outliers_number[o.iloc[i].name]+1

for i in range(0,len(outliers_number)):
    if outliers_number[i]>1: 
        print(i)

# OPTIONAL: Select the indices for data points you wish to remove
outliers  = []
for i in range(0,len(outliers_number)):
    if outliers_number[i]>0: 
        outliers.append(i)
#outliers  = #[66,95,96,218,338,75,38,65,420,75,122,142,154,161,177,204,237,289,338,356,402]#[]#[66,142,154,289]#
print(outliers)

# Remove the outliers, if any were specified
good_data = log_data.drop(log_data.index[outliers]).reset_index(drop = True)


# In[70]:


# Heatmap using percentiles to display outlier data
import matplotlib.pyplot as plt
import seaborn as sns
percentiles = data.rank(pct=True)
percentiles = percentiles.iloc[[65, 66, 75, 128, 154]]
plt.title('Multiple Outliers Heatmap', fontsize=14)
heat = sns.heatmap(percentiles, annot=True)
display(heat)


# ### Question 4
# * Are there any data points considered outliers for more than one feature based on the definition above? 
# * Should these data points be removed from the dataset? 
# * If any data points were added to the `outliers` list to be removed, explain why.
# 
# ** Hint: ** If you have datapoints that are outliers in multiple categories think about why that may be and if they warrant removal. Also note how k-means is affected by outliers and whether or not this plays a factor in your analysis of whether or not to remove them.

# **Answer:** Yes, I found that number 66 is an outlier for Delicatessen/Fresh, and 142/154/289 for Delicatessen/Detergents_Paper.
# The provided above link on Tukey's Method for identfying outliers (which by the way is only accessible in cached mode via Google) warns to not remove outliers from the data but recommends to understand why they are there.
# Especially in the case of outliers for more than one feature I think we might have a special type of customer.
# However, to become a general understanding of the general data points and clusters I decided to remove all outliers from Tukey's Method.
# 
# UPDATE after review: I checked the outliers for each feature and saw now that 65, 66, 75, 128, 154 are outliers for more than one feature.

# ## Feature Transformation
# In this section you will use principal component analysis (PCA) to draw conclusions about the underlying structure of the wholesale customer data. Since using PCA on a dataset calculates the dimensions which best maximize variance, we will find which compound combinations of features best describe customers.

# ### Implementation: PCA
# 
# Now that the data has been scaled to a more normal distribution and has had any necessary outliers removed, we can now apply PCA to the `good_data` to discover which dimensions about the data best maximize the variance of features involved. In addition to finding these dimensions, PCA will also report the *explained variance ratio* of each dimension — how much variance within the data is explained by that dimension alone. Note that a component (dimension) from PCA can be considered a new "feature" of the space, however it is a composition of the original features present in the data.
# 
# In the code block below, you will need to implement the following:
#  - Import `sklearn.decomposition.PCA` and assign the results of fitting PCA in six dimensions with `good_data` to `pca`.
#  - Apply a PCA transformation of `log_samples` using `pca.transform`, and assign the results to `pca_samples`.

# In[71]:


# TODO: Apply PCA by fitting the good data with the same number of dimensions as features
from sklearn.decomposition import PCA
pca = PCA().fit(good_data)

# TODO: Transform log_samples using the PCA fit above
pca_samples =pca.transform(log_samples)

# Generate PCA results plot
pca_results = vs.pca_results(good_data, pca)


# ### Question 5
# 
# * How much variance in the data is explained* **in total** *by the first and second principal component? 
# * How much variance in the data is explained by the first four principal components? 
# * Using the visualization provided above, talk about each dimension and the cumulative variance explained by each, stressing upon which features are well represented by each dimension(both in terms of positive and negative variance explained). Discuss what the first four dimensions best represent in terms of customer spending.
# 
# **Hint:** A positive increase in a specific dimension corresponds with an *increase* of the *positive-weighted* features and a *decrease* of the *negative-weighted* features. The rate of increase or decrease is based on the individual feature weights.

# **Answer:** *Updated after Review* The first and second principal component explains 0.7252 variance in the data, the first four pricipal components even explain 0.9279 variance in the data.
# 
# - Dimension 1: The first dimension is strongly associated with the features Milk, Detergents_Paper and Grocery, which fits to our above observed corelations between these three features. An increase in Dimension 1 ist associated with a large increase in these three features.
# - Dimension 2: This dimension is strongly associated with the three remaining features Fresh, Frozen and Delicatessen. That is, an increase in Dimension 2 is associated with a large increase in Fresh, Frozen and Delicatessen.
# - Dimension 3: This dimension is strongly negatively associated with Fresh, negatively associated with Detergents_Paper, almost neglectable associated with Grocery and positively associated with the other three features. In particular, an increase of Dimension 3 corresponds to a large decrease in Fresh, a decrease in Detergents_Paper and an increase in Delicatessen, Grocery, and Frozen.
# - Dimension 4: This dimension is positively associated with Delicatessen, Fresh, Milk and Grocery,and negatively associated with Frozen and Detergents_Paper. Therefore an increase in Dimension 4 corresponds to an increase in  Delicatessen, Fresh, Milk and Grocery, and a decrease in Frozen and Detergents_Paper.

# ### Observation
# Run the code below to see how the log-transformed sample data has changed after having a PCA transformation applied to it in six dimensions. Observe the numerical value for the first four dimensions of the sample points. Consider if this is consistent with your initial interpretation of the sample points.

# In[72]:


# Display sample log-data after having a PCA transformation applied
display(pd.DataFrame(np.round(pca_samples, 4), columns = pca_results.index.values))


# ### Implementation: Dimensionality Reduction
# When using principal component analysis, one of the main goals is to reduce the dimensionality of the data — in effect, reducing the complexity of the problem. Dimensionality reduction comes at a cost: Fewer dimensions used implies less of the total variance in the data is being explained. Because of this, the *cumulative explained variance ratio* is extremely important for knowing how many dimensions are necessary for the problem. Additionally, if a signifiant amount of variance is explained by only two or three dimensions, the reduced data can be visualized afterwards.
# 
# In the code block below, you will need to implement the following:
#  - Assign the results of fitting PCA in two dimensions with `good_data` to `pca`.
#  - Apply a PCA transformation of `good_data` using `pca.transform`, and assign the results to `reduced_data`.
#  - Apply a PCA transformation of `log_samples` using `pca.transform`, and assign the results to `pca_samples`.

# In[73]:


# TODO: Apply PCA by fitting the good data with only two dimensions
pca = PCA(n_components=2).fit(good_data)

# TODO: Transform the good data using the PCA fit above
reduced_data = pca.transform(good_data)

# TODO: Transform log_samples using the PCA fit above
pca_samples = pca.transform(log_samples)

# Create a DataFrame for the reduced data
reduced_data = pd.DataFrame(reduced_data, columns = ['Dimension 1', 'Dimension 2'])


# ### Observation
# Run the code below to see how the log-transformed sample data has changed after having a PCA transformation applied to it using only two dimensions. Observe how the values for the first two dimensions remains unchanged when compared to a PCA transformation in six dimensions.

# In[74]:


# Display sample log-data after applying PCA transformation in two dimensions
display(pd.DataFrame(np.round(pca_samples, 4), columns = ['Dimension 1', 'Dimension 2']))


# ## Visualizing a Biplot
# A biplot is a scatterplot where each data point is represented by its scores along the principal components. The axes are the principal components (in this case `Dimension 1` and `Dimension 2`). In addition, the biplot shows the projection of the original features along the components. A biplot can help us interpret the reduced dimensions of the data, and discover relationships between the principal components and original features.
# 
# Run the code cell below to produce a biplot of the reduced-dimension data.

# In[75]:


# Create a biplot
vs.biplot(good_data, reduced_data, pca)


# ### Observation
# 
# Once we have the original feature projections (in red), it is easier to interpret the relative position of each data point in the scatterplot. For instance, a point the lower right corner of the figure will likely correspond to a customer that spends a lot on `'Milk'`, `'Grocery'` and `'Detergents_Paper'`, but not so much on the other product categories. 
# 
# From the biplot, which of the original features are most strongly correlated with the first component? What about those that are associated with the second component? Do these observations agree with the pca_results plot you obtained earlier?

# ## Clustering
# 
# In this section, you will choose to use either a K-Means clustering algorithm or a Gaussian Mixture Model clustering algorithm to identify the various customer segments hidden in the data. You will then recover specific data points from the clusters to understand their significance by transforming them back into their original dimension and scale. 

# ### Question 6
# 
# * What are the advantages to using a K-Means clustering algorithm? 
# * What are the advantages to using a Gaussian Mixture Model clustering algorithm? 
# * Given your observations about the wholesale customer data so far, which of the two algorithms will you use and why?
# 
# ** Hint: ** Think about the differences between hard clustering and soft clustering and which would be appropriate for our dataset.

# **Answer:** 
# - The advantages when using Gaussian Mixture Model is that it is much more flexible as a point can belong to different clusters (with varying degrees). The clusters are not only spherical as in K-Means.
# - K-Means is of advantage if the data points can belong only to one cluster. K-Means is also often computationally faster than e.g. hierarchical clustering.
# 
# 
# I choose 2 clusters as the two dimensions seem to have a relatively low covariance, and one data point should belong to one specific cluster.

# ### Implementation: Creating Clusters
# Depending on the problem, the number of clusters that you expect to be in the data may already be known. When the number of clusters is not known *a priori*, there is no guarantee that a given number of clusters best segments the data, since it is unclear what structure exists in the data — if any. However, we can quantify the "goodness" of a clustering by calculating each data point's *silhouette coefficient*. The [silhouette coefficient](http://scikit-learn.org/stable/modules/generated/sklearn.metrics.silhouette_score.html) for a data point measures how similar it is to its assigned cluster from -1 (dissimilar) to 1 (similar). Calculating the *mean* silhouette coefficient provides for a simple scoring method of a given clustering.
# 
# In the code block below, you will need to implement the following:
#  - Fit a clustering algorithm to the `reduced_data` and assign it to `clusterer`.
#  - Predict the cluster for each data point in `reduced_data` using `clusterer.predict` and assign them to `preds`.
#  - Find the cluster centers using the algorithm's respective attribute and assign them to `centers`.
#  - Predict the cluster for each sample data point in `pca_samples` and assign them `sample_preds`.
#  - Import `sklearn.metrics.silhouette_score` and calculate the silhouette score of `reduced_data` against `preds`.
#    - Assign the silhouette score to `score` and print the result.

# In[76]:


#Code from the review:
import time
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt

n = 1000
k = 6

kmeans_train_times = []
for k in range(1, 7):
    cum_time = 0.
    for i in range(n):
        start = time.time()
        KMeans(n_clusters=k).fit(reduced_data)
        cum_time += (time.time() - start)

    train_time = cum_time / n
    kmeans_train_times.append([k, train_time])

km_df = pd.DataFrame(kmeans_train_times, columns=['KM_Clusters', 'KM_Time'])

gmm_train_times = []
for k in range(1, 7):
    cum_time = 0.
    for i in range(n):
        start = time.time()
        GaussianMixture(n_components=k).fit(reduced_data)
        cum_time += (time.time() - start)

    train_time = cum_time / n
    gmm_train_times.append([k, train_time])

gmm_df = pd.DataFrame(gmm_train_times, columns=['GMM_Components', 'GMM_Time'])
times_df = km_df.join(gmm_df)

plt.plot(times_df.GMM_Components, times_df.GMM_Time * 1000., label='GMM Train Time')
plt.plot(times_df.GMM_Components, times_df.KM_Time * 1000., label='Kmeans Train Time')
plt.legend(loc='best')
plt.ylabel('Train time (in millisec.)')
plt.xlabel('Cluster/Components Used in Training')
plt.title('Training Time for Different Cluster/Component Sizes \n Averaged Over {} Runs Per Size'.format(n))
plt.show()


# In[77]:


# TODO: Apply your clustering algorithm of choice to the reduced data 
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
clusterer = KMeans(n_clusters=2,random_state=0).fit(reduced_data)

# TODO: Predict the cluster for each data point
preds = clusterer.predict(reduced_data)

# TODO: Find the cluster centers
centers = clusterer.cluster_centers_

# TODO: Predict the cluster for each transformed sample data point
sample_preds = clusterer.predict(pca_samples)

# TODO: Calculate the mean silhouette coefficient for the number of clusters chosen
from sklearn.metrics import silhouette_score
score = silhouette_score(reduced_data,preds)
print(score)


# ### Question 7
# 
# * Report the silhouette score for several cluster numbers you tried. 
# * Of these, which number of clusters has the best silhouette score?

# **Answer:** *Updated after review* For 2 clusters the silhouette score is 0.447157742293 (highest), for three clusters it is 0.36398647984 and for four clusters it is 0.331150954285.

# ### Cluster Visualization
# Once you've chosen the optimal number of clusters for your clustering algorithm using the scoring metric above, you can now visualize the results by executing the code block below. Note that, for experimentation purposes, you are welcome to adjust the number of clusters for your clustering algorithm to see various visualizations. The final visualization provided should, however, correspond with the optimal number of clusters. 

# In[78]:


# Display the results of the clustering from implementation
vs.cluster_results(reduced_data, preds, centers, pca_samples)


# In[79]:


#from the review
for i, pred in enumerate(sample_preds):
    print("Sample point", i, "predicted to be in Cluster", pred)
    print('The distance between sample point {} and center of cluster {}:'.format(i, pred))
    print((samples.iloc[i] - true_centers.iloc[pred]))


# ### Implementation: Data Recovery
# Each cluster present in the visualization above has a central point. These centers (or means) are not specifically data points from the data, but rather the *averages* of all the data points predicted in the respective clusters. For the problem of creating customer segments, a cluster's center point corresponds to *the average customer of that segment*. Since the data is currently reduced in dimension and scaled by a logarithm, we can recover the representative customer spending from these data points by applying the inverse transformations.
# 
# In the code block below, you will need to implement the following:
#  - Apply the inverse transform to `centers` using `pca.inverse_transform` and assign the new centers to `log_centers`.
#  - Apply the inverse function of `np.log` to `log_centers` using `np.exp` and assign the true centers to `true_centers`.
# 

# In[80]:


# TODO: Inverse transform the centers
log_centers = pca.inverse_transform(centers)

# TODO: Exponentiate the centers
true_centers = np.exp(log_centers)

# Display the true centers
segments = ['Segment {}'.format(i) for i in range(0,len(centers))]
true_centers = pd.DataFrame(np.round(true_centers), columns = data.keys())
true_centers.index = segments
display(true_centers)


# ### Question 8
# 
# * Consider the total purchase cost of each product category for the representative data points above, and reference the statistical description of the dataset at the beginning of this project(specifically looking at the mean values for the various feature points). What set of establishments could each of the customer segments represent?
# 
# **Hint:** A customer who is assigned to `'Cluster X'` should best identify with the establishments represented by the feature set of `'Segment X'`. Think about what each segment represents in terms their values for the feature points chosen. Reference these values with the mean values to get some perspective into what kind of establishment they represent.

# **Answer:** *Updated after review*
# From above, the mean values of the data set are:
# Fresh: 12000.2977
# Milk: 5796.2
# Grocery: 7951.3
# Frozen: 3071.9
# Detergents_paper: 2881.4
# Delicatessen: 1524.8
# 
# Therefore, in Segment 1, Fresh is much lower than average, Milk and Grocery are higher, Frozen is much lower, Detergents_Paper are higher and Delicatessen are lower but higher than in Segment 0.
# In Segment 0, Fresh is a bit higher than average, Milk and Grocery are much lower, Frozen is higher, Detergents_paper are much lower and Delicatessen are a bit lower.
# 
# I think that Segment 1 should be something like a super market even though Frozen is much lower. 
# Segment 0 could be a bakery or restaurant due to fresher and more frozen products.

# In[81]:


compare = true_centers.copy()
compare.loc[true_centers.shape[0]] = data.median()

plt.style.use('ggplot')
compare.plot(kind='bar', figsize=(15, 5))
labels = true_centers.index.values.tolist()
labels.append('Data Median')
plt.xticks(range(compare.shape[0]), labels)
plt.show()


# ### Question 9
# 
# * For each sample point, which customer segment from* **Question 8** *best represents it? 
# * Are the predictions for each sample point consistent with this?*
# 
# Run the code block below to find which cluster each sample point is predicted to be.

# In[82]:


print(samples)
# Display the predictions
for i, pred in enumerate(sample_preds):
    print("Sample point", i, "predicted to be in Cluster", pred)


# **Answer:** *Updated after review*
# The first customer (index 0) should rather be in Cluster 1 as he has a high value for Milk, a low value for Frozen and a high value for Delicatessen. This is consistent with its prediction to be in Cluster 1. As I said in Answer 8, Cluster 1 could be a super market, this is contrary to that I said in the beginning that the first customer could be a bakery. 
# 
# The second customer (index 130) should rather be in Cluster 0 as he has a low value for Milk, Grocery and Detergents_paper, and a high value for Frozen. This fits to its prediction to be in Cluster 0. As I said in Answer 8, Cluster 0 could be a bakery or restaurant so this fits a bit to me saying in the beginning that the second customer could be an ice cream seller. 
# 
# The last customer (index 430) is not that easy to associate to a Cluster. He has high values for Grocery and Delicatessen consistent with Cluster 1 but Detergents_Paper is very low. Fresh and Frozen is low, fitting rather to Cluster 1 too so I would say the last customer belongs to Cluster 1. It is predicted to be in Cluster 0 though.
# In the beginning I said the last customer could be a super market, which would rather fit to Cluster 1 by my Answer 8.
# 
# 

# ## Conclusion

# In this final section, you will investigate ways that you can make use of the clustered data. First, you will consider how the different groups of customers, the ***customer segments***, may be affected differently by a specific delivery scheme. Next, you will consider how giving a label to each customer (which *segment* that customer belongs to) can provide for additional features about the customer data. Finally, you will compare the ***customer segments*** to a hidden variable present in the data, to see whether the clustering identified certain relationships.

# ### Question 10
# Companies will often run [A/B tests](https://en.wikipedia.org/wiki/A/B_testing) when making small changes to their products or services to determine whether making that change will affect its customers positively or negatively. The wholesale distributor is considering changing its delivery service from currently 5 days a week to 3 days a week. However, the distributor will only make this change in delivery service for customers that react positively. 
# 
# * How can the wholesale distributor use the customer segments to determine which customers, if any, would react positively to the change in delivery service?*
# 
# **Hint:** Can we assume the change affects all customers equally? How can we determine which group of customers it affects the most?

# **Answer:** These changes will not affect all customers equally. Especially in Cluster 0 Fresh is much lower than average so it may be sufficient to deliver 3 days a week. If Cluster 0 represents super markets, these may have sufficient storage so that they can store goods from larger and less frequent deliveries.
# However, a restaurant or bakery has less storage and more fresh ingredients so in Cluster 1 it would be more difficult to change to a 3 days a week intervall.
# It would be either good to pick a few customers with values close to the center of Custer 0 resp. Cluster 1 to get a representative idea if delivering 3 days a week would be sufficient.
# 
# *Updated after second review:*
# 
# The wholesale distributor should apply A/B testing on each of the two segments separately. That is, he should pick randomly the same percentage of customers (probably 50%) for the 3 days a week variant and the rest for the 5 days a week variant on each cluster. Looking at each clustor seperately ensures that the customer profiles testing variant A vs variant B are highly similar.
# 
# After measuring the customer satisfaction with respect to the 3 vs 5 days a week delivery with a e.g. a survey on each of the clusters, he can then select a segmented strategy as a result of the A/B test  for each cluster.

# ### Question 11
# Additional structure is derived from originally unlabeled data when using clustering techniques. Since each customer has a ***customer segment*** it best identifies with (depending on the clustering algorithm applied), we can consider *'customer segment'* as an **engineered feature** for the data. Assume the wholesale distributor recently acquired ten new customers and each provided estimates for anticipated annual spending of each product category. Knowing these estimates, the wholesale distributor wants to classify each new customer to a ***customer segment*** to determine the most appropriate delivery service.  
# * How can the wholesale distributor label the new customers using only their estimated product spending and the **customer segment** data?
# 
# **Hint:** A supervised learner could be used to train on the original customers. What would be the target variable?

# **Answer:** We could use any supervised learning algorithm like Decision Tree Classifier or Support Vector Machines to predict the target variable Customer Segment.

# ### Visualizing Underlying Distributions
# 
# At the beginning of this project, it was discussed that the `'Channel'` and `'Region'` features would be excluded from the dataset so that the customer product categories were emphasized in the analysis. By reintroducing the `'Channel'` feature to the dataset, an interesting structure emerges when considering the same PCA dimensionality reduction applied earlier to the original dataset.
# 
# Run the code block below to see how each data point is labeled either `'HoReCa'` (Hotel/Restaurant/Cafe) or `'Retail'` the reduced space. In addition, you will find the sample points are circled in the plot, which will identify their labeling.

# In[83]:


# Display the clustering results based on 'Channel' data
vs.channel_results(reduced_data, outliers, pca_samples)


# ### Question 12
# 
# * How well does the clustering algorithm and number of clusters you've chosen compare to this underlying distribution of Hotel/Restaurant/Cafe customers to Retailer customers? 
# * Are there customer segments that would be classified as purely 'Retailers' or 'Hotels/Restaurants/Cafes' by this distribution? 
# * Would you consider these classifications as consistent with your previous definition of the customer segments?

# **Answer:**
# - My above result with Cluster 0 and Cluster 1 is very similar to this distribultion using Channel.
# - The customer segment to the left would classify as Hotels/Restaurants/Cafes, and the customer segment to the right as Retailers.
# 
# *Updated after review*
# - This distribution is consistent with our assumption that Cluster 1 are super markets and Cluster 0 bakeries or restaurants.

# In[84]:


#from review
#find percentage of correctly classified customers
data = pd.read_csv("customers.csv")
data = data.drop(data.index[outliers]).reset_index(drop = True)
# might need to switch around the 0 and 1, based on your cluster seed
df = np.where(data['Channel'] == 2, 1, 0)
print("Percentage of correctly classified customers: {:.2%}".format(sum(df == preds)/float(len(preds))))


# > **Note**: Once you have completed all of the code implementations and successfully answered each question above, you may finalize your work by exporting the iPython Notebook as an HTML document. You can do this by using the menu above and navigating to  
# **File -> Download as -> HTML (.html)**. Include the finished document along with this notebook as your submission.
