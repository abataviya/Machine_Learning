import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
pd.set_option('display.max_columns',None)

df= pd.read_csv('data/car data.csv')
print(df.columns.values)
print()
print(df.head())
print(df.shape)

# Identifying categorical features and find unique of them
print(df['Seller_Type'].unique())
print()
print(df['Transmission'].unique())
print()
print(df['Owner'].unique())
print(df['Fuel_Type'].unique())

# Checking missing/null values
print(df.isnull().sum())
print()
print(df.describe())

final_dataset= df[['Year','Selling_Price','Present_Price','Kms_Driven','Fuel_Type','Seller_Type','Transmission','Owner']]
print(final_dataset.head())
final_dataset['Current_Year']= 2020
print()
print(final_dataset)
final_dataset['no_years']= final_dataset['Current_Year']-final_dataset['Year']

final_dataset.drop(['Year','Current_Year'], axis=1, inplace=True)
print(final_dataset.head())

final_dataset= pd.get_dummies(final_dataset,drop_first=True)
print(final_dataset.head())
print(final_dataset.columns.values)

print(final_dataset.corr())

corrmat = final_dataset.corr()
top_corr_features= corrmat.index
plt.figure(figsize=(20,20))
# Plot heatmap
# g= sns.heatmap(final_dataset[top_corr_features].corr(), annot=True, cmap="RdYlGn")
# plt.show()

# independent and dependent features
X= final_dataset.iloc[:,1:]
y= final_dataset.iloc[:,0]
print(X.head(3))
print(y.head(3))

# Feature importance
from sklearn.ensemble import ExtraTreesRegressor
model= ExtraTreesRegressor()
model.fit(X,y)
print(model.feature_importances_) #[0.38309998 0.03824574 0.00091369 0.07669352 0.23106969 0.010351710.12801496 0.13161071]
                                  #[Present_price- having highest importance followed by diesel price]

# Plot graph of feature importances for better visualization
feat_importances= pd.Series(model.feature_importances_, index=X.columns)
feat_importances.nlargest(5).plot(kind='barh')
plt.show()

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test= train_test_split(X,y,test_size=0.2)
print(X_train.shape)
print(X_test.shape)

from sklearn.ensemble import RandomForestRegressor

# Hyperparameters
import numpy as np

# Number of trees in RandomForest
n_estimators= [int(x) for x in np.linspace(start=100, stop=1200, num=12)]
# Number of features to be consider in every split
max_features= ['auto','sqrt']
# Maximun no of levels in the tree
max_depth= [int(x) for x in np.linspace(5, 30, num=6)]
#max_depth.append(None)
# Minimum no of samples required to split a node
min_samples_split= [2,5,10,15,100]
# Minimum no of samples required in each leaf node
min_samples_leaf= [1,2,5,10]

from sklearn.model_selection import RandomizedSearchCV

# Create the random grid
random_grid= {'n_estimators': n_estimators,
              'max_features': max_features,
              'max_depth': max_depth,
              'min_samples_split': min_samples_split,
              'min_samples_leaf': min_samples_leaf}
print(random_grid)

# Creating RandomForestRegressor
rf= RandomForestRegressor()
print(rf)
rf_random= RandomizedSearchCV(estimator=rf, param_distributions=random_grid,scoring='neg_mean_squared_error',n_iter=10, cv=5,verbose=2, random_state=42,n_jobs=1)
rf_random.fit(X_train,y_train)
#doing prediction
predictions= rf_random.predict(X_test)
print(predictions)
sns.displot(y_test-predictions)
plt.show()

plt.scatter(y_test,predictions)
plt.show()
