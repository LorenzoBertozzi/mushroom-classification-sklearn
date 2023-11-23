
#%%
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn  as sns

#%%
mushrooms = pd.read_csv('input/mushrooms.csv')

# %%
mushrooms.head()

# %%
mushrooms.isnull().sum().sum()

#%%
mushrooms['class'].unique()

#%%
mushrooms.info()

#%%
mushrooms.shape

#%%
sns.histplot(mushrooms['class'])

#%%
X = mushrooms.drop(['class'],axis=1)
y = mushrooms['class']

#%%
X = pd.get_dummies(X)
X.head()

#%%
from sklearn.preprocessing import LabelEncoder

encoder = LabelEncoder()
y = encoder.fit_transform(y)
print(y)

# %%
#COMEÇANDO A TREINAR A TREE
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

# %%
X_train.shape , X_test.shape

#%%
y_train.shape , y_test.shape

# %%
clf = tree.DecisionTreeClassifier()
clf = clf.fit(X_train, y_train)

# %%
print(clf.score(X_test, y_test))

# %%
plt.figure(dpi = 200)
tree.plot_tree(clf, feature_names = mushrooms['cap-shape'], class_names = mushrooms['class'])
plt.show