# To add a new cell, type '#%%'
# To add a new markdown cell, type '#%% [markdown]'
#%% Change working directory from the workspace root to the ipynb file location. Turn this addition off with the DataScience.changeDirOnImportExport setting
# ms-python.python added
import os
try:
	os.chdir(os.path.join(os.getcwd(), '..\\AppData\Local\Temp'))
	print(os.getcwd())
except:
	pass

#%%
import os 
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

#import scikit-learn tools for machine learning


#%%
DATA_DIR = os.path.join(os.getcwd(), 'breast-cancer-wisconsin-data', 'data.csv')


#%%
data = pd.read_csv(DATA_DIR)


#%%
data.head()


#%%
# let's see what columns do we have in dataset
data.columns


#%%
#Extract target feature that is 'diagnosis'
target = data['diagnosis']

#Drop features that we don't need: target, id, Unnamed:32
x = data.drop(columns=['id', 'diagnosis', 'Unnamed: 32'])


#%%
x.shape


#%%
ax = sns.countplot(target, label='Counts')
B, M = target.value_counts()
print('Number of Benign : ', B)
print('Number of Malignant : ', M)


#%%
data_dia = target
data = x
data_n_2 = (data - data.mean())/data.std()

               
               


#%%
def swarm_violin_and_box(start_col, end_col, plot_type, size=8):
    data = pd.concat([target, data_n_2.iloc[:, start_col:end_col]], axis=1)
    data = pd.melt(data, id_vars='diagnosis',
               var_name='features',
               value_name='value')
    plt.figure(figsize=(size, size))
    if plot_type == 'violin':
        sns.violinplot(x='features', y='value', hue='diagnosis', data=data, split=True, inner='quart')
        plt.xticks(rotation=90)
    if plot_type == 'box':
        sns.boxplot(x='features', y='value', hue='diagnosis', data=data)
        plt.xticks(rotation=90)
    if plot_type =='swarm':
        sns.swarmplot(x="features", y="value", hue="diagnosis", data=data)
        plt.xticks(rotation=90)
        
        


#%%
swarm_violin_and_box(0, 10, "box", size=10)

#%% [markdown]
# The medians of Malignant and Benign in all features except fractal_dimension_mean looks separate. So fractal_dimension_mean feature can't distinguish two groups and give good information for classification.

#%%
swarm_violin_and_box(10, 20, 'violin', size=10)

#%% [markdown]
# In this graph we can say same thing about texture_se, smoothness_se, symmetry_se.

#%%
swarm_violin_and_box(20, 31, 'violin')


#%%
swarm_violin_and_box(20, 31, 'box', size=10)


#%%
sns.jointplot(x.loc[:,'concavity_worst'], x.loc[:,'concave points_worst'], kind="regg", color="#ce1414")

#%% [markdown]
# As we see in the graf above 'concavity_worst' and 'concave points_worst' are highly correlated

#%%
sns.set(style='white')
g = sns.PairGrid(x, diag_sharey=False, vars=['radius_worst', 'perimeter_worst', 'area_worst'])
g.map_lower(sns.kdeplot, cmap="Blues_d")
g.map_upper(plt.scatter)
g.map_diag(sns.kdeplot, lw=3)


#%%
swarm_violin_and_box(0, 10, 'swarm', size=10)


#%%
swarm_violin_and_box(10, 20, 'swarm', size=10)


#%%
swarm_violin_and_box(20, 30, 'swarm', size=10)


#%%
f,ax = plt.subplots(figsize=(18, 18))
sns.heatmap(x.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)


#%%
corr_matr = x.corr()
upper = corr_matr.where(np.triu(np.ones(corr_matr.shape), k=1).astype(np.bool))
to_drop = [column for column in upper.columns if any(abs(upper[column])>0.88) ]
to_drop


#%%
drop_list = ['perimeter_mean','radius_mean','compactness_mean','concave points_mean',
              'radius_se','perimeter_se','radius_worst','perimeter_worst','compactness_worst',
              'concave points_worst','compactness_se','concave points_se','texture_worst','area_worst']


#%%
x_1 = x.drop(columns=drop_list)


#%%
x_1.shape


#%%
f,ax = plt.subplots(figsize=(14, 14))
sns.heatmap(x_1.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)


#%%
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score,confusion_matrix
from sklearn.metrics import accuracy_score

x_train, x_test, y_train, y_test = train_test_split(x_1, target, test_size=0.3, random_state=42)


#%%
clf_rf = RandomForestClassifier(random_state=43)
clf_rf.fit(x_train, y_train)
ac = accuracy_score(y_test, clf_rf.predict(x_test))
print('Accuracy: ', ac)
conf_matrix = confusion_matrix(y_test, clf_rf.predict(x_test))
sns.heatmap(conf_matrix, annot=True, fmt="d")


#%%
importances = clf_rf.feature_importances_
std = np.std([tree.feature_importances_ for tree in clf_rf.estimators_], axis=0)
indices = np.argsort(importances)[::-1]


for f in range(x_train.shape[1]):
    print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))

# Plot the feature importances of the forest

plt.figure(1, figsize=(11, 10))
plt.title("Feature importances")
plt.bar(range(x_train.shape[1]), importances[indices],
       color="g", yerr=std[indices], align="center")
plt.xticks(range(x_train.shape[1]), x_train.columns[indices],rotation=90)
plt.xlim([-1, x_train.shape[1]])
plt.show()


#%%


