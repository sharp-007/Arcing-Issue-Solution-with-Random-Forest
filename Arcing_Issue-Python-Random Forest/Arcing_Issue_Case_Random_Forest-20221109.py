#!/usr/bin/env python
# coding: utf-8

# # Arcing Issue

# In[163]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ## import data

# In[164]:


dataset=r'C:\Users\pj007\exercises\JMP_Case\dataset\Arcing Issue_195.csv'
data=pd.read_csv(dataset)
data


# ### train_set

# In[165]:


train_data=data.iloc[0:32,2:-1]
train_data


# In[166]:


train_labels=data.iloc[0:32,1]
train_labels


# ### test_set

# In[167]:


test_data=data.iloc[32:,2:-1]
test_data


# In[168]:


test_labels=data.iloc[32:,1]
test_labels


# ## export standard dataset

# In[169]:


from sklearn import datasets, preprocessing
#from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
scaler=preprocessing.StandardScaler().fit(train_data)
train_data=scaler.transform(train_data)
test_data=scaler.transform(test_data)


# In[170]:


Arcing_Issue_standard1=data.iloc[:,0:2]
#Arcing_Issue_standard1


# In[171]:


train_data_df=pd.DataFrame(train_data)
#train_data_df


# In[172]:


test_data_df=pd.DataFrame(test_data)
#test_data_df


# In[173]:


Arcing_Issue_standard2=pd.concat([train_data_df,test_data_df]).reset_index(drop=True)
#Arcing_Issue_standard2


# In[174]:


Arcing_Issue_standard3=data.iloc[:,-1:]
#Arcing_Issue_standard3


# In[175]:


cols=list(data.columns)
#cols


# In[176]:


Arcing_Issue_standard=pd.concat([Arcing_Issue_standard1,Arcing_Issue_standard2,Arcing_Issue_standard3],axis=1)
#Arcing_Issue_standard


# In[177]:


Arcing_Issue_standard.columns=cols
Arcing_Issue_standard


# In[178]:


Arcing_Issue_standard.to_csv('Arcing_Issue_standard_195.csv')


# ## train data with RF

# In[179]:


from sklearn.ensemble import RandomForestClassifier

rnd_clf = RandomForestClassifier(n_estimators=100, min_samples_split=10, min_samples_leaf=3, max_features=143, n_jobs=-1, random_state=42)
rnd_clf.fit(train_data, train_labels)
test_pred = rnd_clf.predict(test_data)


# ### accuracy

# In[180]:


from sklearn.metrics import accuracy_score
accuracy_score(test_labels, test_pred)


# ### feature importances

# In[181]:


importances_0=rnd_clf.feature_importances_


# In[182]:


cols_0=data.columns[2:-1]
df_importances_0=pd.DataFrame({'Variable':cols_0,'Importance':importances_0})
df_importances_0=df_importances_0.sort_values(by='Importance',ascending=False).head(20)
df_importances_0


# In[183]:


df_importances_0.head(15).plot(kind='bar',x='Variable', y='Importance', rot=45)


# In[184]:


Arcing_Issue_standard.loc[:, df_importances_0.iloc[:15,0]].head()


# ## GridSearchCV

# In[185]:


'''
RandomForestClassifier(
    n_estimators='warn',
    criterion='gini',
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    min_weight_fraction_leaf=0.0,
    max_features='auto',
    max_leaf_nodes=None,
    min_impurity_decrease=0.0,
    min_impurity_split=None,
    bootstrap=True,
    oob_score=False,
    n_jobs=None,
    random_state=None,
    verbose=0,
    warm_start=False,
    class_weight=None,
)
'''


# rnd_clf = RandomForestClassifier(n_estimators=100, min_samples_split=10, min_samples_leaf=3, max_features=143, n_jobs=-1, random_state=42)
# 
# 0.75

# In[186]:


from sklearn.ensemble import RandomForestClassifier

rnd_clf_grid = RandomForestClassifier(n_jobs=-1, random_state=42)


# In[187]:


from sklearn.model_selection import GridSearchCV

param_distributions = {"n_estimators": [100,200], "max_features":[120,130],"min_samples_split": [3,5],"min_samples_leaf": [1,3,5,7]}
grid_search_cv = GridSearchCV(rnd_clf_grid, param_distributions, cv=4, verbose=2, scoring='accuracy') # 4-fold
grid_search_cv.fit(train_data, train_labels)


# In[188]:


grid_search_cv_best=grid_search_cv.best_estimator_
grid_search_cv_best


# In[189]:


grid_search_cv.best_score_


# In[190]:


grid_search_cv_best.fit(train_data, train_labels)


# In[191]:


train_pred = grid_search_cv_best.predict(train_data)
accuracy_score(train_labels, train_pred)


# ## use RF model to predict labels of test set

# In[192]:


test_pred = grid_search_cv_best.predict(test_data)
test_pred


# ## model evaluation

# ### classification_report

# In[193]:


df=pd.DataFrame({'test_label':test_labels,'test_pred':test_pred})
df


# In[194]:


import sklearn.metrics as sm
cr = sm.classification_report(test_labels,test_pred)
print(cr)


# ### confusion matrix

# In[195]:


from sklearn.metrics import confusion_matrix
confusion_matrix(test_labels,test_pred)


# In[196]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import confusion_matrix, recall_score, classification_report


# In[197]:


def plot_confusion_matrix(cm, classes,normalize=False,title='Confusion matrix',cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')
    print(cm)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


# In[198]:


import itertools
cnf_matrix = confusion_matrix(test_labels,test_pred) #计算混淆矩阵
class_names = [0,1]
plt.figure()
plot_confusion_matrix(cnf_matrix, classes = class_names, title = 'Confusion matrix')  #绘制混淆矩阵
np.set_printoptions(precision=2)
print('Accuracy:', (cnf_matrix[1,1]+cnf_matrix[0,0])/(cnf_matrix[1,1]+cnf_matrix[0,1]+cnf_matrix[0,0]+cnf_matrix[1,0]))
print('Recall:', cnf_matrix[1,1]/(cnf_matrix[1,1]+cnf_matrix[1,0]))     
print('Precision:', cnf_matrix[1,1]/(cnf_matrix[1,1]+cnf_matrix[0,1]))  
print('Specificity:', cnf_matrix[0,0]/(cnf_matrix[0,1]+cnf_matrix[0,0]))
plt.show()


# ### ROC curve

# In[199]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn.metrics as metrics

def plot_ROC(labels,preds):
    """
    Args:
        labels : ground truth
        preds : model prediction
        savepath : save path 
    """
    fpr1, tpr1, threshold1 = metrics.roc_curve(labels, preds)  #计算真正率和假正率
    
    roc_auc1 = metrics.auc(fpr1, tpr1)  #计算auc的值，auc就是曲线包围的面积，越大越好
    plt.figure()
    lw = 2
    plt.figure(figsize=(10, 10))
    plt.plot(fpr1, tpr1, color='darkorange',
            lw=lw, label='AUC = %0.2f' % roc_auc1)  #假正率为横坐标，真正率为纵坐标做曲线
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('1 - Specificity')
    plt.ylabel('Sensitivity')
    # plt.title('ROCs for Densenet')
    plt.legend(loc="lower right")
    # plt.show()


# In[200]:


test_labels_1=np.array(test_labels)
test_labels_1=[1 if x=='Y' else 0 for x in test_labels_1 ]
test_labels_1


# In[201]:


test_pred_1=np.array(test_pred)
test_pred_1=[1 if x=='Y' else 0 for x in test_pred_1 ]
test_pred_1


# In[202]:


plot_ROC(test_labels_1,test_pred_1)


# ### feature importances

# In[203]:


importances=grid_search_cv_best.feature_importances_


# In[204]:


cols=data.columns[2:-1]
df_importances_1=pd.DataFrame({'Variable':cols,'Importance':importances})
df_importances_1=df_importances_1.sort_values(by='Importance',ascending=False).head(20)
df_importances_1


# In[205]:


df_importances_1.head(20).plot(kind='bar',x='Variable', y='Importance', rot=45)


# In[206]:


Arcing_Issue_standard.loc[:, df_importances_1.iloc[:20,0]].head()


# ## save model using joblib

# In[207]:


my_model =grid_search_cv_best
my_model


# In[208]:


#save my_model
import joblib
#from sklearn.externals import joblib
joblib.dump(my_model, "random_forest.pkl") 

