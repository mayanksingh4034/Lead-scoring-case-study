#!/usr/bin/env python
# coding: utf-8

# In[1]:


#importing all the libraries.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
from sklearn.preprocessing import StandardScaler


# In[2]:


# reading the data
data = pd.read_csv("leads.csv")
data.head()


# In[3]:


#checking the shape of the data
data.shape


# In[4]:


data.info()


# In[5]:


#checking the statical info of the data.
data.describe()


# In[6]:


#checking for duplicates.
sum(data.duplicated(subset = 'Prospect ID')) == 0


# In[7]:


#checking for duplicates.
sum(data.duplicated(subset = 'Lead Number')) == 0


# In[8]:


#dropping Lead Number and Prospect ID because they have all unique values.

data.drop(['Prospect ID', 'Lead Number'], 1, inplace = True)


# In[9]:


#Converting 'Select' values to NaN.

data = data.replace('Select', np.nan)


# In[10]:


#checking null values in rows.

data.isnull().sum()


# In[11]:


#checking the percentage of null values in every column.
round(100*(data.isnull().sum()/len(data.index)), 2)


# In[12]:


#dropping columns with more than 45% missing values.
cols=data.columns
for i in cols:
    if((100*(data[i].isnull().sum()/len(data.index))) >= 45):
        data.drop(i, 1, inplace = True)


# In[13]:


#checking null values percentage.
round(100*(data.isnull().sum()/len(data.index)), 2)


# In[14]:


#checking value counts of the Country column.
data['Country'].value_counts(dropna=False)


# In[15]:


#plotting spread of the Country columnn. 
plt.figure(figsize=(15,5))
s1=sns.countplot(data.Country, hue=data.Converted)
s1.set_xticklabels(s1.get_xticklabels(),rotation=90)
plt.show()


# In[16]:


# Since India is the most common occurence among all the non-missing values we can impute all missing values with India.
data['Country'] = data['Country'].replace(np.nan,'India')


# In[17]:


#plotting spread of Country columnn after replacing NaN values.
plt.figure(figsize=(15,5))
s1=sns.countplot(data.Country, hue=data.Converted)
s1.set_xticklabels(s1.get_xticklabels(),rotation=90)
plt.show()


# In[18]:


#creating the list of columns which will be droppped.
cols_to_drop=['Country']


# In[19]:


#checking value counts of the "City" column.
data['City'].value_counts(dropna=False)


# In[20]:


data['City'] = data['City'].replace(np.nan,'Mumbai')


# In[21]:


#plotting the spread of City columnn after replacing NaN values.
plt.figure(figsize=(10,5))
s1=sns.countplot(data.City, hue=data.Converted)
s1.set_xticklabels(s1.get_xticklabels(),rotation=90)
plt.show()


# In[22]:


#checking value counts of Specialization column.
data['Specialization'].value_counts(dropna=False)


# In[23]:


# Lead may not have mentioned specialization because it was not in the list or maybe they are a students 
# and don't have a specialization yet. So we will replace NaN values with 'Not Specified'
data['Specialization'] = data['Specialization'].replace(np.nan, 'Not Specified')


# In[24]:


#plotting spread of Specialization columnn. 
plt.figure(figsize=(15,5))
s1=sns.countplot(data.Specialization, hue=data.Converted)
s1.set_xticklabels(s1.get_xticklabels(),rotation=90)
plt.show()


# In[25]:


#combining Management Specializations because they show similar trends.
data['Specialization'] = data['Specialization'].replace(['Finance Management','Human Resource Management',
                                                           'Marketing Management','Operations Management',
                                                           'IT Projects Management','Supply Chain Management',
                                                    'Healthcare Management','Hospitality Management',
                                                           'Retail Management'] ,'Management_Specializations')


# In[26]:


#visualizing count of Variable based on Converted value.
plt.figure(figsize=(15,5))
s1=sns.countplot(data.Specialization, hue=data.Converted)
s1.set_xticklabels(s1.get_xticklabels(),rotation=90)
plt.show()


# In[27]:


#What is your current occupation.
data['What is your current occupation'].value_counts(dropna=False)


# In[28]:


#imputing Nan values with mode "Unemployed"
data['What is your current occupation'] = data['What is your current occupation'].replace(np.nan, 'Unemployed')


# In[29]:


#checking count of values.
data['What is your current occupation'].value_counts(dropna=False)


# In[30]:


#visualizing count of Variable based on Converted value.
s1=sns.countplot(data['What is your current occupation'], hue=data.Converted)
s1.set_xticklabels(s1.get_xticklabels(),rotation=90)
plt.show()


# In[31]:


#checking value counts.
data['What matters most to you in choosing a course'].value_counts(dropna=False)


# In[32]:


#replacing the Nan values with Mode "Better Career Prospects".
data['What matters most to you in choosing a course'] = data['What matters most to you in choosing a course'].replace(np.nan,'Better Career Prospects')


# In[33]:


#visualizing the count of Variable based on Converted value.
s1=sns.countplot(data['What matters most to you in choosing a course'], hue=data.Converted)
s1.set_xticklabels(s1.get_xticklabels(),rotation=90)
plt.show()


# In[34]:


#checking value counts of the variable.
data['What matters most to you in choosing a course'].value_counts(dropna=False)


# In[35]:


#we have another Column that is worth Dropping. So we Append to the cols_to_drop List.
cols_to_drop.append('What matters most to you in choosing a course')
cols_to_drop


# In[36]:


#checking value counts of Tag variable.
data['Tags'].value_counts(dropna=False)


# In[37]:


#replacing Nan values with "Not Specified".
data['Tags'] = data['Tags'].replace(np.nan,'Not Specified')


# In[38]:


#visualizing count of Variable based on Converted value.
plt.figure(figsize=(15,5))
s1=sns.countplot(data['Tags'], hue=data.Converted)
s1.set_xticklabels(s1.get_xticklabels(),rotation=90)
plt.show()


# In[39]:


#replacing tags with low frequency with "Other Tags".
data['Tags'] = data['Tags'].replace(['In confusion whether part time or DLP', 'in touch with EINS','Diploma holder (Not Eligible)',
                                     'Approached upfront','Graduation in progress','number not provided', 'opp hangup','Still Thinking',
                                    'Lost to Others','Shall take in the next coming month','Lateral student','Interested in Next batch',
                                    'Recognition issue (DEC approval)','Want to take admission but has financial problems',
                                    'University not recognized'], 'Other_Tags')

data['Tags'] = data['Tags'].replace(['switched off',
                                      'Already a student',
                                       'Not doing further education',
                                       'invalid number',
                                       'wrong number given',
                                       'Interested  in full time MBA'] , 'Other_Tags')


# In[40]:


#checking percentage of the missing values.
round(100*(data.isnull().sum()/len(data.index)), 2)


# In[41]:


#checking value counts of Lead Source column.
data['Lead Source'].value_counts(dropna=False)


# In[42]:


#replacing Nan Values and combining low frequency values.
data['Lead Source'] = data['Lead Source'].replace(np.nan,'Others')
data['Lead Source'] = data['Lead Source'].replace('google','Google')
data['Lead Source'] = data['Lead Source'].replace('Facebook','Social Media')
data['Lead Source'] = data['Lead Source'].replace(['bing','Click2call','Press_Release',
                                                     'youtubechannel','welearnblog_Home',
                                                     'WeLearn','blog','Pay per Click Ads',
                                                    'testone','NC_EDM'] ,'Others')       


# In[43]:


#visualizing count of Variable based on Converted value.
plt.figure(figsize=(15,5))
s1=sns.countplot(data['Lead Source'], hue=data.Converted)
s1.set_xticklabels(s1.get_xticklabels(),rotation=90)
plt.show()


# In[44]:


# Last Activity.
data['Last Activity'].value_counts(dropna=False)


# In[45]:


#replacing the Nan Values and combining low frequency values.
data['Last Activity'] = data['Last Activity'].replace(np.nan,'Others')
data['Last Activity'] = data['Last Activity'].replace(['Unreachable','Unsubscribed',
                                                        'Had a Phone Conversation', 
                                                        'Approached upfront',
                                                        'View in browser link Clicked',       
                                                        'Email Marked Spam',                  
                                                        'Email Received','Resubscribed to emails',
                                                         'Visited Booth in Tradeshow'],'Others')


# In[46]:


# Last Activity.
data['Last Activity'].value_counts(dropna=False)


# In[47]:


#Checking the Null Values in All Columns:
round(100*(data.isnull().sum()/len(data.index)), 2)


# In[48]:


#Drop all the rows which have Nan Values. Since the number of Dropped rows is less than 2%, it will not affect the model.
data = data.dropna()


# In[49]:


#Checking percentage of Null Values in All Columns.
round(100*(data.isnull().sum()/len(data.index)), 2)


# In[50]:


#Lead Origin.
data['Lead Origin'].value_counts(dropna=False)


# In[51]:


#visualizing the count of Variable based on Converted value.
plt.figure(figsize=(8,5))
s1=sns.countplot(data['Lead Origin'], hue=data.Converted)
s1.set_xticklabels(s1.get_xticklabels(),rotation=90)
plt.show()


# In[52]:


#Do Not Email & Do Not Call.
#visualizing the count of Variable based on Converted value.
plt.figure(figsize=(15,5))
ax1=plt.subplot(1, 2, 1)
ax1=sns.countplot(data['Do Not Call'], hue=data.Converted)
ax1.set_xticklabels(ax1.get_xticklabels(),rotation=90)
ax2=plt.subplot(1, 2, 2)
ax2=sns.countplot(data['Do Not Email'], hue=data.Converted)
ax2.set_xticklabels(ax2.get_xticklabels(),rotation=90)
plt.show()


# In[53]:


#checking the value counts for Do Not Call.
data['Do Not Call'].value_counts(dropna=False)


# In[54]:


#checking the value counts for Do Not Email.
data['Do Not Email'].value_counts(dropna=False)


# In[55]:


cols_to_drop.append('Do Not Call')
cols_to_drop


# In[56]:


data.Search.value_counts(dropna=False)


# In[57]:


data.Magazine.value_counts(dropna=False)


# In[58]:


data['Newspaper Article'].value_counts(dropna=False)


# In[59]:


data['X Education Forums'].value_counts(dropna=False)


# In[60]:


data['Newspaper'].value_counts(dropna=False)


# In[61]:


data['Digital Advertisement'].value_counts(dropna=False)


# In[62]:


data['Through Recommendations'].value_counts(dropna=False)


# In[63]:


data['Receive More Updates About Our Courses'].value_counts(dropna=False)


# In[64]:


data['Update me on Supply Chain Content'].value_counts(dropna=False)


# In[65]:


data['Get updates on DM Content'].value_counts(dropna=False)


# In[66]:


data['I agree to pay the amount through cheque'].value_counts(dropna=False)


# In[67]:


data['A free copy of Mastering The Interview'].value_counts(dropna=False)


# In[68]:


#adding imbalanced columns to the list of columns to be dropped

cols_to_drop.extend(['Search','Magazine','Newspaper Article','X Education Forums','Newspaper',
                 'Digital Advertisement','Through Recommendations','Receive More Updates About Our Courses',
                 'Update me on Supply Chain Content',
                 'Get updates on DM Content','I agree to pay the amount through cheque'])


# In[69]:


#checking value counts of last Notable Activity
data['Last Notable Activity'].value_counts()


# In[70]:


#clubbing the lower frequency values.
data['Last Notable Activity'] = data['Last Notable Activity'].replace(['Had a Phone Conversation',
                                                                       'Email Marked Spam',
                                                                         'Unreachable',
                                                                         'Unsubscribed',
                                                                         'Email Bounced',                                                                    
                                                                       'Resubscribed to emails',
                                                                       'View in browser link Clicked',
                                                                       'Approached upfront', 
                                                                       'Form Submitted on Website', 
                                                                       'Email Received'],'Other_Notable_activity')


# In[71]:


#visualizing the count of Variable based on Converted value.
plt.figure(figsize = (14,5))
ax1=sns.countplot(x = "Last Notable Activity", hue = "Converted", data = data)
ax1.set_xticklabels(ax1.get_xticklabels(),rotation=90)
plt.show()


# In[72]:


#checking the value counts for variable.
data['Last Notable Activity'].value_counts()


# In[73]:


#list of the columns to be dropped.
cols_to_drop


# In[74]:


#dropping the columns.
data = data.drop(cols_to_drop,1)
data.info()


# In[75]:


#Checking the percentage of Data that has Converted Values 1.
Converted = (sum(data['Converted'])/len(data['Converted'].index))*100
Converted


# In[76]:


#Checking the correlations of numeric values.
# figure size
plt.figure(figsize=(10,8))
# heatmap
sns.heatmap(data.corr(), cmap="YlGnBu", annot=True)
plt.show()


# In[77]:


#Total Visits
#visualizing the spread of variable.
plt.figure(figsize=(6,4))
sns.boxplot(y=data['TotalVisits'])
plt.show()


# In[78]:


#checking the percentile values for "Total Visits".
data['TotalVisits'].describe(percentiles=[0.05,.25, .5, .75, .90, .95, .99])


# In[79]:


#Outlier Treatment, Removing the top & the bottom 1% of the Column Outlier values.
Q3 = data.TotalVisits.quantile(0.99)
data = data[(data.TotalVisits <= Q3)]
Q1 = data.TotalVisits.quantile(0.01)
data = data[(data.TotalVisits >= Q1)]
sns.boxplot(y=data['TotalVisits'])
plt.show()


# In[80]:


data.shape


# In[81]:


#checking percentiles for "Total Time Spent on Website".
data['Total Time Spent on Website'].describe(percentiles=[0.05,.25, .5, .75, .90, .95, .99])


# In[82]:


#visualizing spread of the numeric variable.
plt.figure(figsize=(6,4))
sns.boxplot(y=data['Total Time Spent on Website'])
plt.show()


# In[83]:


#checking the spread of "Page Views Per Visit"
data['Page Views Per Visit'].describe()


# In[84]:


#visualizing the spread of numeric variable.
plt.figure(figsize=(6,4))
sns.boxplot(y=data['Page Views Per Visit'])
plt.show()


# In[85]:


#Outlier Treatment, Removing the top & bottom 1% .
Q3 = data['Page Views Per Visit'].quantile(0.99)
data = data[data['Page Views Per Visit'] <= Q3]
Q1 = data['Page Views Per Visit'].quantile(0.01)
data = data[data['Page Views Per Visit'] >= Q1]
sns.boxplot(y=data['Page Views Per Visit'])
plt.show()


# In[86]:


data.shape


# In[87]:


#checking the Spread of "Total Visits" vs Converted variable.
sns.boxplot(y = 'TotalVisits', x = 'Converted', data = data)
plt.show()


# In[88]:


#checking the Spread of "Total Time Spent on Website" vs Converted variable.
sns.boxplot(x=data.Converted, y=data['Total Time Spent on Website'])
plt.show()


# In[89]:


#checking the Spread of "Page Views Per Visit" vs Converted variable
sns.boxplot(x=data.Converted,y=data['Page Views Per Visit'])
plt.show()


# In[90]:


#checking missing values in the leftover columns.
round(100*(data.isnull().sum()/len(data.index)),2)


# In[91]:


#getting the list of categorical columns.
cat_cols= data.select_dtypes(include=['object']).columns
cat_cols


# In[92]:


# List of variables for map.
varlist =  ['A free copy of Mastering The Interview','Do Not Email']
# Defining the map function.
def binary_map(x):
    return x.map({'Yes': 1, "No": 0})
# Applying the function to the housing list.
data[varlist] = data[varlist].apply(binary_map)


# In[93]:


#getting the dummies and dropping the first column and adding the results to the master dataframe.
dummy = pd.get_dummies(data[['Lead Origin','What is your current occupation',
                             'City']], drop_first=True)
data = pd.concat([data,dummy],1)


# In[94]:


dummy = pd.get_dummies(data['Specialization'], prefix  = 'Specialization')
dummy = dummy.drop(['Specialization_Not Specified'], 1)
data = pd.concat([data, dummy], axis = 1)


# In[95]:


dummy = pd.get_dummies(data['Lead Source'], prefix  = 'Lead Source')
dummy = dummy.drop(['Lead Source_Others'], 1)
data = pd.concat([data, dummy], axis = 1)


# In[96]:


dummy = pd.get_dummies(data['Last Activity'], prefix  = 'Last Activity')
dummy = dummy.drop(['Last Activity_Others'], 1)
data = pd.concat([data, dummy], axis = 1)


# In[97]:


dummy = pd.get_dummies(data['Last Notable Activity'], prefix  = 'Last Notable Activity')
dummy = dummy.drop(['Last Notable Activity_Other_Notable_activity'], 1)
data = pd.concat([data, dummy], axis = 1)


# In[98]:


dummy = pd.get_dummies(data['Tags'], prefix  = 'Tags')
dummy = dummy.drop(['Tags_Not Specified'], 1)
data = pd.concat([data, dummy], axis = 1)


# In[99]:


#dropping the original columns after creating the dummy variable.
data.drop(cat_cols,1,inplace = True)


# In[100]:


data.head()


# In[101]:


from sklearn.model_selection import train_test_split
# Putting response variable to y
y = data['Converted']
y.head()
X=data.drop('Converted', axis=1)


# In[102]:


# Splitting the data for train and test.
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, test_size=0.3, random_state=100)


# In[103]:


X_train.info()


# In[104]:


#scaling numeric columns
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
num_cols=X_train.select_dtypes(include=['float64', 'int64']).columns
X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
X_train.head()


# In[105]:


from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
from sklearn.feature_selection import RFE
rfe = RFE(logreg, n_features_to_select=15) 
rfe = rfe.fit(X_train, y_train)


# In[106]:


import statsmodels.api as sm


# In[107]:


data.head()


# In[108]:


rfe.support_


# In[109]:


list(zip(X_train.columns, rfe.support_, rfe.ranking_))


# In[110]:


#list of RFE supported columns.
col = X_train.columns[rfe.support_]
col


# In[111]:


X_train.columns[~rfe.support_]


# In[112]:


#BUILDING MODEL no1.
X_train_sm = sm.add_constant(X_train[col])
logm1 = sm.GLM(y_train,X_train_sm, family = sm.families.Binomial())
res = logm1.fit()
res.summary()


# In[113]:


#dropping the columns with high p-value.
col = col.drop('Lead Source_Referral Sites',1)


# In[114]:


#BUILDING MODEL no2.
X_train_sm = sm.add_constant(X_train[col])
logm2 = sm.GLM(y_train,X_train_sm, family = sm.families.Binomial())
res = logm2.fit()
res.summary()


# In[115]:


# Checking for the VIF values of the feature variables. 
from statsmodels.stats.outliers_influence import variance_inflation_factor


# In[116]:


# Creating a dataframe that will contain the names of all the feature variables and their respective VIFs
vif = pd.DataFrame()
vif['Features'] = X_train[col].columns
vif['VIF'] = [variance_inflation_factor(X_train[col].values, i) for i in range(X_train[col].shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif


# In[117]:


#dropping the variable with high VIF.
col = col.drop('Last Notable Activity_SMS Sent',1)


# In[118]:


#BUILDING MODEL no3.
X_train_sm = sm.add_constant(X_train[col])
logm3 = sm.GLM(y_train,X_train_sm, family = sm.families.Binomial())
res = logm3.fit()
res.summary()


# In[119]:


# Creating a dataframe that will contain the names of all the feature variables and their respective VIFs
vif = pd.DataFrame()
vif['Features'] = X_train[col].columns
vif['VIF'] = [variance_inflation_factor(X_train[col].values, i) for i in range(X_train[col].shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif


# In[120]:


# Getting the Predicted values of the train set.
y_train_pred = res.predict(X_train_sm)
y_train_pred[:10]


# In[121]:


y_train_pred = y_train_pred.values.reshape(-1)
y_train_pred[:10]


# In[122]:


y_train_pred_final = pd.DataFrame({'Converted':y_train.values, 'Converted_prob':y_train_pred})
y_train_pred_final['Prospect ID'] = y_train.index
y_train_pred_final.head()


# In[123]:


y_train_pred_final['Predicted'] = y_train_pred_final.Converted_prob.map(lambda x: 1 if x > 0.5 else 0)
# checking the head.
y_train_pred_final.head()


# In[124]:


from sklearn import metrics
# Confusion matrix 
confusion = metrics.confusion_matrix(y_train_pred_final.Converted, y_train_pred_final.Predicted )
print(confusion)


# In[125]:


# Let's check the overall accuracy.
print(metrics.accuracy_score(y_train_pred_final.Converted, y_train_pred_final.Predicted))


# In[126]:


TP = confusion[1,1] # true positive 
TN = confusion[0,0] # true negatives
FP = confusion[0,1] # false positives
FN = confusion[1,0] # false negatives


# In[127]:


# Let's see the sensitivity of our logistic regression model.
TP / float(TP+FN)


# In[128]:


# Let us calculate the specificity.
TN / float(TN+FP)


# In[129]:


# Calculate False Postive Rate, predicting conversion when customer does not have convert
print(FP/ float(TN+FP))


# In[130]:


# positive predictive value.
print (TP / float(TP+FP))


# In[131]:


# Negative predictive value.
print (TN / float(TN+ FN))


# In[132]:


def draw_roc( actual, probs ):
    fpr, tpr, thresholds = metrics.roc_curve( actual, probs,
                                              drop_intermediate = False )
    auc_score = metrics.roc_auc_score( actual, probs )
    plt.figure(figsize=(5, 5))
    plt.plot( fpr, tpr, label='ROC curve (area = %0.2f)' % auc_score )
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate or [1 - True Negative Rate]')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()

    return None


# In[133]:


fpr, tpr, thresholds = metrics.roc_curve( y_train_pred_final.Converted, y_train_pred_final.Converted_prob, drop_intermediate = False )


# In[134]:


draw_roc(y_train_pred_final.Converted, y_train_pred_final.Converted_prob)


# In[135]:


# Let's create columns with different probability cutoffs.
numbers = [float(x)/10 for x in range(10)]
for i in numbers:
    y_train_pred_final[i]= y_train_pred_final.Converted_prob.map(lambda x: 1 if x > i else 0)
y_train_pred_final.head()


# In[136]:


# Now let's calculate the accuracy sensitivity and specificity for various probability cutoffs.
cutoff_df = pd.DataFrame( columns = ['prob','accuracy','sensi','speci'])
from sklearn.metrics import confusion_matrix
num = [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
for i in num:
    cm1 = metrics.confusion_matrix(y_train_pred_final.Converted, y_train_pred_final[i] )
    total1=sum(sum(cm1))
    accuracy = (cm1[0,0]+cm1[1,1])/total1
    
    speci = cm1[0,0]/(cm1[0,0]+cm1[0,1])
    sensi = cm1[1,1]/(cm1[1,0]+cm1[1,1])
    cutoff_df.loc[i] =[ i ,accuracy,sensi,speci]
print(cutoff_df)


# In[137]:


# Let's plot accuracy sensitivity and specificity for various probabilities.
cutoff_df.plot.line(x='prob', y=['accuracy','sensi','speci'])
plt.show()


# In[138]:


# From the curve above, 0.3 is the optimum point to take it as a cutoff probability.
y_train_pred_final['final_Predicted'] = y_train_pred_final.Converted_prob.map( lambda x: 1 if x > 0.3 else 0)
y_train_pred_final.head()


# In[139]:


y_train_pred_final['Lead_Score'] = y_train_pred_final.Converted_prob.map( lambda x: round(x*100))
y_train_pred_final[['Converted','Converted_prob','Prospect ID','final_Predicted','Lead_Score']].head()


# In[140]:


# Let's check the overall accuracy.
metrics.accuracy_score(y_train_pred_final.Converted, y_train_pred_final.final_Predicted)


# In[141]:


confusion2 = metrics.confusion_matrix(y_train_pred_final.Converted, y_train_pred_final.final_Predicted )
confusion2


# In[142]:


TP = confusion2[1,1] 
TN = confusion2[0,0] 
FP = confusion2[0,1] 
FN = confusion2[1,0] 


# In[143]:


# checking the sensitivity of our logistic regression model.
TP / float(TP+FN)


# In[144]:


#calculating specificity
TN / float(TN+FP)


# In[145]:


# Calculating False Postive Rate, predicting conversion when customer does not have convert.
print(FP/ float(TN+FP))


# In[146]:


# Positive predictive value.
print (TP / float(TP+FP))


# In[147]:


# Negative predictive value.
print (TN / float(TN+ FN))


# In[148]:


#Looking at the confusion matrix.
confusion = metrics.confusion_matrix(y_train_pred_final.Converted, y_train_pred_final.final_Predicted )
confusion


# In[152]:


# Precision.
TP / TP + FP

confusion[1,1]/(confusion[0,1]+confusion[1,1])


# In[153]:


# Recall.
TP / TP + FN

confusion[1,1]/(confusion[1,0]+confusion[1,1])


# In[154]:


from sklearn.metrics import precision_score, recall_score


# In[155]:


precision_score(y_train_pred_final.Converted , y_train_pred_final.final_Predicted)


# In[156]:


recall_score(y_train_pred_final.Converted, y_train_pred_final.final_Predicted)


# In[157]:


from sklearn.metrics import precision_recall_curve


# In[158]:


y_train_pred_final.Converted, y_train_pred_final.final_Predicted
p, r, thresholds = precision_recall_curve(y_train_pred_final.Converted, y_train_pred_final.Converted_prob)


# In[159]:


plt.plot(thresholds, p[:-1], "g-")
plt.plot(thresholds, r[:-1], "r-")
plt.show()


# In[160]:


#scaling test set.
num_cols=X_test.select_dtypes(include=['float64', 'int64']).columns

X_test[num_cols] = scaler.fit_transform(X_test[num_cols])

X_test.head()


# In[161]:


X_test = X_test[col]
X_test.head()


# In[162]:


X_test_sm = sm.add_constant(X_test)


# In[163]:


y_test_pred = res.predict(X_test_sm)


# In[164]:


y_test_pred[:10]


# In[165]:


# Converting y_pred to a dataframe which is already an array.
y_pred_1 = pd.DataFrame(y_test_pred)


# In[166]:


# Let's see the head.
y_pred_1.head()


# In[167]:


# Converting y_test to dataframe.
y_test_df = pd.DataFrame(y_test)


# In[168]:


# Putting CustID to index.
y_test_df['Prospect ID'] = y_test_df.index


# In[169]:


# Removing index for both dataframes to append them side by side.
y_pred_1.reset_index(drop=True, inplace=True)
y_test_df.reset_index(drop=True, inplace=True)


# In[170]:


# Appending y_test_df and y_pred_1.
y_pred_final = pd.concat([y_test_df, y_pred_1],axis=1)


# In[171]:


y_pred_final.head()


# In[172]:


# Renaming the column 
y_pred_final= y_pred_final.rename(columns={ 0 : 'Converted_prob'})


# In[173]:


y_pred_final.head()


# In[174]:


# Rearranging the columns.
y_pred_final = y_pred_final[['Prospect ID','Converted','Converted_prob']]
y_pred_final['Lead_Score'] = y_pred_final.Converted_prob.map( lambda x: round(x*100))


# In[175]:


# Let's see the head of y_pred_final.
y_pred_final.head()


# In[176]:


y_pred_final['final_Predicted'] = y_pred_final.Converted_prob.map(lambda x: 1 if x > 0.3 else 0)


# In[177]:


y_pred_final.head()


# In[178]:


# Let's check the overall accuracy.
metrics.accuracy_score(y_pred_final.Converted, y_pred_final.final_Predicted)


# In[179]:


confusion2 = metrics.confusion_matrix(y_pred_final.Converted, y_pred_final.final_Predicted )
confusion2


# In[180]:


TP = confusion2[1,1] # true positive 
TN = confusion2[0,0] # true negatives
FP = confusion2[0,1] # false positives
FN = confusion2[1,0] # false negatives


# In[181]:


# Let's see the sensitivity of our logistic regression model.
TP / float(TP+FN)


# In[182]:


# Let us calculate specificity.
TN / float(TN+FP)


# In[183]:


precision_score(y_pred_final.Converted , y_pred_final.final_Predicted)


# In[184]:


recall_score(y_pred_final.Converted, y_pred_final.final_Predicted)


# In[ ]:





# In[ ]:





# In[ ]:




