#!/usr/bin/env python
# coding: utf-8

# ### Importing Libraries

# In[131]:


import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from matplotlib import style
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
import warnings


import datetime

from scipy.stats import chi2_contingency
from scipy.stats import chi


# # Task 1

#    ## Reading Data Set

# In[132]:


service_311 = pd.read_csv('311_Service_Requests_from_2010_to_Present.csv', low_memory = False)
warnings.filterwarnings('ignore' )


# In[133]:


service_311.head()


# In[134]:


service_311.shape


# In[135]:


service_311.info()


# ###### There are a lot of columns in our dataset. However we don't need all of them. So we can drop the columns which have a very large number of null values in it.

# ## Reading Columns

# In[136]:


service_311.columns


# ### Dropping irrelevant columns from datasets

# In[137]:


drop_columns = ['Agency Name','Incident Address','Street Name','Cross Street 1','Cross Street 2','Intersection Street 1',
'Intersection Street 2','Address Type','Park Facility Name','Park Borough','School Name',
'School Number','School Region','School Code','School Phone Number','School Address','School City',
'School State','School Zip','School Not Found','School or Citywide Complaint','Vehicle Type',
'Taxi Company Borough','Taxi Pick Up Location','Bridge Highway Name','Bridge Highway Direction',
'Road Ramp','Bridge Highway Segment','Garage Lot Name','Ferry Direction','Ferry Terminal Name','Landmark',
'X Coordinate (State Plane)','Y Coordinate (State Plane)','Due Date','Resolution Action Updated Date','Community Board','Facility Type',
'Location']


service_311.drop(drop_columns, axis = 1, inplace = True)


# In[138]:


service_311


# In[139]:


service_311.shape


# ##### checking the number of null values in the columns

# In[140]:


service_311.isnull().sum()


# ###  selecting the closed cases only to eliminate the null values

# In[141]:


service_311_2 = service_311[service_311['Status'] == 'Closed']


# In[142]:


service_311_2.isnull().sum()


# ###  since all the cases are of closed cases now we can drop the column Status as every value of its data are same

# In[143]:


service_311_3 = service_311_2.drop(['Status'], axis = 1)


# In[144]:


service_311_3


# In[145]:


service_311_3.shape


# In[146]:


service_311_3 = service_311_3[(service_311_3['Descriptor'].notnull()) & (service_311_3['Latitude'].notnull()) &(service_311_3['Longitude'].notnull())]
service_311_3.info()


# In[147]:


service_311_3.isnull().sum()


# In[148]:


service_311_3 = service_311_3[(service_311_3['Location Type'].notnull()) & (service_311_3['Incident Zip'].notnull()) &(service_311_3['City'].notnull())]
service_311_3.isnull().sum()


# ## All the null values have been removed

# In[149]:


service_311_3.shape


# # Task 2

# ## converting 'Created Date' and 'Closed Date' to datetime datatype

# In[150]:


cols = ['Created Date', 'Closed Date']
for col in cols:
    service_311_3[col] = pd.to_datetime(service_311_3[col],infer_datetime_format=True)


service_311_3['Request_Closing_Time'] = service_311_3[cols[1]] - service_311_3[cols[0]]
    
service_311_3.head(10)


# ## Task 3
# 
# * Visualization
# * Atleast 4 Main Conclusions

# In[151]:


service_311_3.describe()


# In[152]:


service_311_3['Complaint Type'].value_counts()


# In[153]:


service_311_3['Agency'].value_counts()


# In[154]:


service_311_3['Complaint Type'].value_counts().plot(kind = 'bar', figsize = (12,5), title = 'Complaint Type', color = 'c', grid = True)


# ######         As we can see the Blocked Driveway is the Maximum Complaint type followed by Illegal Parking, Noise-Street/Sidewalk, Noise-Commercial

# In[155]:


service_311_3['Descriptor'].value_counts().head(10)


# In[156]:


service_311_3['Descriptor'].value_counts().head(10).plot(kind = 'barh', figsize = (12,5), title = 'Top 10 Descriptor', color = 'b', grid = True)


# #### we can clearly see the Loud Music/Party is the maximum descriptor for the complaints followed by No Access, Posted Parking Sign Violation and Loud Taking.

# In[157]:


service_311_3['Location Type'].value_counts().head(10)


# In[158]:


service_311_3['Location Type'].value_counts().head(10).plot(kind = 'barh',figsize = (12,6), title = 'Top 10 Location', grid = True)


# ###   As we can see the Location Type of 'Street/Sidewalk' is a lot more than any other members of its category

# In[159]:


service_311_3['City'].value_counts().head(10)


# In[160]:


service_311_3['City'].value_counts().head(10).plot(kind = 'barh',figsize = (12,6), title = 'Top 10 Location', grid = True)
plt.xlabel('Complaint Type')


# ####                  As we can see the most complaints are from 'BROOKLYN' followed New York, Bronx, Staten Island in City wise

#  ### Analysing Borough and Complaint Types
# 
#  

# In[161]:


top_six_complaints = service_311_3['Complaint Type'].value_counts()[:6].keys()
top_six_complaints


# In[162]:


borough_complaints = service_311_3.groupby(['Borough', 'Complaint Type']).size().unstack()
borough_complaints = borough_complaints[top_six_complaints]
borough_complaints


# In[163]:


col_number = 2
row_number = 3
fig, axes = plt.subplots(row_number,col_number, figsize=(12,8))

for i, (label,col) in enumerate(borough_complaints.iteritems()):
    ax = axes[int(i/col_number), i%col_number]
    col = col.sort_values(ascending=True)[:15]
    col.plot(kind='barh', ax=ax, grid=True)
    ax.set_title(label)
    
plt.tight_layout()


# # What we have analysed
# 
# ** Blocked Driveway is maximum in QUEENS
# ** Illegal Parking is maximum in BROOKLYN
# ** Noise - Street/Sidewalk is maximum in MANHATTAN
# ** Noise - Commercial is maximum in MANHATTAN
# ** Derelict Vehicle is maximum in QUEENS
# ** Noise - Vehicle is maximum in QUEENS

# In[164]:


top_borough = service_311_3['Borough'].value_counts().keys()

complaint_per_borough = service_311_3.groupby(['Complaint Type', 'Borough']).size().unstack()
complaint_per_borough = complaint_per_borough[top_borough]
complaint_per_borough


# In[165]:


col_number = 2
row_number = 3
fig, axes = plt.subplots(row_number,col_number, figsize=(12,10))

for i, (label,col) in enumerate(complaint_per_borough.iteritems()):
    ax = axes[int(i/col_number), i%col_number]
    col = col.sort_values(ascending=True)[:15]
    col.plot(kind='barh', ax=ax, grid=True)
    ax.set_title(label)
    
plt.tight_layout()


# #### What we have analysed :
# 
# * (i) BROOKLYN, QUEENS and BRONX has most complaints of Blocked Driveway.
# * (ii) MANHATTAN has most complaints of Noise - Street/Sidewalk.
# * (iii) STATEN ISLAND has most complaints of Illegal Parking.

# # Task 4
#     
#     -Ordering the complaint types based on average response time for different locations

# In[166]:


service_311_3['Request_Closing_Time_in_Hours'] = service_311_3['Request_Closing_Time'].astype('timedelta64[h]')+1
service_311_3[['Request_Closing_Time', 'Request_Closing_Time_in_Hours']].head()


# In[167]:


data_avg_time_in_hrs = service_311_3.groupby(['City', 'Complaint Type'])['Request_Closing_Time_in_Hours'].mean()
data_avg_time_in_hrs.head()


# In[168]:


service_311_3['Request_Closing_Time_in_Seconds'] = service_311_3['Request_Closing_Time'].astype('timedelta64[s]')
service_311_3[['Request_Closing_Time', 'Request_Closing_Time_in_Hours','Request_Closing_Time_in_Seconds']].head()


# In[169]:


data_avg_in_seconds = service_311_3.groupby(['City', 'Complaint Type']).Request_Closing_Time_in_Seconds.mean()
data_avg_in_seconds.head(10)


# In[170]:


service_311_3['Request_Closing_Time'].describe()


# In[172]:


mean_hrs = service_311_3['Request_Closing_Time_in_Hours'].mean()
std_hrs = service_311_3['Request_Closing_Time_in_Hours'].std()

mean_seconds = service_311_3['Request_Closing_Time_in_Seconds'].mean()
std_seconds = service_311_3['Request_Closing_Time_in_Seconds'].std()

print('The mean hours is {0:.2f} hours and mean Secondsis {1:.2f} seconds'.format(mean_hrs, mean_seconds))
print('The standard hours is {0:.2f} hours and standard Seconds is {1:.2f} seconds'.format(std_hrs, std_seconds))


# ### Analysing Complaint Types column on the basis of Months by refering to Created Date

# In[173]:


service_311_3['Year-Month'] = service_311_3['Created Date'].apply(lambda x:datetime.datetime.strftime(x, '%Y-%m'))


# In[174]:


service_311_3['Year-Month'].unique()


#   ### we got incident complaints from March to December

# In[175]:


monthly_incidents =  service_311_3.groupby('Year-Month').size().plot(figsize=(12,5),
                                                               title='Incident Counts on a monthly basis', ylabel='Counts')


# ###  As we can see that we don't have any complaints from January and Ferbruary in our dataset because we might have eliminated them as Null Values earlier.

# In[180]:


service_311_3.groupby(['Year-Month','Borough']).size().unstack().plot(figsize=(12,7))
plt.legend(loc='center left', bbox_to_anchor=(2.0, 1))


# ### As we can see Brooklyn raised most cases all over and most of them were raised in May-June and SEPTEMBER.

# In[181]:


service_311_3.groupby(['Borough', 'Year-Month']).size().unstack().plot(figsize=(15,7))
plt.legend(loc='center left', bbox_to_anchor=(2.0, 1))


# ### December has raised least complaints.

# In[185]:


service_311_3.groupby(['Year-Month','Borough'])['Request_Closing_Time_in_Hours'].mean().unstack().plot(figsize=(16,9),
                                                                        title='Processing time per Borough on a monthly basis');


# ### *Bronx has the maximum Processing time every month moreover it has the least complaints.

# # Task 5
# 
# ### Statistical Test
# 
# - Whether the average response time across complaint types is similar or not (overall)
# - Are the type of complaint or service requested and location related?
# 

# In[204]:



service_311_3.columns


# ### F- Test
# 
# ##### Testing at Confidence level(95%) => alpha value = 0.05
# * Null Hypothesis : H0 : There is no significant difference in average response time across different complaint types
# * Alternate Hypothesis : H1 : There is a significant difference in average response time across different complaint types

# In[205]:


avg_response_time = service_311_3.groupby(['Complaint Type']).Request_Closing_Time_in_Seconds.mean().sort_values(ascending=True)
avg_response_time


# In[206]:


data = {}
for complaint in service_311_3['Complaint Type'].unique():
    data[complaint] = np.log(service_311_3[service_311_3['Complaint Type']==complaint]['Request_Closing_Time_in_Seconds'])


# In[207]:


data.keys()


# In[208]:


for complaint in data.keys():
    print(data[complaint].std())


# In[209]:


from scipy.stats import f_oneway
# taking top 5 complaints
stat, p = f_oneway(data['Blocked Driveway'], data['Illegal Parking'], data['Noise - Street/Sidewalk'],
                   data['Derelict Vehicle'], data['Noise - Commercial'])
print('Statistics= %.3f, p = %.3f' % (stat, p))
# interpret
alpha = 0.05
if p > alpha:
    print('Same distributions (fail to reject H0)')
else:
    print('We have Different distributions (reject H0)')


# #### As we can see that our p-value is quite low , hence it is being converted to 0.0
# 
# * Since our p-value is lower than our critical p-value, we will conclude that we have enough evidence to reject our Null Hypothesis and that is:
# 
# * Average response time for all the complaints type is not same.

# ##  For relation between Complaint Type and Location
# - we will use Crosstab and Chi-square Test

# In[212]:


city_type = pd.crosstab(service_311_3['City'], service_311_3['Complaint Type'])
city_type.head()


# # Chi - Square Test

# In[213]:


table = city_type 
# table -->> The contingency table. The table contains the observed frequencies (i.e. number of occurrences) in each category.
# stat -->> chi2 or Test Statistic
# p -->> The p-value of the Test
# dof -->> Degrees of Freedom
# expected -->> The expected frequencies, based on the marginal sums of the table.
stat, p, dof, expected = chi2_contingency(table)


# print('The Degrees of Freedom are : {}'. format(dof))
# print('The P-Value of the Testing is {}: '.format(p))
# print('Expected values : \n')
# print(expected)

# In[216]:


location_complaint_type = pd.crosstab(service_311_3['Complaint Type'],service_311_3['Location Type'])


# In[218]:


import scipy.stats as stats


# In[219]:


cscore,pval,df,et = stats.chi2_contingency(location_complaint_type)
print("score : {:.2f} , pvalue : {:.2f}".format(cscore,pval))


# #### Here , pvalue (0.00) < alpha value(0.05)
# 

# ### We reject our Null Hypothesis
# - There is some significant relation between type of complaint and location (i.e) The type 
#  of complaint or service requested and the location are related

# In[ ]:




