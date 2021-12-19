# -*- coding: utf-8 -*-
"""
Created on Sun Dec 19 18:00:37 2021

@author: Ommarselim
"""
#import Liberaries ...
import pandas as pd 
from sklearn.impute import SimpleImputer
import numpy as np
import matplotlib.pyplot as plt


#1.	Read the dataset, convert it to DataFrame and display some from it.
#2.	Display structure and summary of the data.
dataset = pd.read_csv("Wuzzuf_Jobs.csv")
dataset.describe()
print(dataset.describe())
print(dataset)

#sorting data
dataset.sort_values("Title", inplace = True)

#3 Clean the data (duplications)
dataset.drop_duplicates(subset =["Title","Company","Location","Type","Level","YearsExp","Country","Skills"],keep = "first", inplace = True)

#Cleaning the Null data
imp = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
imp = imp.fit(dataset)
transformed_data = imp.transform(dataset)


#4 Count the jobs for each company and display that in order (What are the most demanding companies for jobs?)
#the most deamanded company for jobs is "CONFIDENTAL"
x=dataset['Company'].value_counts()


mini_data = x[0:10]  #عشان الداتا ححمها كبير , نخلي الجراف شكله احسن
#5.	Show step 4 in a pie chart
labels = mini_data.index
explode = (0.1, 0, 0, 0, 0, 0,0,0,0,0) #Specifying the exploding position of pie element
plt.figure(figsize=(15,5))
plt.pie(mini_data , labels = labels ,explode=explode,autopct='%1.2f%%', shadow=True)
plt.title('jobs for each company')
plt.show()
# creating the bar plot
plt.figure(figsize=(15,5))
plt.bar(mini_data.index ,mini_data , color ='maroon',
		width = 0.4)
plt.xlabel("Companies")
plt.ylabel("number of jobs")
plt.title("jobs for each company")
plt.show()

#6.	Find out what are the most popular job titles.
jobtitle=dataset['Title'].value_counts()
#7.	Show step 6 in bar chart
jobtitle=jobtitle[0:10]
plt.figure(figsize=(15,5))

plt.bar(jobtitle.index ,jobtitle , color ='green',	width = 0.4)
plt.show()


#8.	Find out the most popular areas?
Location=dataset['Location'].value_counts()
Location=Location[0:10]
#9.	Show step 8 in bar chart
plt.figure(figsize=(15,5))
plt.bar(Location.index ,Location , color ='red', width = 0.4)
plt.show()


#10.	Print skills one by one and how many each repeated and order the output to find out the most important skills required
#print skills one by one
skills=dataset['Skills']
print(skills)
#count skills> how many repeated
L=dataset['Skills'].value_counts()
#
print()
print()
print()

print(" the most skills required is :     "+ L.index[0])



