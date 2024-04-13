#!/usr/bin/env python
# coding: utf-8

# In[1]:


#################################### Load and Explore the Data ##############################################################
import pandas as pd
import numpy as np

# Load the dataset
df = pd.read_csv(r"C:\Users\ciana\OneDrive\Documents\DCU\Final Year\MT412 - Professional Business Analytics\IMDB_Dataset.csv")


# In[2]:


# Display the first few rows of the dataframe
print(df.head())


# In[3]:


# Display the first review
df['review'][1]


# In[4]:


# Check for any missing values
print(df.isnull().sum())


# In[5]:


# Get a basic description of the dataset
print(df.describe())


# In[6]:


# Check for duplicates
df.duplicated().sum()


# In[7]:


# Remove duplicates
df.drop_duplicates(inplace = True)


# In[8]:


# Check to see if duplicates are gone
df.duplicated().sum()


# In[9]:


# What is the breakdown of positive review to negative reviews
df['sentiment'].value_counts()


# In[10]:


##################################### Data Processing ####################################################


# In[11]:


import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer


# In[12]:


# Clean text
def clean_text(text):
    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)
    # Remove special characters and numbers
    text = re.sub(r'\W', ' ', text)
    # Remove single characters
    text = re.sub(r'\s+[a-zA-Z]\s+', ' ', text)
    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text, flags=re.I)
    return text


# In[13]:


# Clean the reviews
df['review'] = df['review'].apply(clean_text)


# In[14]:


# Examining the text
print(df)


# In[15]:


# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df['review'], df['sentiment'], test_size=0.25, random_state=0)

# The test set is 25% and the training set is 75%


# In[16]:


# Vectorise the text
## We are vectorising the text to convert the tokens into numerical values so we can model the data later.

vectorizer = TfidfVectorizer(max_features=5000)

# Fit the vectorizer on the training data and transform it
X_train = vectorizer.fit_transform(X_train)

# Transform the test data
X_test = vectorizer.transform(X_test)


# In[27]:


# Converting the sentimental labels to numeric values
y_train = y_train.apply(lambda x: 1 if x == 'positive' else 0)
y_test = y_test.apply(lambda x: 1 if x == 'positive' else 0)

# If the review is positive it will return a 1 and if it is negative it will return a 0.


# In[18]:


# Ensuring sentiment labels were changed to binary
print(y_test)


# In[19]:


# If a review is positive it returns a 1 and if the review is neagtive it returns a 0


# In[20]:


############################### Building Model and Training #########################################################


# In[21]:


# I used a Random Forest Classifier to build this model
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Initialise the Random Forest classifier with 100 trees
model = RandomForestClassifier(n_estimators=100, random_state=0)


# In[22]:


# Train the model on the training data
model.fit(X_train, y_train)


# In[23]:


#################################### Prediction and Evaluation ################################################


# In[24]:


# Predict the sentiments on the test data
y_pred = model.predict(X_test)
print(y_pred)


# In[25]:


# Calculate and print the accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')


# In[26]:


# Detailed classification report
print(classification_report(y_test, y_pred))


# In[ ]:




