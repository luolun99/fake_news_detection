#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[2]:


path = "/Users/huanzhang/Desktop/Computer_Science/final project/data/liar/"


# In[3]:


# load data into dataframes and add variable names
column_names =  [
    'id',
    'label',
    'statement',
    'subject',
    'speaker',
    'job_title',
    'state_info',
    'party_affiliation',
    'barely_true_counts',
    'false_counts',
    'half_true_counts',
    'mostly_true_counts',
    'pants_onfire_counts',
    'context'
]
train = pd.read_csv(path+"train.tsv", sep = "\t", header=None)
valid = pd.read_csv(path+"valid.tsv", sep = "\t", header=None)
test = pd.read_csv(path+"test.tsv", sep = "\t", header=None)
df = pd.concat([train, valid, test])
df.columns = column_names
df.astype(str)
random_seed = 42


# In[4]:


df.head()


# In[5]:


print(f"There are in total {df.shape[0]} records.")
print("Here are some sample statements:")
print(df['statement'][:3])

# Use the apply function with len to find the length of each text in the 'TextColumn'
df['statement_length'] = df['statement'].apply(len)

# Calculate and print the longest, shortest, and average length
max_length = df['statement_length'].max()
min_length = df['statement_length'].min()
avg_length = df['statement_length'].mean()

print(f"Longest statement has: {max_length} words.")
print(f"Shortest statement has: {min_length}")
print(f"Average length of the statements are: {round(avg_length)}")


# In[6]:


import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

# Get English stop words from NLTK
stop_words = set(stopwords.words('english'))

# Function to remove stop words and punctuation from a text
def process_text(text):
       # Check if the value is a non-null string
    if isinstance(text, str):
        # Tokenize the text
        tokens = word_tokenize(text)
        
        # Remove stop words
        tokens = [word.lower() for word in tokens if word.lower() not in stop_words]
        
        # Create a Porter Stemmer instance
        porter_stemmer = PorterStemmer()
        
        # Remove punctuation and apply stemming
        tokens = [porter_stemmer.stem(word) for word in tokens if word not in string.punctuation]
        
        # Join the tokens back into a string
        processed_text = ' '.join(tokens)
        
        return processed_text
    else:
        # If the value is not a string, return an empty string
        return ''


# In[7]:


df1 = df[['statement', 'subject', 'speaker', 'job_title', 'state_info', 'party_affiliation', 'context']]
# Function to merge values in a row
def merge_row(row):
    return ' '.join(map(str, row))

def encode_label(l):
    encoded_label = []
    for label in l:
        if label == 'mostly-true' or label == 'true':
            label = 0
        else:
            label = 1
        encoded_label.append(label)
    return encoded_label

# Apply the function to each row
combined_features = df1.apply(merge_row, axis=1)
combined_features = combined_features.apply(process_text).tolist()

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

# Vectorize the text data
#vectorizer = CountVectorizer()
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(combined_features)
y = np.array(encode_label(df['label']))


# In[8]:


from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Split the data into training and validation sets
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=random_seed)

# Train the Naive Bayes classifier
clf = MultinomialNB()
clf.fit(X_train, y_train)

# Make predictions on the validation set
y_pred = clf.predict(X_valid)

# Evaluate the model

# Perform cross-validation
cv_scores = cross_val_score(clf, X_train, y_train, cv=5, scoring='accuracy')
print(f"Cross-Validation Accuracy: {np.mean(cv_scores):.2f}")

# Print confusion matrix and classification report
conf_matrix = confusion_matrix(y_valid, y_pred)
print("Confusion Matrix:")
print(conf_matrix)

class_report = classification_report(y_valid, y_pred)
print("\nClassification Report:")
print(class_report)


# In[9]:


import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Tokenize the text
tokenizer = Tokenizer(oov_token='<OOV>',  
                      filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',
                      lower=True,
                      split=' ')
tokenizer.fit_on_texts(combined_features)
word_index = tokenizer.word_index

# Convert text to sequences and pad sequences
sequences = tokenizer.texts_to_sequences(combined_features)
padded_sequences = pad_sequences(sequences, padding='post', truncating='post')

input_length = max(len(seq) for seq in sequences)

vocabulary_size = len(word_index) + 1


# In[10]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, GlobalMaxPooling1D, Flatten, Dense, Dropout
from tensorflow.keras import regularizers
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(padded_sequences, y, test_size=0.2, random_state=random_seed)

# Build a neural network for text classification
model = Sequential()
model.add(Embedding(input_dim=vocabulary_size, output_dim=150, input_length=input_length))
model.add(GlobalMaxPooling1D()) 
model.add(Dense(32, activation='relu', kernel_regularizer=regularizers.l2(0.01)))
model.add(Dense(32, activation='relu', kernel_regularizer=regularizers.l2(0.01)))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
history = model.fit(X_train, 
                    y_train,
                    epochs=10, 
                    batch_size=128,
                    validation_split=0.2)


# In[11]:


import matplotlib.pyplot as plt
# Plot training history
plt.figure(figsize=(12, 6))

# Plot training & validation accuracy values
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(['Train', 'Validation'], loc='upper left')

# Plot training & validation loss values
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(['Train', 'Validation'], loc='upper left')

plt.tight_layout()
plt.show()


# In[12]:


from sklearn.metrics import confusion_matrix, classification_report

# Evaluate the model on the test set
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print("\nTest accuracy:")
print(test_accuracy)


# Evaluate the model on the test set
y_pred = model.predict(X_test)
y_pred_binary = np.round(y_pred)  # Convert probabilities to binary predictions (0 or 1)

# Print confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred_binary)
print("\nConfusion Matrix:")
print(conf_matrix)

# Print classification report
class_report = classification_report(y_test, y_pred_binary)
print("\nClassification Report:")
print(class_report)

