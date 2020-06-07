#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import pickle
from keras.utils import to_categorical


# In[2]:


#Loading data
filename = 'Data_Spectrograms.pkl' #Spectrograms
filename1= 'Data_Raw_signals.pkl' #Raw signals

spec= pd.read_pickle(filename)
signals = pd.read_pickle(filename1)


# In[3]:


spec_data = spec[0]
label = spec[1]

#Seperating the channels of raw signals
sequence_fpz_train = signals[0][:,0,:]
sequence_pz_train = signals[0][:,1,:]


# In[4]:


print(sequence_pz_train.shape)


# In[5]:


import scipy.stats
import pywt
import numpy as np
from collections import Counter

#Defining preprocessing functions. 
#A discrete wavelet transform is performed and features are extracted from different sub bands

def calculate_entropy(list_values):
    counter_values = Counter(list_values).most_common()
    probabilities = [elem[1]/len(list_values) for elem in counter_values]
    entropy=scipy.stats.entropy(probabilities)
    return entropy
 
def calculate_statistics(list_values):
    n5 = np.nanpercentile(list_values, 5)
    n25 = np.nanpercentile(list_values, 25)
    n75 = np.nanpercentile(list_values, 75)
    n95 = np.nanpercentile(list_values, 95)
    median = np.nanpercentile(list_values, 50)
    mean = np.nanmean(list_values)
    std = np.nanstd(list_values)
    var = np.nanvar(list_values)
    rms = np.nanmean(np.sqrt(list_values**2))
    return [n5, n25, n75, n95, median, mean, std, var, rms]
 
def calculate_crossings(list_values):
    zero_crossing_indices = np.nonzero(np.diff(np.array(list_values) > 0))[0]
    no_zero_crossings = len(zero_crossing_indices)
    mean_crossing_indices = np.nonzero(np.diff(np.array(list_values) > np.nanmean(list_values)))[0]
    no_mean_crossings = len(mean_crossing_indices)
    return [no_zero_crossings, no_mean_crossings]


def get_features(list_values):
    entropy = calculate_entropy(list_values)
    crossings = calculate_crossings(list_values)
    statistics = calculate_statistics(list_values)
    return [entropy] + crossings + statistics
 
def get_eeg_features(eeg_data, waveletname):
    eeg_feature_list = []
    
    for i in range(eeg_data.shape[0]):
        features = []
        signal = eeg_data[i,:]
        list_coeff = pywt.wavedec(signal,waveletname, level = 6)
        
        for coeff in list_coeff:
            features += get_features(coeff)
        
        eeg_feature_list.append(features)
    
    X = np.array(eeg_feature_list)
    
    return X


# In[6]:


feature_fpz_train = get_eeg_features(sequence_fpz_train,"db2")
feature_pz_train = get_eeg_features(sequence_pz_train, "db2")

X_raw = np.concatenate((feature_fpz_train,feature_pz_train),axis = 1)
print(X_raw.shape)

del sequence_fpz_train, sequence_pz_train, feature_fpz_train, feature_pz_train, signals


# In[7]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

X_raw = scaler.fit_transform(X_raw)


# In[8]:


X_spec_p1 = spec_data[:,0,:,:] #(15357, 100, 30)
X_spec_p2 = spec_data[:,1,:,:] #(15357, 100, 30)
X_spec_p1 = X_spec_p1.reshape(X_spec_p1.shape[0],X_spec_p1.shape[1]*X_spec_p1.shape[2]) #(15357, 3000)
X_spec_p2 = X_spec_p2.reshape(X_spec_p2.shape[0],X_spec_p2.shape[1]*X_spec_p2.shape[2]) #(15357, 3000)
print(X_spec_p1.shape)
print(X_spec_p2.shape)


# In[9]:


del spec_data


# In[10]:


X = np.concatenate((X_raw,X_spec_p1,X_spec_p2),axis = 1)
print(X.shape)


# In[11]:


del X_raw, X_spec_p1, X_spec_p2


# In[12]:


#Upsampling
from imblearn.over_sampling import SMOTE 
sm = SMOTE(random_state=42)
X_up, y_up = sm.fit_resample(X,label)


# In[13]:


#Train validation split
from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(X_up, y_up, test_size=0.25, random_state=42, stratify = y_up)


# In[14]:


y_train_onehot = to_categorical(y_train)
y_val_onehot = to_categorical(y_val)


# In[15]:


#See the class distributions
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

n, bins, patches = plt.hist(y_train, 20, facecolor='blue', alpha=0.5)
plt.ylabel('Class distribution of train data')
plt.show()  

n, bins, patches = plt.hist(y_val, 20, facecolor='blue', alpha=0.5)
plt.ylabel('Class distribution of val data')
plt.show() 


# In[16]:


# Giving original shape after train test split
X_train_raw = X_train[:,:168]
X_val_raw = X_val[:,:168]
X_train_spec_p1 = X_train[:,168:3168].reshape(X_train.shape[0],100,30)
X_train_spec_p2 = X_train[:,3168:].reshape(X_train.shape[0],100,30)
X_val_spec_p1 = X_val[:,168:3168].reshape(X_val.shape[0],100,30)
X_val_spec_p2 = X_val[:,3168:].reshape(X_val.shape[0],100,30)


# In[17]:


print(X_up.shape)
print(X_train_raw.shape)
print(X_train_spec_p1.shape)
print(X_train_spec_p2.shape)
print(X_val_raw.shape)
print(X_val_spec_p1.shape)
print(X_val_spec_p2.shape)


# In[18]:


from keras import Input
from keras import models
from keras.models import Model
from keras import layers
from keras import regularizers
from keras.layers import LSTM
from keras.layers import Dropout
from sklearn.model_selection import train_test_split
from keras.utils import plot_model 
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import Conv1D, MaxPooling1D
from keras.layers import SpatialDropout1D
from keras.layers import Dropout
from tensorflow.keras import regularizers
import keras.optimizers


# In[19]:


#Model
inp_1 = layers.Input(shape=(100,30))
c=layers.Conv1D(30, kernel_size=3, strides = 1, activation='relu')(inp_1)
c= MaxPooling1D(pool_size=4)(c)
c = Dropout(0.2)(c)
c = layers.Conv1D(30, kernel_size=3, strides=1, activation ='relu')(c)
c = MaxPooling1D(pool_size=4)(c)
c = Dropout(0.2)(c)
c = layers.LSTM(30, input_shape=(100, 30))(c)

inp_2 = layers.Input(shape=(100,30))
c1=layers.Conv1D(30, kernel_size=3, strides = 1, activation='relu')(inp_2)
c1 = MaxPooling1D(pool_size=4)(c1)
c1 = Dropout(rate=0.2)(c1)
c1 = layers.Conv1D(30, kernel_size=3, strides=1, activation ='relu')(c1)
c1 = MaxPooling1D(pool_size=4)(c1)
c1 = Dropout(rate=0.2)(c1)
c1 = layers.LSTM(30, input_shape=(100, 30))(c1)

inp_3 = layers.Input(shape = (168,))
c2 = layers.Dense(32, activation = "relu")(inp_3)
c2 = Dropout(rate=0.2)(c2)
c2 = layers.Dense(16, activation = "relu")(c2)

concatenated = layers.concatenate([c, c1, c2],axis=-1)

output_tensor = layers.Dense(6, activation='softmax')(concatenated)
model = Model([inp_1, inp_2, inp_3],output_tensor)


# In[20]:


model.summary()


# In[21]:


model.compile(optimizer= "adam",loss='categorical_crossentropy', metrics = ['accuracy'])
history = model.fit([X_train_spec_p1, X_train_spec_p2, X_train_raw], y_train_onehot, epochs= 30 ,validation_data=([X_val_spec_p1, X_val_spec_p2, X_val_raw],y_val_onehot), batch_size = 64)
val_los, val_acc = model.evaluate([X_val_spec_p1, X_val_spec_p2, X_val_raw], y_val_onehot)


# In[22]:


val_acc


# In[23]:


outcomes = history.history
train_accuracy = outcomes["accuracy"]
train_loss =outcomes["loss"]
val_accuracy = outcomes["val_accuracy"]
val_loss = outcomes["val_loss"]
epochs = range(1, len(val_accuracy) + 1)


# In[25]:


#With upsampling
y_train_predict = model.predict([X_train_spec_p1,X_train_spec_p2,X_train_raw])
y_val_predict = model.predict([X_val_spec_p1,X_val_spec_p2,X_val_raw])

y_train_predict_labels = np.zeros((len(y_train_predict,)))

for i in range(len(y_train_predict)):
    y_train_predict_labels[i] = np.argmax(y_train_predict[i])
    
y_train_predict_labels = y_train_predict_labels.astype("int")

y_val_predict_labels = np.zeros((len(y_val_predict,)))

for i in range(len(y_val_predict)):
    y_val_predict_labels[i] = np.argmax(y_val_predict[i])
    
y_val_predict_labels = y_val_predict_labels.astype("int")

get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
plt.plot(epochs, train_accuracy, 'g', label='Training accuracy') 
plt.plot(epochs, val_accuracy, 'b', label='Validation accuracy') 
plt.title('Training vs validation accuracy') 
plt.ylabel('Accuracy')
plt.xlabel('Epochs')
plt.legend()
plt.show()

plt.plot(epochs, train_loss, 'g', label='Training loss') 
plt.plot(epochs, val_loss, 'b', label='Validation loss') 
plt.title('Training vs validation loss') 
plt.ylabel('loss')
plt.xlabel('Epochs')
plt.legend()
plt.show()

from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

print(confusion_matrix(y_train, y_train_predict_labels, labels=[0,1,2,3,4,5]))
print(confusion_matrix(y_val, y_val_predict_labels, labels= [0,1,2,3,4,5]))
print(classification_report(y_train,y_train_predict_labels))
print(classification_report(y_val, y_val_predict_labels))
a = confusion_matrix(y_train,y_train_predict_labels) 
print(np.diag(a/a.sum(axis = 1))) #Train classwise accuracy
b = confusion_matrix(y_val,y_val_predict_labels)
print(np.diag(b/b.sum(axis = 1))) #Validation classwise accuracy


# In[154]:


# Upsampled Test predictions
test_data_spec = pd.read_pickle('Test_Spectrograms_no_labels.pkl')
X_test_spec_p1 = test_data_spec[0][:,0,:,:]
X_test_spec_p2 = test_data_spec[0][:,1,:,:]
print(X_test_spec_p1.shape)
print(X_test_spec_p2.shape)

test_data_raw = pd.read_pickle('Test_Raw_signals_no_labels.pkl')
sequence_fpz_test = test_data_raw[0][:,0,:]
sequence_pz_test  = test_data_raw[0][:,1,:]

feature_fpz_test = get_eeg_features(sequence_fpz_test,"db2")
feature_pz_test  = get_eeg_features(sequence_pz_test, "db2")

X_test_raw = np.concatenate((feature_fpz_test,feature_pz_test),axis = 1)

X_test_raw = scaler.transform(X_test_raw)

print(X_test_raw.shape)
del sequence_fpz_test, sequence_pz_test, feature_fpz_test, feature_pz_test, test_data_raw, test_data_spec


# In[155]:


#Prediction on test set
y_pred = model.predict([X_test_spec_p1,X_test_spec_p2, X_test_raw])

y_pred_labels = np.zeros((len(y_pred,)))

for i in range(len(y_pred)):
    y_pred_labels[i] = np.argmax(y_pred[i])
    
y_pred_labels = y_pred_labels.astype("int")

print(y_pred_labels.shape)
print(y_pred_labels[0:20])


# In[156]:
#Writing results to a text file

f = open("answer.txt","w")

np.savetxt("answer.txt",y_pred_labels, fmt ='%.0f')

f.close() 

