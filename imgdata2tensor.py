
import numpy as np
import tensorflow as tf
import glob
import cv2
import os

paths = glob.glob("training_set\BreastDataset\**\*.tif")
#%%
X = []
labels = []


for path in paths:
  label = path.split("\\")[-2] 
  labels.append(label)
  img = cv2.imread(path)  
  img = cv2.resize(img, (300, 300))   
  img = np.array(img)
  X.append(img)



#%%

from sklearn.preprocessing import LabelEncoder
X = np.array(X)

le = LabelEncoder()
y = le.fit_transform(labels)
y = tf.keras.utils.to_categorical(y)

#%%%

from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size = 0.1, random_state = 0)


#%%
import pickle
pickle.dump(X_train, open("X_train.p", "wb"))
pickle.dump(y_train, open("y_train.p", "wb"))
pickle.dump(X_val, open("X_val.p", "wb"))
pickle.dump(y_val, open("y_val.p", "wb"))
