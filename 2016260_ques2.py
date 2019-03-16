#!/usr/bin/env python
# coding: utf-8

# In[11]:


import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import math
import random
from sklearn.tree import DecisionTreeClassifier


# In[12]:


X= np.array(pd.read_csv('./letter-recognition.data'))
print (X[0])
images=X[:,1:]
label=X[:,0]
print(images)
print(label)


# In[13]:


#PART A
randarray= np.arange(len(X))
np.random.shuffle(randarray)
breakpt= round(0.7*len(images))
X_train= []
Y_train= []
X_test= []
Y_test= []
for i in range(breakpt):
    X_train.append(images[randarray[i]])
    Y_train.append(label[randarray[i]])
for i in range(breakpt, len(images)):
    X_test.append(images[randarray[i]])
    Y_test.append(label[randarray[i]])
    
print (len(X_train), len(X_test))
X_train= np.array(X_train)
Y_train= np.array(Y_train)
print (Y_train[:10])
X_test= np.array(X_test)
Y_test= np.array(Y_test)
Y_train= [ord(char)-65 for char in Y_train]
Y_test= [ord(char)-65 for char in Y_test]

                 


# In[14]:


def getIndex(num, cum):
    index= -1
    for i in range(len(cum)):
        if(num<= cum[i]):
            return i #between 0to149 format
    


# In[15]:


def getAcc(yy_train, probsmatrix):
    trainpred= []
    for i in range(len(yy_train)):
        trainpred.append(np.argmax(probsmatrix[i]))
    trainpred= np.array(trainpred)
    #print (np.unique(trainpred))
    return ((yy_train== trainpred).sum())/len(yy_train)


# In[16]:


def boosting(X_train, Y_train, X_test, N):
    originaldt= np.zeros(len(X_train))
    originaldt+= 1/len(X_train)
    uniq= np.unique(Y_train)
    probstest= np.zeros([len(X_test), len(uniq)])
    probstrain= np.zeros([len(X_train), len(uniq)])
    print (originaldt)
    alphas= []
    for i in range(N):
        print (str(i) +  " " + "Iteration")
        indices= []
        #labeltemp= np.zeros(len(Y_train))
        cum= np.cumsum(originaldt)
        print (cum)
        #temp= np.zeros([len(X_train), len(X_train[0])])
#         for j in range(len(X_train)):
#             num= random.uniform(0, 1)
#             a= getIndex(num, cum)
#             indices.append(a)
#             labeltemp[j]= Y_train[a]
#             temp[j]= X_train[a]
        
#         originaldt=np.zeros(len(X_train))
#         originaldt+= 1/len(X_train)
        #print (labeltemp.shape, temp.shape)
        clf= DecisionTreeClassifier(random_state= 0, max_depth= 2, max_leaf_nodes= 5)
        #clf= clf.fit(temp, labeltemp)
        clf= clf.fit(X_train, Y_train, sample_weight= originaldt)
        #pred= clf.predict(temp)
        pred= clf.predict(X_train)
        pred= np.array(pred)
        error= (((pred!= Y_train)).sum())/len(pred)
        print (error)
        alpha= abs(1/2*math.log((1-error)/(error)))
        alphas.append(alpha)
        print (alpha)
        for j in range(len(pred)):
            if(pred[j]== Y_train[j]):
                originaldt[j]= originaldt[j]*math.exp(-alpha)
            else:
                originaldt[j]= originaldt[j]*math.exp(alpha)
        
        #now normalize the weights updated
        #print (originaldt)
        originaldt= originaldt/np.sum(originaldt)#normalize
    
        #for th final training and testing accuracies and errors on the dataset
        probstest+= clf.predict_proba(X_test)*alphas[i]#multiply by aplhas?
        probstrain+= clf.predict_proba(X_train)*alphas[i]#multiply by aplhas?
    
    print (probstrain.shape, probstest.shape)
    return (probstrain, probstest)
    
    
    


# In[17]:


# clf= DecisionTreeClassifier(max_depth= 2, max_leaf_nodes= 5)
# clf= clf.fit(X_train, Y_train)
# pred= clf.predict(X_test)
# print (np.unique(pred))
probstrain, probstest= boosting(X_train, Y_train, X_test, 10)


# In[18]:


accuracy= getAcc(Y_test, probstest)
accuracy2= getAcc(Y_train, probstrain)
print ("The accuracy on the train set for Boosting is: " + str(accuracy2))
print ("The accuracy on the test set for Boosting is: " + str(accuracy))


# In[19]:


#5 fold cross validation for Boosting
randarray2= np.arange(len(X_train))
breakpt2= round(0.2*len(X_train))
print (breakpt2)
accuracies= np.zeros(5)
testaccuracies= np.zeros(5)
for i in range(5):
    xx_train= []
    yy_train= []
    xx_val= []
    yy_val= []
    for j in range(i*breakpt2, min((i+1)*breakpt2, len(X_train))):
        xx_val.append(X_train[randarray2[j]])
        yy_val.append(Y_train[randarray2[j]])
    for j in range(len(X_train)):
        if(j< i*breakpt2 or j>= (i+1)*breakpt2):
            xx_train.append(X_train[randarray2[j]])
            yy_train.append(Y_train[randarray2[j]])
    
    print (len(xx_val), len(xx_train))
    probs_train, probs_val= boosting(xx_train, yy_train, xx_val, 10)
    acc=  getAcc(yy_val, probs_val)
    accuracies[i]= acc
    print ("The accuracy on validation set for " + str(i) + " fold is: " + str(acc))
    probs_train, probs_test= boosting(xx_train, yy_train, X_test, 10)
    acc2=  getAcc(Y_test, probs_test)
    testaccuracies[i]= acc2
    #print ("The accuracy on validation set for " + str(i) + " fold is: " + str(acc))

print("The mean accuracy on Validation set is: " + str(np.mean(accuracies)))
print("The standard deviation of accuracies on Validation set is: " + str(np.std(accuracies)))
index= np.argmax(testaccuracies)
print("The best model during C.V. gives accuraacy of " + str(testaccuracies[index]) + " on the Test set")    
    


# In[20]:


#Bagging done right
def bagging(X_train, Y_train, X_test, N):
    uniq= np.unique(Y_train)
    probstrain= np.zeros([len(X_train), len(uniq)])
    probstest= np.zeros([len(X_test), len(uniq)])
    for i in range(N):
        print (str(i) + " Iteration")
        indices= []
        temp= np.zeros([len(X_train), len(X_train[0])])
        labeltemp= np.zeros(len(Y_train))
        for j in range(len(X_train)):
            a= random.randint(0, len(X_train)-1)#since both are inclusive otherwise
            indices.append(a)
            temp[j]= X_train[a]
            labeltemp[j]= Y_train[a]
        
        
        print(labeltemp.shape, temp.shape)
        clf= DecisionTreeClassifier(random_state= 0, max_depth= 2, max_leaf_nodes= 5)
        clf= clf.fit(temp, labeltemp)
        probstest+= clf.predict_proba(X_test)#multiply by alphas??
        probstrain+= clf.predict_proba(X_train) 
    
    return (probstrain, probstest)    
    


# In[30]:


probstrain, probstest= bagging(X_train, Y_train, X_test, 10)


# In[31]:


accuracy= getAcc(Y_test, probstest)
accuracy2= getAcc(Y_test, probstrain)
print ("The accuracy on the train set for Bagging is: " + str(accuracy2))
print ("The accuracy on the test set for Bagging is: " + str(accuracy))


# In[35]:


#normalization on Bagging!
#MIN-MAX
temp= probstest
# print (temp.shape)
# for i in range(len(probstest)):
#     maxx= np.max(temp[i])
#     minn= np.min(temp[i])
#     temp[i]= (temp[i]-minn)/(maxx-minn)

#Z-score normalization
# for i in range(len(probstest)):
#     mean= np.mean(temp[i])
#     std= np.std(temp[i])
#     temp[i]= (temp[i]-mean)/std

#tanh normalization
for i in range(len(probstest)):
    temp[i]= np.tanh(temp[i])

accuracy= getAcc(Y_test, temp)
print ("The accuracy on the test set for Bagging is: " + str(accuracy))


# In[34]:


a= np.array([1,2,3])
b= np.max(a)
a-= b
print (np.tanh(a))
print (a)


# In[ ]:


#5 fold cross validation for Bagging
randarray2= np.arange(len(X_train))
breakpt2= round(0.2*len(X_train))
print (breakpt2)
valaccuracies= np.zeros(5)
testaccuracies= np.zeros(5)
for i in range(5):
    xx_train= []
    yy_train= []
    xx_val= []
    yy_val= []
    for j in range(i*breakpt2, min((i+1)*breakpt2, len(X_train))):
        xx_val.append(X_train[randarray2[j]])
        yy_val.append(Y_train[randarray2[j]])
    for j in range(len(X_train)):
        if(j< i*breakpt2 or j>= (i+1)*breakpt2):
            xx_train.append(X_train[randarray2[j]])
            yy_train.append(Y_train[randarray2[j]])
    
    print (len(xx_val), len(xx_train))
    probs_train, probs_val= bagging(xx_train, yy_train, xx_val, 10)
    acc=  getAcc(yy_val, probs_val)
    valaccuracies[i]= acc
    print ("The accuracy on validation set for " + str(i) + " fold is: " + str(acc))
    probs_train, probs_test= bagging(xx_train, yy_train, X_test, 10)
    acc2=  getAcc(Y_test, probs_test)
    testaccuracies[i]= acc2
    #print ("The accuracy on validation set for " + str(i) + " fold is: " + str(acc))

print("The mean accuracy on Validation set is: " + str(np.mean(valaccuracies)))
print("The standard deviation of accuracies on Validation set is: " + str(np.std(valaccuracies)))
index= np.argmax(testaccuracies)
print("The best model during C.V. gives accuraacy of " + str(testaccuracies[index]) + " on the Test set")    
    


# In[ ]:




