# -*- coding: utf-8 -*-
"""
Created on Fri Jun  8 10:21:50 2018

@author: Aqeuous Carlos
"""

import pandas as pd
import numpy as np
import random
from keras import utils
from keras import models
from keras import layers
from keras import optimizers
from keras import activations
from keras import backend as K
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import auc


def create_nn(n_features,n_categories,act_fun,opt):

    network = models.Sequential()
    
    network.add(layers.Dense(units=10, activation= 'relu', input_shape=(n_features,)))
    
    network.add(layers.Dense(units=10, activation= 'relu'))
    
    network.add(layers.Dense(units=n_categories, activation=act_fun))
    
    network.compile(loss='binary_crossentropy',
                    optimizer=opt, 
                    metrics=['accuracy']) 
    return network

def get_data(df, ini_sample):

	close = df['close'].tolist()

	id_x = []

	for i in range(3,20+1):
		for j in range(7,65+1):
			for k in range(1,10+1):
				for l in range(1,10+1):
					id_x.append(([i,j,k,l],sum([i,j,k,l])))

	id_x = random.sample(id_x, ini_sample)

	lst = []
	cup_lst = []
	for entry in id_x:

		span = entry[1]
		period = entry[0]

		all_lists = [sublist for sublist in (close[x:x+span] for x in range(len(close) - span + 1))]

		for val in all_lists:
			cup = 0
			lst.append(val)
			if (val[period[0]] - val[0])/val[0] > 0.20 and (val[period[0]] - val[0])/val[0] <= 0.30:
				if (val[period[1]] - val[period[0]])/val[period[0]] > -0.10 and (val[period[1]] - val[period[0]])/val[period[0]] <= -0.01:
					aux = ''
					for n in range(period[0],period[1]+1):
						if (val[n] - val[period[0]])/val[period[0]] < -0.10 and (val[n] - val[period[0]])/val[period[0]] >= -0.35:
							aux = True
							break
					if aux == True:
						if (val[period[2]] - val[period[1]])/val[period[1]] > -0.20 and (val[period[2]] - val[period[1]])/val[period[1]] <= -0.05:
							if (val[period[3]] - val[period[2]])/val[period[2]] > 0.06 and (val[period[3]] - val[period[2]])/val[period[2]] <=0.30:
								cup = 1
			cup_lst.append(cup)

	max_length = 0
	for vals in lst:
		if len(vals) > max_length:
			max_length = len(vals)

	X = np.zeros((len(lst), max_length))

	for i in range(0,len(X)):
		X[i][0:len(lst[i])] = lst[i]

	Y = np.array(cup_lst)  

	return X, Y


def nn_model(df,ini_sample=1000):

    per = 0.90

    sw = 0
    while sw == 0:
	    X, Y = get_data(df,ini_sample)

	    X_train, X_test = X[:int(round(len(X)*per)-1)], X[int(round(len(X)*per)):len(X)]
	    Y_train, Y_test = Y[:int(round(len(X)*per)-1)], Y[int(round(len(X)*per)):len(Y)]

	    if np.sum(Y_test) > 1:
	    	sw = 1


    n_features = X_train.shape[1]

    scaler = StandardScaler()
    scaler.fit(X_train)

    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    network = create_nn(n_features,
                        len(Y.shape),
                        'sigmoid',
                        'adam')

    # Train neural network
    history = network.fit(X_train,
                          Y_train, 
                          epochs=5, 
                          verbose=1, 
                          batch_size=25,
                          validation_data=(X_test, Y_test)) 


    y_pred_keras = network.predict(X_test).ravel()
    fpr_keras, tpr_keras, thresholds_keras = roc_curve(Y_test, y_pred_keras)
    roc_score = roc_auc_score(Y_test, y_pred_keras)
    auc_score = auc(fpr_keras, tpr_keras)
    print 'roc score: {}'.format(roc_score)
    acc = history.history['val_acc'][-1]
    print 'accuracy: {}'.format(acc)


if __name__ == "__main__":
	df = pd.read_csv('HK.00700Adj@Futu.csv')
	nn_model(df,1000)