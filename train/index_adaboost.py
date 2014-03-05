import numpy as np
from sklearn import svm
from nolearn.dbn import DBN
from sklearn.datasets import make_gaussian_quantiles
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.externals.six.moves import xrange
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn import preprocessing

filename=[]
vfeatures = []
vfilename=[]



data = np.genfromtxt('feature_vectors.txt',dtype=float)[:,1:]

#data_scaled = preprocessing.normalize(data, norm='l2')


label = np.genfromtxt('labels.txt',usecols=1, dtype=int)

#####################################################################  starts support vector machines part
print "Support vector machine starts"

bdt_real = AdaBoostClassifier(
    DecisionTreeClassifier(max_depth=9),
    n_estimators=700,
    learning_rate=1)
C=1.0  # SVM regularization parameter
#Can check for the performance by changing the regularization parameter and plot the performance to better understand the data.
#        
#label=label.reshape(500,1)



print "Fitting data.."
bdt_real.fit(data,label)

print "Fitted ... "

################################################################ starts cross validation part

print "Reading cross validation dataset"

vfilename = np.genfromtxt('../Validation/feature_vectors.txt',usecols=0, dtype=str)

vdata = np.genfromtxt('../Validation/feature_vectors.txt',dtype=float)[:,1:]

print "Predicting for validation data"


submitlabel=[]
for i in range(len(vdata)):
	submitlabel.append('{} {}\n'.format(vfilename[i],int(bdt_real.predict(vdata[i]))))	
	
submit = open('submit_decision_tree.txt','w')
submit.writelines(submitlabel)	






	




	

