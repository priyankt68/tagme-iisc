  import numpy as np
from sklearn import svm
from nolearn.dbn import DBN
features = []
filename=[]
vfeatures = []
vfilename=[]

lab = np.genfromtxt('feature_vectors.txt',usecols=0, dtype=str)
print "labels generated from feature vectors",lab
data = np.genfromtxt('feature_vectors.txt',dtype=float)[:,1:]
#data = np.genfromtxt('feature_vectors.txt')

print "Data:",data[0]
#print "Data:",data[2]



#datas=np.empty([500,239])

print "length of the data \n", len(data)

#for i in range(500):
#	datas[i] = data[i][1:241]          # storing the feature vectors
#	features[i].append((features[i][1:241]))

#print datas[0]
#print datas[499]

#Reading the filenames
#readfilename = open('feature_vectors.txt','r')

#for line in readfilename:
#	features.append((line.split(" ")))
	
#for i in range(500):
#	filename.append(features[i][0])    # storing the filenames

######################################


label = np.genfromtxt('labels.txt',usecols=1, dtype=int)

print label.shape
#for each in filename:
#	if 

#fi = label['filen']   ## FIlename

#la = label['labels']        ## Corresponding labels

#####################################################################  starts support vector machines part
print "Support vector machine starts"
C=1.0  # SVM regularization parameter
#Can check for the performance by changing the regularization parameter and plot the performance to better understand the data.
#        
#label=label.reshape(500,1)

clf=svm.SVC(kernel='rbf', gamma=0.001, C=4.8)

print "Fitting data.."
clf.fit(data,label)


print "Fitted ... "

#print "Prediction for first image: ", clf.predict(data[0])
#print "Prediction for last image: ", clf.predict(data[499])
#print "Prediction for 287^th image: ", clf.predict(data[288])

#dec=clf.decision_function()


################################################################ starts cross validation part

print "Reading cross validation dataset"

vfilename = np.genfromtxt('../Validation/feature_vectors.txt',usecols=0, dtype=str)
#print "labels generated from feature vectors",lab
vdata = np.genfromtxt('../Validation/feature_vectors.txt',dtype=float)[:,1:]
#vdata = np.genfromtxt('../Validation/feature_vectors.txt')

#vdatas=np.empty([500,240])

#print "length of the vdata : ", len(vdata)

#print vdata[0]

#for i in range(500):
#	vdatas[i] = vdata[i][1:241]          # storing the feature vectors


#Reading the filenames FOR validation dataset

#readfilename = open('../Validation/feature_vectors.txt','r')

#for line in readfilename:
#	vfeatures.append((line.split(" ")))
	
#for i in range(500):
#	vfilename.append(vfeatures[i][0])    # storing the filenames


print "Predicting for validation data"

#p={"1.0":"building","2.0":"cars","3.0":"face","4.0":"flowers","5.0":"shoes"}   # dictionary for labels

#Predicting for the lables of validation set

#vlabel=np.empty(500)
#vlab=[]
#for i in range(len(vdata)):
#	vlab.append(str(clf.predict(vdata[i])))
	#print "for",i,p[str(vlab[i])]  


submitlabel=[]
for i in range(len(vdata)):
	submitlabel.append('{} {}\n'.format(vfilename[i],int(clf.predict(vdata[i]))))	
	
submit = open('submit_dbn1.txt','w')
submit.writelines(submitlabel)	

#submit.write("tst")




	




	

