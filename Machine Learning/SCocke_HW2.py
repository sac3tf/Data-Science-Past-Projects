# -*- coding: utf-8 -*-
"""

@author: Steven with office hours assistance from Chris Havenstein
"""

import numpy as np
import numpy.lib.recfunctions as rfn
from collections import OrderedDict
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from itertools import product
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier


##Homework questions answered after data prep and exploration below!!!!


#Read the two first two lines of the file.
with open('/Users/stevencocke/Desktop/Machine Learning/HW2/claim.sample.csv', 'r') as f:
    print(f.readline())
    print(f.readline())


#Colunn names that will be used in the below function, np.genfromtxt
names = ["V1","Claim.Number","Claim.Line.Number",
         "Member.ID","Provider.ID","Line.Of.Business.ID",
         "Revenue.Code","Service.Code","Place.Of.Service.Code",
         "Procedure.Code","Diagnosis.Code","Claim.Charge.Amount",
         "Denial.Reason.Code","Price.Index","In.Out.Of.Network",
         "Reference.Index","Pricing.Index","Capitation.Index",
         "Subscriber.Payment.Amount","Provider.Payment.Amount",
         "Group.Index","Subscriber.Index","Subgroup.Index",
         "Claim.Type","Claim.Subscriber.Type","Claim.Pre.Prince.Index",
         "Claim.Current.Status","Network.ID","Agreement.ID"]


#https://docs.scipy.org/doc/numpy-1.12.0/reference/arrays.dtypes.html
#These are the data types or dtypes that will be used in the below function, np.genfromtxt()
types = ['S8', 'f8', 'i4', 'i4', 'S14', 'S6', 'S6', 'S6', 'S4', 'S9', 'S7', 'f8',
         'S5', 'S3', 'S3', 'S3', 'S3', 'S3', 'f8', 'f8', 'i4', 'i4', 'i4', 'S3', 
         'S3', 'S3', 'S4', 'S14', 'S14']


#NumPy Structured Arrays: https://docs.scipy.org/doc/numpy/user/basics.rec.html
# Though... I like this Structured Array explanation better in some cases: https://jakevdp.github.io/PythonDataScienceHandbook/02.09-structured-data-numpy.html
#np.genfromtxt:  https://docs.scipy.org/doc/numpy/reference/generated/numpy.genfromtxt.html

#read in the claims data into a structured numpy array
CLAIMS = np.genfromtxt('/Users/stevencocke/Desktop/Machine Learning/HW2/claim.sample.csv', dtype=types, delimiter=',', names=True, 
                       usecols=[0,1,2,3,4,5,
                                6,7,8,9,10,11,
                                12,13,14,15,16,
                                17,18,19,20,21,
                                22,23,24,25,26,
                                27,28])

#print dtypes and field names
print(CLAIMS.dtype)

#Notice the shape differs since we're using structured arrays.
print(CLAIMS.shape)

#However, you can still subset it to get a specific row.
print(CLAIMS[0])

#Subset it to get a specific value.
print(CLAIMS[0][1])

#Get the names
print(CLAIMS.dtype.names)

#Subset into a column
print(CLAIMS['MemberID'])

#Subset into a column and a row value
print(CLAIMS[0]['MemberID'])


#String Operations in NumPy - https://docs.scipy.org/doc/numpy-1.13.0/reference/routines.char.html

#Sorting, Searching, and Counting in NumPy - https://docs.scipy.org/doc/numpy-1.13.0/reference/routines.sort.html

# You might see issues here: https://stackoverflow.com/questions/23319266/using-numpy-genfromtxt-gives-typeerror-cant-convert-bytes-object-to-str-impl

# If you do, encode as a unicode byte object
#A test string
test = 'J'
test = test.encode()

#A test NumPy array of type string
testStrArray = np.array(['Ja','JA', 'naJ', 'na' ],dtype='S9')

#Showing what the original string array looks like
print('Original String Array: ', testStrArray)

#Now try using startswith()
Test1Indexes = np.core.defchararray.startswith(testStrArray, test, start=0, end=None)
testResult1 = testStrArray[Test1Indexes]

#Showing what the original subset string array looks like with startswith()
print('Subset String Array with startswith(): ', testResult1)

#Now try using find()
TestIndexes = np.flatnonzero(np.core.defchararray.find(testStrArray,test)!=-1)

testResult2 = testStrArray[TestIndexes]

#Showing what the original subset string array looks like with find()
print('Subset String Array with find(): ', testResult2)

#Try startswith() on CLAIMS
JcodeIndexes = np.flatnonzero(np.core.defchararray.startswith(CLAIMS['ProcedureCode'], test, start=0, end=None)!=-1)

np.set_printoptions(threshold=500, suppress=True)

#Using those indexes, subset CLAIMS to only Jcodes
Jcodes = CLAIMS[JcodeIndexes]

print(Jcodes)

print(Jcodes[0][9])


#Try find() on CLAIMS
JcodeIndexes = np.flatnonzero(np.core.defchararray.find(CLAIMS['ProcedureCode'], test, start=1, end=2)!=-1)

#Using those indexes, subset CLAIMS to only Jcodes
Jcodes = CLAIMS[JcodeIndexes]

print(Jcodes)
print(Jcodes[1][9])

print(Jcodes.dtype.names)

print(Jcodes[0][14])

## HW notes:
'''    
A medical claim is denoted by a claim number ('Claim.Number'). Each claim consists of one or more medical lines denoted by a claim line number ('Claim.Line.Number').

1. J-codes are procedure codes that start with the letter 'J'.

     A. Find the number of claim lines that have J-codes.'''
print(Jcodes.shape) ##51,029 have J-codes

''' B. How much was paid for J-codes to providers for 'in network' claims?''' ##$2418429.57
#Sorted Jcodes, by ProviderPaymentAmount
Sorted_Jcodes = np.sort(Jcodes, order='ProviderPaymentAmount')


# Reverse the sorted Jcodes (A.K.A. in descending order)
Sorted_Jcodes = Sorted_Jcodes[::-1]

# Subset
ProviderPayments = Sorted_Jcodes['ProviderPaymentAmount']
Jcodes_Sorted = Sorted_Jcodes['ProcedureCode']



#Join arrays together
arrays = [Jcodes_Sorted, ProviderPayments]

#https://www.numpy.org/devdocs/user/basics.rec.html
Jcodes_with_ProviderPayments = rfn.merge_arrays(arrays, flatten = True, usemask = False)

#GroupBy JCodes using a dictionary
JCode_dict = {}

#Aggregate with Jcodes - code  modifiedfrom a former student's code, Anthony Schrams
for aJCode in Jcodes_with_ProviderPayments:
    if aJCode[0] in JCode_dict.keys():
        JCode_dict[aJCode[0]] += aJCode[1]
    else:
        aJCode[0] not in JCode_dict.keys()
        JCode_dict[aJCode[0]] = aJCode[1]

#sum the JCodes
np.sum([v1 for k1,v1 in JCode_dict.items()])


''' C. What are the top five J-codes based on the payment to providers?'''
#create an OrderedDict (which we imported from collections): https://docs.python.org/3.7/library/collections.html#collections.OrderedDict
#Then, sort in descending order
JCodes_PaymentsAgg_descending = OrderedDict(sorted(JCode_dict.items(), key=lambda aJCode: aJCode[1], reverse=True))
    
#print the results        
print(JCodes_PaymentsAgg_descending)


'''2. For the following exercises, determine the number of providers that were paid for at least one J-code. Use the J-code claims for these providers to complete the following exercises.

    A. Create a scatter plot that displays the number of unpaid claims (lines where the ‘Provider.Payment.Amount’ field is equal to zero) for each provider versus the number of paid claims.'''
    
Jcodes_Claim = Jcodes['ProviderPaymentAmount']
Jcodes_Providers = Jcodes['ProviderID']
arrays = [Jcodes_Providers,Jcodes_Claim]
Providers_with_ProviderPayments = rfn.merge_arrays(arrays, flatten = True, usemask = False)

Providers_with_ProviderPayments.dtype

Provider_Payments_dict = {}
for aProvider in Providers_with_ProviderPayments:
    if aProvider[0] not in Provider_Payments_dict.keys():
        if aProvider[1] == 0:
            Provider_Payments_dict[aProvider[0]] = [1,0] 
        else:
            Provider_Payments_dict[aProvider[0]] = [0,1] 
    else:
        if aProvider[1] == 0:
            Provider_Payments_dict[aProvider[0]][0] += 1
        else: 
            Provider_Payments_dict[aProvider[0]][1] += 1
            
colors = list("rgbcmykrgbcmykr")

for data_dict in Provider_Payments_dict:
   x = Provider_Payments_dict[data_dict][0]
   y = Provider_Payments_dict[data_dict][1]
   plt.scatter(x,y,color=colors.pop(), label = data_dict)   
plt.title('Unpaid Claims vs Paid Claims')
plt.legend(bbox_to_anchor=(1.1,1.05))
plt.xlabel('Unpaid Claims')
plt.ylabel('Paid Claims')
plt.show()



'''B. What insights can you suggest from the graph?'''
print('The insights I suggest are that there are several providers that have not paid many more times then they have paid')

'''C. Based on the graph, is the behavior of any of the providers concerning? Explain.'''
print('Yes several providers have many unpaid claims compared to the claims that they have paid on')


'''3. Consider all claim lines with a J-code.

     A. What percentage of J-code claim lines were unpaid?'''
print(100*(sum(item[0] for item in Provider_Payments_dict.values())/(sum(item[1] for item in Provider_Payments_dict.values())+sum(item[0] for item in Provider_Payments_dict.values()))), 'Percent')

'''B. Create a model to predict when a J-code is unpaid. Explain why you choose the modeling approach.'''
print('The model is created below. Random forest is used and it predicts the outcome of a claim being unpaid.Random forest is one of the most popular and accurate classifiers in the machine learning arsenal. It is alsomost easily understood in my opinion. The model and preparation is in the code below. Much of it was suppliedin office hours, but it is understood and this part was mainly from hw1.')
''' C. How accurate is your model at predicting unpaid claims?'''
print('The accuracy for several different settings, for k=5 folds are all 97% or higher.') 
''' D. What data attributes are predominately influencing the rate of non-payment?'''


print(Sorted_Jcodes.dtype.names)

##We need to come up with labels for paid and unpaid Jcodes

## find unpaid row indexes  

unpaid_mask = (Sorted_Jcodes['ProviderPaymentAmount'] == 0)

## find paid row indexes
paid_mask = (Sorted_Jcodes['ProviderPaymentAmount'] > 0)


#Here are our
Unpaid_Jcodes = Sorted_Jcodes[unpaid_mask]

Paid_Jcodes = Sorted_Jcodes[paid_mask]

#These are still structured numpy arrays
print(Unpaid_Jcodes.dtype.names)
print(Unpaid_Jcodes[0])

print(Paid_Jcodes.dtype.names)
print(Paid_Jcodes[0])

#Now I need to create labels
print(Paid_Jcodes.dtype.descr)
print(Unpaid_Jcodes.dtype.descr)

#create a new column and data type for both structured arrays
new_dtype1 = np.dtype(Unpaid_Jcodes.dtype.descr + [('IsUnpaid', '<i4')])
new_dtype2 = np.dtype(Paid_Jcodes.dtype.descr + [('IsUnpaid', '<i4')])

print(new_dtype1)
print(new_dtype2)

#create new structured array with labels

#first get the right shape for each.
Unpaid_Jcodes_w_L = np.zeros(Unpaid_Jcodes.shape, dtype=new_dtype1)
Paid_Jcodes_w_L = np.zeros(Paid_Jcodes.shape, dtype=new_dtype2)

#check the shape
Unpaid_Jcodes_w_L.shape
Paid_Jcodes_w_L.shape

#Look at the data
print(Unpaid_Jcodes_w_L)
print(Paid_Jcodes_w_L)



#copy the data
Unpaid_Jcodes_w_L['V1'] = Unpaid_Jcodes['V1']
Unpaid_Jcodes_w_L['ClaimNumber'] = Unpaid_Jcodes['ClaimNumber']
Unpaid_Jcodes_w_L['ClaimLineNumber'] = Unpaid_Jcodes['ClaimLineNumber']
Unpaid_Jcodes_w_L['MemberID'] = Unpaid_Jcodes['MemberID']
Unpaid_Jcodes_w_L['ProviderID'] = Unpaid_Jcodes['ProviderID']
Unpaid_Jcodes_w_L['LineOfBusinessID'] = Unpaid_Jcodes['LineOfBusinessID']
Unpaid_Jcodes_w_L['RevenueCode'] = Unpaid_Jcodes['RevenueCode']
Unpaid_Jcodes_w_L['ServiceCode'] = Unpaid_Jcodes['ServiceCode']
Unpaid_Jcodes_w_L['PlaceOfServiceCode'] = Unpaid_Jcodes['PlaceOfServiceCode']
Unpaid_Jcodes_w_L['ProcedureCode'] = Unpaid_Jcodes['ProcedureCode']
Unpaid_Jcodes_w_L['DiagnosisCode'] = Unpaid_Jcodes['DiagnosisCode']
Unpaid_Jcodes_w_L['ClaimChargeAmount'] = Unpaid_Jcodes['ClaimChargeAmount']
Unpaid_Jcodes_w_L['DenialReasonCode'] = Unpaid_Jcodes['DenialReasonCode']
Unpaid_Jcodes_w_L['PriceIndex'] = Unpaid_Jcodes['PriceIndex']
Unpaid_Jcodes_w_L['InOutOfNetwork'] = Unpaid_Jcodes['InOutOfNetwork']
Unpaid_Jcodes_w_L['ReferenceIndex'] = Unpaid_Jcodes['ReferenceIndex']
Unpaid_Jcodes_w_L['PricingIndex'] = Unpaid_Jcodes['PricingIndex']
Unpaid_Jcodes_w_L['CapitationIndex'] = Unpaid_Jcodes['CapitationIndex']
Unpaid_Jcodes_w_L['SubscriberPaymentAmount'] = Unpaid_Jcodes['SubscriberPaymentAmount']
Unpaid_Jcodes_w_L['ProviderPaymentAmount'] = Unpaid_Jcodes['ProviderPaymentAmount']
Unpaid_Jcodes_w_L['GroupIndex'] = Unpaid_Jcodes['GroupIndex']
Unpaid_Jcodes_w_L['SubscriberIndex'] = Unpaid_Jcodes['SubscriberIndex']
Unpaid_Jcodes_w_L['SubgroupIndex'] = Unpaid_Jcodes['SubgroupIndex']
Unpaid_Jcodes_w_L['ClaimType'] = Unpaid_Jcodes['ClaimType']
Unpaid_Jcodes_w_L['ClaimSubscriberType'] = Unpaid_Jcodes['ClaimSubscriberType']
Unpaid_Jcodes_w_L['ClaimPrePrinceIndex'] = Unpaid_Jcodes['ClaimPrePrinceIndex']
Unpaid_Jcodes_w_L['ClaimCurrentStatus'] = Unpaid_Jcodes['ClaimCurrentStatus']
Unpaid_Jcodes_w_L['NetworkID'] = Unpaid_Jcodes['NetworkID']
Unpaid_Jcodes_w_L['AgreementID'] = Unpaid_Jcodes['AgreementID']

#And assign the target label 
Unpaid_Jcodes_w_L['IsUnpaid'] = 1

#Look at the data..
print(Unpaid_Jcodes_w_L)


# Do the same for the Paid set.

#copy the data
Paid_Jcodes_w_L['V1'] = Paid_Jcodes['V1']
Paid_Jcodes_w_L['ClaimNumber'] = Paid_Jcodes['ClaimNumber']
Paid_Jcodes_w_L['ClaimLineNumber'] = Paid_Jcodes['ClaimLineNumber']
Paid_Jcodes_w_L['MemberID'] = Paid_Jcodes['MemberID']
Paid_Jcodes_w_L['ProviderID'] = Paid_Jcodes['ProviderID']
Paid_Jcodes_w_L['LineOfBusinessID'] = Paid_Jcodes['LineOfBusinessID']
Paid_Jcodes_w_L['RevenueCode'] = Paid_Jcodes['RevenueCode']
Paid_Jcodes_w_L['ServiceCode'] = Paid_Jcodes['ServiceCode']
Paid_Jcodes_w_L['PlaceOfServiceCode'] = Paid_Jcodes['PlaceOfServiceCode']
Paid_Jcodes_w_L['ProcedureCode'] = Paid_Jcodes['ProcedureCode']
Paid_Jcodes_w_L['DiagnosisCode'] = Paid_Jcodes['DiagnosisCode']
Paid_Jcodes_w_L['ClaimChargeAmount'] = Paid_Jcodes['ClaimChargeAmount']
Paid_Jcodes_w_L['DenialReasonCode'] = Paid_Jcodes['DenialReasonCode']
Paid_Jcodes_w_L['PriceIndex'] = Paid_Jcodes['PriceIndex']
Paid_Jcodes_w_L['InOutOfNetwork'] = Paid_Jcodes['InOutOfNetwork']
Paid_Jcodes_w_L['ReferenceIndex'] = Paid_Jcodes['ReferenceIndex']
Paid_Jcodes_w_L['PricingIndex'] = Paid_Jcodes['PricingIndex']
Paid_Jcodes_w_L['CapitationIndex'] = Paid_Jcodes['CapitationIndex']
Paid_Jcodes_w_L['SubscriberPaymentAmount'] = Paid_Jcodes['SubscriberPaymentAmount']
Paid_Jcodes_w_L['ProviderPaymentAmount'] = Paid_Jcodes['ProviderPaymentAmount']
Paid_Jcodes_w_L['GroupIndex'] = Paid_Jcodes['GroupIndex']
Paid_Jcodes_w_L['SubscriberIndex'] = Paid_Jcodes['SubscriberIndex']
Paid_Jcodes_w_L['SubgroupIndex'] = Paid_Jcodes['SubgroupIndex']
Paid_Jcodes_w_L['ClaimType'] = Paid_Jcodes['ClaimType']
Paid_Jcodes_w_L['ClaimSubscriberType'] = Paid_Jcodes['ClaimSubscriberType']
Paid_Jcodes_w_L['ClaimPrePrinceIndex'] = Paid_Jcodes['ClaimPrePrinceIndex']
Paid_Jcodes_w_L['ClaimCurrentStatus'] = Paid_Jcodes['ClaimCurrentStatus']
Paid_Jcodes_w_L['NetworkID'] = Paid_Jcodes['NetworkID']
Paid_Jcodes_w_L['AgreementID'] = Paid_Jcodes['AgreementID']

#And assign the target label 
Paid_Jcodes_w_L['IsUnpaid'] = 0

#Look at the data..
print(Paid_Jcodes_w_L)


#now combine the rows together (axis=0)
Jcodes_w_L = np.concatenate((Unpaid_Jcodes_w_L, Paid_Jcodes_w_L), axis=0)

#check the shape
Jcodes_w_L.shape


#44961 + 6068

#look at the transition between the rows around row 44961
print(Jcodes_w_L[44955:44968])

#We need to shuffle the rows before using classifers in sklearn
Jcodes_w_L.dtype.names

print(Jcodes_w_L[44957:44965])

# Apply the random shuffle
np.random.shuffle(Jcodes_w_L)


print(Jcodes_w_L[44957:44965])

#Columns are still in the right order
Jcodes_w_L

#Now get in the form for sklearn
Jcodes_w_L.dtype.names


label =  'IsUnpaid'



# Removed V1 and Diagnosis Code
cat_features = ['ProviderID','LineOfBusinessID','RevenueCode', 
                'ServiceCode', 'PlaceOfServiceCode', 'ProcedureCode',
                'DenialReasonCode','PriceIndex', 'InOutOfNetwork', 'ReferenceIndex', 
                'PricingIndex', 'CapitationIndex', 'ClaimSubscriberType',
                'ClaimPrePrinceIndex', 'ClaimCurrentStatus', 'NetworkID',
                'AgreementID', 'ClaimType']

numeric_features = ['ClaimNumber', 'ClaimLineNumber', 'MemberID', 
                    'ClaimChargeAmount',
                    'SubscriberPaymentAmount', 'ProviderPaymentAmount',
                    'GroupIndex', 'SubscriberIndex', 'SubgroupIndex']


#convert features to list, then to np.array 
# This step is important for sklearn to use the data from the structured NumPy array

#separate categorical and numeric features
Mcat = np.array(Jcodes_w_L[cat_features].tolist())
Mnum = np.array(Jcodes_w_L[numeric_features].tolist())

L = np.array(Jcodes_w_L[label].tolist())


ohe = OneHotEncoder(sparse=False) #Easier to read
Mcat = ohe.fit_transform(Mcat)

#If you want to go back to the original mappings.
ohe.inverse_transform(Mcat)
ohe_features = ohe.get_feature_names(cat_features).tolist()

#What is the shape of the matrix categorical columns that were OneHotEncoded?   
Mcat.shape
Mnum.shape


#You can subset if you have memory issues.
#You might be able to decide which features are useful and remove some of them before the one hot encoding step

#If you want to recover from the memory error then subset
#Mcat = np.array(Jcodes_w_L[cat_features].tolist())

Mcat_subset = Mcat[0:10000]
Mcat_subset.shape

Mnum_subset = Mnum[0:10000]
Mnum_subset.shape

L_subset = L[0:10000]

# Uncomment if you need to run again from a subset.


#What is the size in megabytes before subsetting?
# https://www.w3resource.com/python-exercises/numpy/python-numpy-exercise-33.php
# and using base2 (binary conversion), https://www.gbmb.org/bytes-to-mb
print("%d Megabytes" % ((Mcat.size * Mcat.itemsize)/1048576))
print("%d Megabytes" % ((Mnum.size * Mnum.itemsize)/1048576))

#What is the size in megabytes after subsetting?
print("%d Megabytes" % ((Mcat_subset.size * Mcat_subset.itemsize)/1048576)) 
print("%d Megabytes" % ((Mnum_subset.size * Mnum_subset.itemsize)/1048576))

#Concatenate the columns
M = np.concatenate((Mcat, Mnum), axis=1)
#M = np.concatenate((Mcat_subset, Mnum_subset), axis=1)


L = Jcodes_w_L[label].astype(int)

# Match the label rows to the subset matrix rows.
#L = L[0:10000]

M.shape
L.shape

# Now you can use your DeathToGridsearch code.


n_folds = 5

#EDIT: pack the arrays together into "data"
data = (M,L,n_folds)



def run(a_clf, data, clf_hyper={}):
  M, L, n_folds = data #EDIT: unpack the "data" container of arrays
  kf = KFold(n_splits=n_folds) # JS: Establish the cross validation 
  ret = {} # JS: classic explicaiton of results
  
  for ids, (train_index, test_index) in enumerate(kf.split(M, L)): 
    
    clf = a_clf(**clf_hyper) 
            
    clf.fit(M[train_index], L[train_index])                             
    
    pred = clf.predict(M[test_index])         
    
    ret[ids]= {'clf': clf,                   
               'train_index': train_index,
               'test_index': test_index,
               'accuracy': accuracy_score(L[test_index], pred)}    
  return ret




def populateClfAccuracyDict(results):
    for key in results:
        k1 = results[key]['clf'] 
        v1 = results[key]['accuracy']
        k1Test = str(k1) 
                        
        #String formatting            
        k1Test = k1Test.replace('            ',' ') # remove large spaces from string
        k1Test = k1Test.replace('          ',' ')
        
        #Then check if the string value 'k1Test' exists as a key in the dictionary
        if k1Test in clfsAccuracyDict:
            clfsAccuracyDict[k1Test].append(v1) #append the values to create an array (techically a list) of values
        else:
            clfsAccuracyDict[k1Test] = [v1] #create a new key (k1Test) in clfsAccuracyDict with a new value, (v1)            
        
            

def myHyperSetSearch(clfsList,clfDict):
    #hyperSet = {}
    for clf in clfsList:
    
    #I need to check if values in clfsList are in clfDict
        clfString = str(clf)
        #print("clf: ", clfString)
        
        for k1, v1 in clfDict.items(): # go through the inner dictionary of hyper parameters
            #Nothing to do here, we need to get into the inner nested dictionary.
            if k1 in clfString:
                #allows you to do all the matching key and values
                k2,v2 = zip(*v1.items()) # explain zip (https://docs.python.org/3.3/library/functions.html#zip)
                for values in product(*v2): #for the values in the inner dictionary, get their unique combinations from product()
                    hyperSet = dict(zip(k2, values)) # create a dictionary from their values
                    results = run(clf, data, hyperSet) # pass the clf and dictionary of hyper param combinations to run; get results
                    populateClfAccuracyDict(results) # populate clfsAccuracyDict with results
 



clfsList = [RandomForestClassifier] 

clfDict = {'RandomForestClassifier': {"min_samples_split": [2,3,4], 
                                      "n_jobs": [1,2,3]}}#,

clfs_1 = { 
           'RandomForestClassifier' : {RandomForestClassifier: 
                                       {'max_depth': [5, 6, 7],
                                        'min_samples_split': [50, 75, 100],
                                        'max_features': ["auto", "sqrt", "log2"]}},
           'GradientBoostingClassifier' : {GradientBoostingClassifier : 
                                           {'learning_rate':[0.05, 0.075, 1],
                                            'min_samples_split': [50, 75, 100],
                                            'max_depth':[8,9]}},
           'KNeighborsClassifier' : {KNeighborsClassifier : 
                                     {'n_neighbors': [4, 6, 8],
                                      'leaf_size': [25, 30, 35],
                                      'algorithm': ['ball_tree', 'kd_tree', 'brute']}}
          }
                                     

                   
#Declare empty clfs Accuracy Dict to populate in myHyperSetSearch     
clfsAccuracyDict = {}

#Run myHyperSetSearch
myHyperSetSearch(clfsList,clfDict)    

print(clfsAccuracyDict)

               


