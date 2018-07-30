from metaheuristic_algorithms.firefly_algorithm import FireflyAlgorithm


from sklearn.feature_extraction.text import CountVectorizer 
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
import re
from sklearn.svm import SVC

from sklearn.model_selection import train_test_split

from sklearn.model_selection import StratifiedKFold
import sys


# 1. INIT DATASET==================================================

#Menghitung jumlah data negatif, netral, dan positif
with open('negatif.txt') as f:
    contentnegatif = f.readlines()

contentnegatif = [x.strip() for x in contentnegatif]
jumlah_negatif = len(contentnegatif)

with open('positif.txt') as f:
    contentpositif = f.readlines()

contentpositif = [x.strip() for x in contentpositif] #untuk memisakhan data dri fileteks berdasarkan spasi
jumlah_positif = len(contentpositif)


#Memberi label pada tiap data
label=[]
for i in range(jumlah_negatif):
    label.append(0) 

for i in range(jumlah_positif):
    label.append(1)



X=contentnegatif+contentpositif
y=label

X, X_test, y, y_test = train_test_split(X, y, test_size=0.20)


# PREPROCESSING
# Slangword
swlist = eval(open("slangwords.txt").read())#Membuka dictionasry slangword
pattern = re.compile(r'\b( ' + '|'.join (swlist.keys())+r')\b')#untuk mencari pola kara contoh tdk menjadi tidak

content = []
for comment in X:
   filteredSlang = pattern.sub(lambda x: swlist[x.group()],comment)#mengganti slangwsd berdasarkan pola kata yg telah ditentukan 
   content.append(filteredSlang.lower())

# hasil proses slangword
X= content #berisi data yg sudah dicocokan dgn slangword

# Stopword
with open('stopword_list_tala.txt') as s:
   stopword = s.readlines()
stopword = [x.strip() for x in stopword]

wordFiltered = []
for w in X:
   filtered_words = [word for word in w.split() if word not in stopword] #menyimpan kata2 yg tidak ada di stopword
   wordFiltered.append(' '.join(filtered_words)) #merubah list menjadi string
X=wordFiltered #berisi data yg telah di slangword dan stop word
# END PREP



# INIT DATASET END==================================================







# 2. ALL FUNC DEFINITION============================================

REGEX = re.compile(r"\s")
def tokenize(text):
    return [tok.strip().lower() for tok in REGEX.split(text)]


def Train(subset_X,subset_y,pars):    
    text_clf = Pipeline([('vect', CountVectorizer(tokenizer=tokenize)),
    ('tfidf', TfidfTransformer()),
    ('clf', SVC(C=pars[0], gamma=pars[1])),
    ])
    clf=text_clf.fit(subset_X,subset_y)
    return clf


def Testing(clf,subset_X,subset_y):
    expected = subset_y
    predicted = clf.predict(subset_X)
    acc=accuracy_score(expected, predicted)

    # cm=metrics.confusion_matrix(expected, predicted).astype(int)
    print("Accuracy "+str(acc))
    return acc

def SearchingParameters(train_X,train_y,test_X,test_y,pars):
    cl=Train(train_X,train_y,pars)
    print ("Parameters : "+str(pars))
    return Testing(cl,test_X,test_y)


# ALL FUNC DEFINITION END=================================================






# 3. START SEARCHING==================================================

C_PARAMETER=0.5
GAMMA_PARAMETER=0.5


def KFOLDS(kf):
    skf = StratifiedKFold(n_splits=kf)
    folds=skf.split(X, y)

    fold=1
    REPORT=""

    for train_index, test_index in folds:
        print ("\n\nFOLDS="+str(fold))
        

        
        training_X=[]
        training_y=[]
        testing_X=[]
        testing_y=[]


        for i in train_index:
            training_X.append(X[i])
            training_y.append(y[i])

        for i in test_index:
            testing_X.append(X[i])
            testing_y.append(y[i])


        Akurasi=SearchingParameters(training_X,training_y,testing_X,testing_y,[C_PARAMETER,GAMMA_PARAMETER])

        REPORT+="FOLD ke"+str(fold)+"\n"
        REPORT+="Akurasi : "+str(Akurasi)+"\n"
        fold+=1


    # TESTING
    Akurasi2=SearchingParameters(training_X,training_y,testing_X,testing_y,[C_PARAMETER,GAMMA_PARAMETER])
    REPORT+="TESTING\n"
    REPORT+="Akurasi : "+str(Akurasi2)+"\n"

    text_file = open("Output_manual.txt", "w")
    text_file.write(REPORT)



KFOLDS(3)

# START SEARCHING END==================================================
