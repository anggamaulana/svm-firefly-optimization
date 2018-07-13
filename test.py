from metaheuristic_algorithms.firefly_algorithm import FireflyAlgorithm


from sklearn.feature_extraction.text import CountVectorizer 
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
import re
from sklearn.svm import SVC


from sklearn.model_selection import StratifiedKFold



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


number_of_variables = 2
objective = "maximization"

# firefly_algorithm 
number_of_fireflies=5 
maximun_generation=5 
randomization_parameter_alpha=0.2 
absorption_coefficient_gamma=1.0

nilai_min_pars=[0.1,0.0]
nilai_max_pars=[1.0,1.0]
nilai_awal_pars=[0.3,0.001]


from metaheuristic_algorithms.function_wrappers.abstract_wrapper import AbstractWrapper

class PencarianSVMParameter(AbstractWrapper):

    def __init__(self,subset_X_train,subset_y_train,subset_X_test,subset_y_test):
        self.subset_X_train=subset_X_train
        self.subset_y_train=subset_y_train

        self.subset_X_test=subset_X_test
        self.subset_y_test=subset_y_test


    def maximum_decision_variable_values(self):
        return nilai_max_pars

    def minimum_decision_variable_values(self):
        return nilai_min_pars

    def objective_function_value(self, decision_variable_values):
        return SearchingParameters(self.subset_X_train,self.subset_y_train,self.subset_X_test,self.subset_y_test,decision_variable_values)

    def initial_decision_variable_value_estimates(self):
        return nilai_awal_pars




# ALL FUNC DEFINITION END=================================================






# 3. START SEARCHING==================================================

def SearchFireFly(subset_X_train,subset_y_train,subset_X_test,subset_y_test):

  fc = PencarianSVMParameter(subset_X_train,subset_y_train,subset_X_test,subset_y_test)

  firefly_algorithm = FireflyAlgorithm(fc, number_of_variables, objective)

  result = firefly_algorithm.search(
      number_of_fireflies=number_of_fireflies, 
      maximun_generation=maximun_generation, 
      randomization_parameter_alpha=randomization_parameter_alpha, 
      absorption_coefficient_gamma=absorption_coefficient_gamma)

  print("Parameter terbaik : "+str(result["best_decision_variable_values"])) 
  print("Akurasi terbaik : "+str(result["best_objective_function_value"]) )  

def KFOLDS(kf):
    skf = StratifiedKFold(n_splits=kf)
    folds=skf.split(X, y)

    fold=1

    for train_index, test_index in folds:
        print ("\n\nFOLDS="+str(fold))
        fold+=1

        
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


        # SearchingParameters(training_X,training_y,testing_X,testing_y,[0.5])

        SearchFireFly(training_X,training_y,testing_X,testing_y)


KFOLDS(5)

# START SEARCHING END==================================================
