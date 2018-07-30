from metaheuristic_custom.firefly_algorithm import FireflyAlgorithm


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
    # print("Accuracy "+str(acc))
    return acc

def SearchingParameters(train_X,train_y,test_X,test_y,pars):
    cl=Train(train_X,train_y,pars)
    # print ("Parameters : "+str(pars))
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


from metaheuristic_custom.function_wrappers.abstract_wrapper import AbstractWrapper

class PencarianSVMParameter(AbstractWrapper):

    def __init__(self,subset_X_train,subset_y_train,subset_X_test,subset_y_test):
        self.subset_X_train=subset_X_train
        self.subset_y_train=subset_y_train

        self.subset_X_test=subset_X_test
        self.subset_y_test=subset_y_test
        self.REPORT=""
        self.LOGGING=[]


    def maximum_decision_variable_values(self):
        return nilai_max_pars

    def minimum_decision_variable_values(self):
        return nilai_min_pars

    def objective_function_value(self, decision_variable_values):
        return SearchingParameters(self.subset_X_train,self.subset_y_train,self.subset_X_test,self.subset_y_test,decision_variable_values)

    def initial_decision_variable_value_estimates(self):
        return nilai_awal_pars

    def logging(self,datas):
        
        for k,v in datas.items():
            nilai=str(k)+" : "+str(v)+"\n"
            self.REPORT+=nilai
            # print(nilai)

        self.REPORT+="\n\n"
        self.LOGGING.append(datas)
        # print("\n")

    def report(self):
        return self.REPORT
    




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

  print(fc.report())

  print("Parameter terbaik : "+str(result["best_decision_variable_values"])) 
  print("Akurasi terbaik : "+str(result["best_objective_function_value"]) ) 
  return result,fc.report(),fc.LOGGING 

def KFOLDS(kf):
    skf = StratifiedKFold(n_splits=kf)
    folds=skf.split(X, y)

    fold=1

    REPORT=""
    REPORT_EXCEL=""
    best_gamma=0
    best_c=0
    best_acc=0
    

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


        # SearchingParameters(training_X,training_y,testing_X,testing_y,[0.5])
        # Firefly untuk cross validation
        res,report,logs=SearchFireFly(training_X,training_y,testing_X,testing_y)

        if res["best_objective_function_value"]>best_acc:
            best_acc=res["best_objective_function_value"]
            best_c=res["best_decision_variable_values"][0]
            best_gamma=res["best_decision_variable_values"][1]

        
        for log in logs:
            REPORT_EXCEL+=str(fold)+","
            REPORT_EXCEL+=str(log["generation"])+","
            REPORT_EXCEL+=str(log["best_decision_variable_values"][0])+","
            REPORT_EXCEL+=str(log["best_decision_variable_values"][1])+","
            REPORT_EXCEL+=str(log["best_objective_function_value"])+"\n"
        


        REPORT+="FOLD ke"+str(fold)+"\n"
        REPORT+= report
        REPORT+="Parameter terbaik : "+str(res["best_decision_variable_values"])+"\n"
        REPORT+="Akurasi terbaik : "+str(res["best_objective_function_value"])+"\n\n\n\n"
        fold+=1


    # TESTING
    # res2,report=SearchFireFly(X,y,X_test,y_test)
    res2=SearchingParameters(X,y,X_test,y_test,[best_c,best_gamma])
    REPORT+="TESTING\n"
    REPORT+="Parameter terbaik : "+str(best_c)+","+str(best_gamma)+"\n"
    REPORT+="Akurasi : "+str(res2)+"\n"

    text_file = open("Output_firefly.txt", "w")
    text_file.write(REPORT)

    csv_file = open("csv_firefly.csv", "w")
    csv_file.write(REPORT_EXCEL)



KFOLDS(3)

# START SEARCHING END==================================================
