from sklearn.linear_model import LogisticRegression
from sklearn import tree
from sklearn import ensemble 
from sklearn.ensemble import GradientBoostingClassifier
import lightgbm
from lightgbm import LGBMClassifier
import catboost
from catboost import CatBoostClassifier
import xgboost
from xgboost import XGBClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

models = {"lr":LogisticRegression(max_iter=5000,random_state=46),
        "dtg": tree.DecisionTreeClassifier(criterion="gini",random_state=46),
        "dte": tree.DecisionTreeClassifier(criterion="entropy",random_state=46),
        "rf": ensemble.RandomForestClassifier(random_state=46),
        "gb": GradientBoostingClassifier(random_state=46),
        "lgb": LGBMClassifier(random_state=46),
        "cb": CatBoostClassifier(verbose=0,n_estimators=100,random_state=46),
        "xgb": XGBClassifier(use_label_encoder=False,objective= 'binary:logistic',eval_metric='logloss',random_state=46),
        "nb":GaussianNB(),
        "knn":KNeighborsClassifier(n_neighbors=15),
        "svm":SVC(kernel="linear",random_state=46)}



