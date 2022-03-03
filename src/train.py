import os 
import config
import model_dispatcher
import features
import joblib
import pandas as pd
import numpy as np
from sklearn import metrics 
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')  # "error", "ignore", "always", "default", "module" or "once"
from imblearn.over_sampling import SMOTE



def run(fold,model):
    
    df = pd.read_csv(config.TRAINING_FILE)
    df=df[features.features]

    
    ## Scale numerical variables
    standardScaler = StandardScaler()
    df[features.numerical_features] = standardScaler.fit_transform(df[features.numerical_features])

    

    # ### one hot encode categorical variables
    # for col in features.ohe_features:
    #     df.loc[:,col]=df[col].astype(str).fillna("NONE")

    # df=pd.get_dummies(df,columns=features.ohe_features,drop_first=True)

    
    
    ###Label encoding
    for col in features.labelencode_features:
        df.loc[:,col]=df[col].astype(str).fillna("NONE")

    for col in features.labelencode_features:
        lbl=preprocessing.LabelEncoder()
        lbl.fit(df[col])
        df.loc[:,col]=lbl.transform(df[col])
    
    
    df_train = df[df.kfold != fold].reset_index(drop=True)
    df_valid = df[df.kfold == fold].reset_index(drop=True)

    
    x_train = df_train.drop(features.target_feature, axis=1)
    x_train = x_train.drop('kfold', axis=1)
    y_train = df_train[features.target_feature]
    x_valid = df_valid.drop(features.target_feature, axis=1)
    x_valid=x_valid.drop('kfold',axis=1)
    y_valid = df_valid[features.target_feature]

    
    
    clf = model_dispatcher.models[model]
    
    clf.fit(x_train, y_train)
    
    preds = clf.predict(x_valid)
    

    ### Model evaluation
    accuracy = metrics.accuracy_score(y_valid, preds)
    precision = metrics.precision_score(y_valid, preds)
    recall = metrics.recall_score(y_valid, preds)
    f1 = metrics.f1_score(y_valid, preds)
    auc = metrics.roc_auc_score(y_valid, preds)
    log_loss = metrics.log_loss(y_valid, preds)


    print(choose_model[i],f"Fold={fold},Accuracy={accuracy*100:.2f}%,Precision={precision*100:.2f}%,Recall={recall*100:.2f}%,f1={f1*100:.2f}%,AUC={auc*100:.2f}%,logloss={log_loss:.2f}")  
    
    joblib.dump( clf,os.path.join(config.MODEL_OUTPUT, f"{choose_model[i]}_{fold}.bin") )
    
    

if __name__ == "__main__": 
    
    choose_model = ["rf","xgb"]
    #choose_model = ["lr","dtg","dte","rf","gb","lgb","cb","xgb","nb","knn","svm"]
    for i in range(0,len(choose_model)):
        print('\n',choose_model[i])

        for f in range(0,5):
            run(fold=f,model=choose_model[i])
                  
        
        
    