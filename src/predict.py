from asyncio.base_tasks import _task_get_stack
from operator import index
import os
import config

from pandas.core.algorithms import mode 
import joblib
import pandas as pd
import numpy as np
from sklearn import metrics 
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')  # "error", "ignore", "always", "default", "module" or "once"
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from imblearn.over_sampling import SMOTE
import features
import time
from pandas.io.parsers import read_csv
from numpy import sqrt
from numpy import argmax
from sklearn.metrics import roc_curve
import data_prep
import calculate_attrition
    

model='rf'
trained_model0=joblib.load(f"../models/{model}_0.bin")
trained_model1=joblib.load(f"../models/{model}_1.bin")
trained_model2=joblib.load(f"../models/{model}_2.bin")
trained_model3=joblib.load(f"../models/{model}_3.bin")
trained_model4=joblib.load(f"../models/{model}_4.bin")

model_chain='xgb'
trained_model5=joblib.load(f"../models/{model_chain}_0.bin")
trained_model6=joblib.load(f"../models/{model_chain}_1.bin")
trained_model7=joblib.load(f"../models/{model_chain}_2.bin")
trained_model8=joblib.load(f"../models/{model_chain}_3.bin")
trained_model9=joblib.load(f"../models/{model_chain}_4.bin")


if __name__=='__main__':
    start_time=time.time()
    prediction_input = pd.read_csv(config.PREDICTION_INPUT_FILE)
    df=prediction_input
    df['kfold']=0
    df=df[features.features]
    df = df.drop('kfold', axis=1)
    input_describe=df.describe()
    input_describe_cat=df.describe(include=['object'])
    input_features=df.isnull().sum().sort_values(ascending=False)/len(df)
    input_features=pd.DataFrame(input_features,columns=['% Missing values'])
    
    def scale_and_convert(df):

        ## Scale numerical variables
        standardScaler = StandardScaler()
        df[features.numerical_features] = standardScaler.fit_transform(df[features.numerical_features])

        ###Label encoding
        for col in features.labelencode_features:
            df.loc[:,col]=df[col].astype(str).fillna("NONE")

        for col in features.labelencode_features:
            lbl=preprocessing.LabelEncoder()
            lbl.fit(df[col])
            df.loc[:,col]=lbl.transform(df[col])
        return df

    
    input=scale_and_convert(df)
    input = input.drop(features.target_feature, axis=1)
    preds_proba = (trained_model0.predict_proba(input)[:,1]+trained_model1.predict_proba(input)[:,1]+trained_model2.predict_proba(input)[:,1]+
    trained_model3.predict_proba(input)[:,1]+trained_model4.predict_proba(input)[:,1])/5
    preds_proba1=pd.DataFrame(preds_proba).rename(columns={0:"Prediction Probability_rf"})
    predictions=pd.concat([prediction_input,preds_proba1],axis=1)
    predictions=predictions.sort_values(by="Prediction Probability_rf",ascending=False)
    predictions['Decile']=pd.qcut(predictions['Prediction Probability_rf'],10,labels=['10','9','8','7','6','5','4','3','2','1'])
    

    prediction_input_chain=predictions[predictions['Decile']!='1'].reset_index(drop=True)
    input_chain=prediction_input_chain[features.features]
    input_chain=scale_and_convert(input_chain)
    input_chain = input_chain.drop(features.target_feature, axis=1)
    input_chain = input_chain.drop('kfold', axis=1)
    preds_proba_chain = (trained_model5.predict_proba(input_chain)[:,1]+trained_model6.predict_proba(input_chain)[:,1]+trained_model7.predict_proba(input_chain)[:,1]+
    trained_model8.predict_proba(input_chain)[:,1]+trained_model9.predict_proba(input_chain)[:,1])/5
    preds_proba2=pd.DataFrame(preds_proba_chain).rename(columns={0:"Prediction Probability_xgb"})
    predictions_chain=pd.concat([prediction_input_chain,preds_proba2],axis=1)
    
    predictions=pd.merge(predictions,predictions_chain[['Emp.No.','Prediction Probability_xgb']],on='Emp.No.',how='left')
   
    

    predictions['Prediction Probability']=predictions['Prediction Probability_rf']
    predictions.loc[predictions['Decile']!='1','Prediction Probability']=0.50*predictions['Prediction Probability_rf']+0.50*predictions['Prediction Probability_xgb']


    
    predictions['Decile']=pd.qcut(predictions['Prediction Probability'],10,labels=['10','9','8','7','6','5','4','3','2','1'])

    predictions['Prediction Bucket']=0
    predictions['Prediction Bucket']=pd.qcut(predictions['Prediction Probability'],[0,0.85,1],labels=["Low Risk","High Risk"])

    predictions=predictions.drop(['Attrition','kfold'],axis=1)

    

    predictions=predictions[['Emp.No.','Prediction Probability','Decile','Prediction Bucket']]
    predictions['Intervention']="Not Recommended"
    predictions.loc[predictions['Prediction Bucket']=="High Risk",'Intervention']="Strictly Recommended"
    
    predictions=predictions.sort_values(by="Prediction Probability",ascending=False)

    
    predictions.to_csv(config.PREDICTION_OUTPUT_FILE,index=False)

    input_describe=prediction_input.describe()

    
    
    
    importances = pd.DataFrame(data={'Attribute': input.columns,'Importance': (trained_model0.feature_importances_+trained_model1.feature_importances_+trained_model2.feature_importances_+trained_model3.feature_importances_+trained_model4.feature_importances_)/5})
    importances = importances.sort_values(by='Importance', ascending=False)
    print(importances,'\n')

    oot=pd.read_excel(config.OOT,index_col=0)
    oot=pd.merge(oot,predictions[['Emp.No.','Prediction Probability','Decile','Prediction Bucket']],on='Emp.No.',how='left')
    #print(oot,'\n')
    #print(pd.crosstab(oot['Prediction Bucket'],oot['Resignation Date'],margins=True),'\n')
    #print(oot['Prediction Bucket'].value_counts())

    total_pred=predictions[predictions['Prediction Bucket']=="High Risk"]
    actual_res=oot[(oot['Prediction Bucket']=="High Risk")|(oot['Prediction Bucket']=="Low Risk")]
    correct_pred=oot[(oot['Prediction Bucket']=="High Risk")]
    

    print("\nTotal Employees:",f"{len(predictions)}")
    print(f'Total Resignations Predicted ={len(total_pred)}')
    print(f'Actual Resignations ={len(actual_res)}')
    print(f'Resignations correctly predicted ={len(correct_pred)}')
    print(f'Recall ={len(correct_pred)/(len(actual_res)+0.001)*100:.2f}%')
    print(f'Precision ={len(correct_pred)/(len(total_pred))*100:.2f}%')
    
    Recall =len(correct_pred)/(len(actual_res)+0.001)*100
    Precision =len(correct_pred)/len(total_pred)*100
    F2=5/((4/(Precision+0.001))+(1/(Recall+0.001)))
    print(f'F2={F2:.2f}%\n')
    
 

    model_monitoring = {'Timestamp':[pd.datetime.now().strftime("%Y-%m-%d %H:%M:%S")],'Model':[model+"-"+model_chain],'Total Employees':[len(predictions)],'Total Resignations Predicted':[len(total_pred)],
    'Actual Resignations':[len(actual_res)],'Resignations correctly predicted':[len(correct_pred)],'Recall':[Recall],'Precision':[Precision],'F2':[F2]}
    model_monitoring=pd.DataFrame(model_monitoring)

    predictions=pd.merge(predictions,oot[['Emp.No.','Resignation Date']],on='Emp.No.',how='left')

    end_time=time.time()
    
    importance_monitoring=pd.DataFrame(importances)

    processing_time=end_time-start_time
    processing_time = {'Timestamp':[pd.datetime.now().strftime("%Y-%m-%d %H:%M:%S")],'Metric':['Processing Time'],'Time in Seconds':[processing_time]}
    processing_time=pd.DataFrame(processing_time)
    
    
    writer = pd.ExcelWriter("../output/log/sales_prediction_output_{}.xlsx".format(pd.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), engine='xlsxwriter'))


    # Write each dataframe to a different worksheet.
    processing_time.to_excel(writer, sheet_name='System Monitoring',index=False)
    
    input_features.to_excel(writer, sheet_name='Model Metrics Monitoring',startrow=0,startcol=0)
    input_describe.to_excel(writer, sheet_name='Model Metrics Monitoring',startrow=0,startcol=3)
    input_describe_cat.to_excel(writer, sheet_name='Model Metrics Monitoring',startrow=12,startcol=3)
    
    model_monitoring.to_excel(writer, sheet_name='Model Metrics Monitoring',startrow=35,startcol=0)
    importance_monitoring.to_excel(writer, sheet_name='Model Metrics Monitoring',startrow=40,startcol=0)
    predictions.to_excel(writer, sheet_name='Model Metrics Monitoring',startrow=70,startcol=0)
    

    predictions.to_excel(writer, sheet_name='Business Metrics Monitoring',startrow=10,startcol=0,index=False)
    predictions['Resignation Date']=predictions['Resignation Date'].dt.year.astype(str)+'-'+predictions['Resignation Date'].dt.month.astype(str)
    predictions.loc[predictions['Resignation Date']=='nan-nan','Resignation Date']='active'
    predictions_b=pd.crosstab(predictions['Prediction Bucket'],predictions['Resignation Date'],margins=True)
    predictions_b.to_excel(writer, sheet_name='Business Metrics Monitoring',startrow=0,startcol=0)


    sales_attrition_annualized=pd.read_excel("../output/attrition_calculation/sales_attrition_Annualized_Attrition_Percentage.xlsx")
    sales_exits=pd.read_excel("../output/attrition_calculation/sales_attrition_Exits.xlsx")
    sales_headcount=pd.read_excel("../output/attrition_calculation/sales_attrition_Closing_Headcount.xlsx")
    
    sales_attrition_annualized.to_excel(writer, sheet_name='Business Metrics Monitoring',startrow=10,startcol=9,index=False)
    sales_exits.to_excel(writer, sheet_name='Business Metrics Monitoring',startrow=20,startcol=9,index=False)
    sales_headcount.to_excel(writer, sheet_name='Business Metrics Monitoring',startrow=30,startcol=9,index=False)


    
    #PSI & CSI

    def calculate_psi(expected, actual, buckettype='bins', buckets=10, axis=0):
        '''Calculate the PSI (population stability index) across all variables
        Args:
        expected: numpy matrix of original values (Training)
        actual: numpy matrix of new values, same size as expected (Validation)
        buckettype: type of strategy for creating buckets, bins splits into even splits, quantiles splits into quantile buckets
        buckets: number of quantiles to use in bucketing variables
        axis: axis by which variables are defined, 0 for vertical, 1 for horizontal
        Returns:
        psi_values: ndarray of psi values for each variable
    
        '''

        def psi(expected_array, actual_array, buckets):
            '''Calculate the PSI for a single variable
            Args:
            expected_array: numpy array of original values
            actual_array: numpy array of new values, same size as expected
            buckets: number of percentile ranges to bucket the values into
            Returns:
            psi_value: calculated PSI value
            '''

            def scale_range (input, min, max):
                input += -(np.min(input))
                input /= np.max(input) / (max - min)
                input += min
                return input


            breakpoints = np.arange(0, buckets + 1) / (buckets) * 100

            if buckettype == 'bins':
                breakpoints = scale_range(breakpoints, np.min(expected_array), np.max(expected_array))
            elif buckettype == 'quantiles':
                breakpoints = np.stack([np.percentile(expected_array, b) for b in breakpoints])



            expected_percents = np.histogram(expected_array, breakpoints)[0] / len(expected_array)
            actual_percents = np.histogram(actual_array, breakpoints)[0] / len(actual_array)

            def sub_psi(e_perc, a_perc):
                '''Calculate the actual PSI value from comparing the values.
                Update the actual value to a very small number if equal to zero
                '''
                if a_perc == 0:
                    a_perc = 0.0001
                if e_perc == 0:
                    e_perc = 0.0001

                value = (e_perc - a_perc) * np.log(e_perc / a_perc)
                return(value)

            psi_value = np.sum(sub_psi(expected_percents[i], actual_percents[i]) for i in range(0, len(expected_percents)))

            return(psi_value)

        if len(expected.shape) == 1:
            psi_values = np.empty(len(expected.shape))
        else:
            psi_values = np.empty(expected.shape[axis])

        for i in range(0, len(psi_values)):
            if len(psi_values) == 1:
                psi_values = psi(expected, actual, buckets)
            elif axis == 0:
                psi_values[i] = psi(expected[:,i], actual[:,i], buckets)
            elif axis == 1:
                psi_values[i] = psi(expected[i,:], actual[i,:], buckets)

        return(psi_values)

    ## Calculate csi for top features


    df_exp = pd.read_csv(config.PREDICTION_INPUT_FILE)
    
    df_actual = pd.read_csv(config.PREDICTION_INPUT_FILE)


    psi_list = []
    feat=[]
    for feature in features.numerical_features:
            psi_t = calculate_psi(df_exp[feature], df_actual[feature])
            psi_list.append(psi_t)
            feat.append(feature)      


    psi_list=pd.DataFrame(psi_list,columns=["CSI"])
    feat=pd.DataFrame(feat,columns=['Feature'])
    csi=pd.concat([feat,psi_list],axis=1)
    
    
    #Calculate PSI


    pred_exp=pd.read_csv(config.PREDICTION_OUTPUT_FILE)
    pred_actual=pd.read_csv(config.PREDICTION_OUTPUT_FILE)

    overall_psi = calculate_psi(pred_exp["Prediction Probability"], pred_actual["Prediction Probability"])
    overall_psi=pd.DataFrame([overall_psi],columns=['PSI'])

    

    overall_psi.to_excel(writer, sheet_name='Model Metrics Monitoring',startrow=80,startcol=8)
    csi.to_excel(writer, sheet_name='Model Metrics Monitoring',startrow=84,startcol=8)




    #ROC Curve:
    def roc(predictions):    
        
        predictions_roc=predictions
        predictions_roc['status']=1
        predictions_roc.loc[predictions_roc['Resignation Date']=='active','status']=0
        # calculate roc curves
        fpr, tpr, thresholds = roc_curve(predictions_roc['status'], predictions_roc['Prediction Probability'])
        # calculate the g-mean for each threshold
        gmeans = sqrt(tpr * (1-fpr))
        # locate the index of the largest g-mean
        ix = argmax(gmeans)
        print('Best Threshold=%f, G-Mean=%.3f' % (thresholds[ix], gmeans[ix]))
        # plot the roc curve for the model
        plt.plot([0,1], [0,1], linestyle='--')
        plt.plot(fpr, tpr, marker='.')
        plt.scatter(fpr[ix], tpr[ix], marker='o', color='black', label='Best')
        # axis labels
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.legend()
        # show the plot
        plt.show()

    #roc(predictions)

    
    
    #KS Table

    df_ks=predictions
    df_ks['status']=1
    df_ks.loc[df_ks['Resignation Date']=='active','status']=0 
    df_ks=df_ks[['Decile','Emp.No.','status']].groupby('Decile').agg({'Emp.No.':'count','status':'sum'}).reset_index()
    df_ks=df_ks.sort_values(by='Decile',ascending=False).reset_index(drop=True)
    df_ks=df_ks.rename(columns={"Emp.No.":"Headcount","status":"Resignations"})
    df_ks['Active_Headcount']=df_ks['Headcount']-df_ks['Resignations']
    df_ks['cum_Headcount']=df_ks.Headcount.cumsum()
    df_ks['cum_Resignations']=df_ks.Resignations.cumsum()
    df_ks['cum_Active_Headcount']=df_ks.Active_Headcount.cumsum()
    df_ks['Resigned_Headcount_rate']=df_ks['Resignations']/df_ks['Headcount']
    df_ks['Active_Headcount_rate']=df_ks['Active_Headcount']/df_ks['Headcount']
    df_ks['cum_Resigned_Headcount_rate']=df_ks['cum_Resignations']/df_ks['cum_Headcount']
    df_ks['cum_Active_Headcount_rate']=df_ks['cum_Active_Headcount']/df_ks['cum_Headcount']
    df_ks['capture_rate']=df_ks['Resignations']/(df_ks['Resignations'].sum()+0.001)
    df_ks['non_capture_rate']=df_ks['Active_Headcount']/df_ks['Active_Headcount'].sum()
    df_ks['cum_capture_rate']=df_ks['capture_rate'].cumsum()
    df_ks['cum_non_capture_rate']=df_ks['non_capture_rate'].cumsum()
    df_ks['ks']=round(100*(df_ks['cum_capture_rate']-df_ks['cum_non_capture_rate']),2)

    print("KS Table",'\n',df_ks)

        
    
    df_ks.to_excel(writer, sheet_name='Model Metrics Monitoring',startrow=84,startcol=14,index=False)

    # Close the Pandas Excel writer and output the Excel file.
    writer.save()



    #Streamlit

    import streamlit as st

    my_page = st.sidebar.radio('Page Navigation', ['System Monitoring', 'Business Metrics Monitoring','Model Metrics Monitoring'])

    if my_page == 'System Monitoring':
        st.title("SYSTEM MONITORING")
        st.dataframe(processing_time)
        
        
    elif my_page == 'Business Metrics Monitoring':
        st.title('Business Metrics Monitoring')
        st.header('Snapshot')
        st.table(predictions_b)
        st.header('Overall Predictions')
        st.dataframe(predictions.astype(str))
        st.header('Attrition Details')
        st.subheader('Annualized Attrition')
        st.dataframe(sales_attrition_annualized.astype(str))
        st.subheader('Exits')
        st.table(sales_exits.astype(str))
        st.subheader('Headcount')
        st.table(sales_headcount.astype(str))
        
    else:
        st.title('Model Metrics Monitoring')
        st.header('Features')
        st.subheader('Features Missing values')
        st.bar_chart(input_features)
        st.subheader('Numerical Features Statistics')
        st.dataframe(input_describe)
        st.subheader('Categorical Features Statistics')
        st.dataframe(input_describe_cat.astype(str))
        st.header('Model Metrics')
        st.table(model_monitoring)
        st.header('Feature Importance')
        st.dataframe(importance_monitoring.astype(str))
        st.header('PSI')
        st.table(overall_psi)
        st.header('CSI')
        st.table(csi)
        st.header('KS Table')
        st.dataframe(df_ks.astype(str))
       


        
    
    
    
    
    