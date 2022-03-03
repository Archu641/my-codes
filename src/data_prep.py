import pandas as pd
import numpy as np
import datetime 
import warnings
warnings.filterwarnings(action = 'ignore')

active_employees=pd.read_excel("../input/active_employees.xlsx")
attrition=pd.read_excel("../input/attrition.xlsx")
training=pd.read_excel("../input/training.xlsx")
incentive=pd.read_excel("../input/sales_incentive.xls",index_col=0)
progression=pd.read_excel("../input/progression.xlsx")
recognition=pd.read_excel("../input/recognition.xlsx",index_col=0)
market_value=pd.read_excel("../input/market_value.xlsx",index_col=0).reset_index()

training.rename(columns={'empno':'Emp.No.', 'empname':'Employee Name', 'org_dept':'Department'},inplace=True)


incentive.rename(columns={"PERNR":'Emp.No.','BETRG':"incentive"},inplace=True)

training=training[training['sdate']<="2021-03-31"].reset_index()
incentive=incentive[incentive['BEGDA']<="2021-03-31"].reset_index()
recognition=recognition[recognition['Date']<="2021-03-31"].reset_index()
active_employees=active_employees[active_employees['Date of Join']<="2021-03-31"].reset_index()

dat=attrition[['Emp.No.', 'Employee Name',
       'Date of Birth', 'Date of Join', 'Leaving Date', 'Reason', 'Department',
       'Designation', 'Location', 'Exp. in TVSM (Years)',
       'Ext. Exp in years',  'Grade','This Grade from', 
       '2015 - App.Perf','2016 - App.Perf', '2017 - App.Perf', '2018 - App.Perf',
       '2019 - App.Perf', '2020 - App.Perf',
       'Education',  'Mode of Selection', 'Education.1', 'CTC', 'Status.1', 'Gender','Potential Rating','Role','This Role from']]

dat['Resignation Date']=dat['Leaving Date']-pd.DateOffset(months=4)


dac=active_employees[[ 'Emp.No.','Employee Name','Date of Birth', 'Date of Join','Department','Designation','Location','Exp.in TVS in Years', 'Ext. Exp in Years',
'Present Grade', 'Grade Effective.Date','PR 20 - 21', 'PR 19 - 20', 'PR 18 - 19','Education','Mode of Selection','Total CTC','Contract Type','Gender','POT 20 - 21','Role','This Role from']]


dac['Age']=dac['Date of Birth'].apply(lambda x: (pd.to_datetime("2021-03-31").toordinal()-x.toordinal())/365)

dac['This Role from']=dac['This Role from'].fillna(dac['Date of Join'])
dac['Years_in_Current_Role']=dac['This Role from'].apply(lambda x: (pd.to_datetime("2021-03-31").toordinal()-x.toordinal())/365)

dac['Grade Effective.Date']=dac['Grade Effective.Date'].fillna(dac['Date of Join'])
dac['Years_in_Current_Grade']=dac['Grade Effective.Date'].apply(lambda x: (pd.to_datetime("2021-03-31").toordinal()-x.toordinal())/365)

dac['Attrition']="No"

dac.rename(columns={'Exp.in TVS in Years':"TVS_exp_in_Years", 'Ext. Exp in Years':"External_exp_in_Years","Contract Type":"Type of employment","POT 20 - 21":"Potential Rating",
"Present Grade":"Grade",'Total CTC':"CTC",'Location':'work_location'},inplace=True)

dac['TVS_exp_in_Years']=dac['Date of Join'].apply(lambda x: (pd.to_datetime("2021-03-31").toordinal()-x.toordinal())/365)



doo=dat[dat['Resignation Date']>"2021-03-31"].reset_index(drop=True)
doo['Attrition']="No"
dat=dat[dat['Resignation Date']<="2021-03-31"].reset_index(drop=True)

dat['Age']=((dat['Resignation Date']- dat['Date of Birth'])/ np.timedelta64(1, 'D'))/365

dat['This Role from']=dat['This Role from'].fillna(dac['Date of Join'])
dat['Years_in_Current_Role']=((dat['Resignation Date']- dat['This Role from'])/ np.timedelta64(1, 'D'))/365

dat['This Grade from']=pd.to_datetime(dat['This Grade from'].replace({"00.00.0000":""}))
dat['This Grade from']=dat['This Grade from'].fillna(dat['Date of Join'])
dat['Years_in_Current_Grade']=((dat['Resignation Date']- dat['This Grade from'])/ np.timedelta64(1, 'D'))/365

dat['Education']=dat['Education'].fillna('Education.1')

dat['Attrition']="Yes"

dat.rename(columns={'Exp. in TVSM (Years)':"TVS_exp_in_Years", 'Ext. Exp in years':"External_exp_in_Years","Status.1":"Type of employment",
'2016 - App.Perf':'PR 16 - 17','2017 - App.Perf':'PR 17 - 18', '2018 - App.Perf':'PR 18 - 19', '2019 - App.Perf':"PR 19 - 20",'2020 - App.Perf':'PR 20 - 21','Location':'work_location'},inplace=True)
dat=dat[dat['Reason'].str.contains("Resignation",na=False)]


dat['TVS_exp_in_Years']=((dat['Resignation Date']- dat['Date of Join'])/ np.timedelta64(1, 'D'))/365


doo['Age']=doo['Date of Birth'].apply(lambda x: (pd.to_datetime("2021-03-31").toordinal()-x.toordinal())/365)

doo['This Role from']=doo['This Role from'].fillna(dac['Date of Join'])
doo['Years_in_Current_Role']=doo['This Role from'].apply(lambda x: (pd.to_datetime("2021-03-31").toordinal()-x.toordinal())/365)


doo['This Grade from']=pd.to_datetime(doo['This Grade from'].replace({"00.00.0000":""}))
doo['This Grade from']=doo['This Grade from'].fillna(dat['Date of Join'])
doo['Years_in_Current_Grade']=doo['This Grade from'].apply(lambda x: (pd.to_datetime("2021-03-31").toordinal()-x.toordinal())/365)
doo['Education']=doo['Education'].fillna('Education.1')


doo.rename(columns={'Exp. in TVSM (Years)':"TVS_exp_in_Years", 'Ext. Exp in years':"External_exp_in_Years","Status.1":"Type of employment",
'2016 - App.Perf':'PR 16 - 17','2017 - App.Perf':'PR 17 - 18', '2018 - App.Perf':'PR 18 - 19', '2019 - App.Perf':"PR 19 - 20",'2020 - App.Perf':'PR 20 - 21','Location':'work_location'},inplace=True)
doo=doo[doo['Reason'].str.contains("Resignation",na=False)]


doo['TVS_exp_in_Years']=doo['Date of Join'].apply(lambda x: (pd.to_datetime("2021-03-31").toordinal()-x.toordinal())/365)



df=pd.concat([dac,dat,doo]).reset_index(drop=True)

df['CTC'].replace(0,np.nan,inplace=True)
df['CTC'].fillna(df.groupby('Grade')['CTC'].transform('median'),inplace=True)
df['CTC'].fillna(df['CTC'].median(),inplace=True)


df.loc[df['Department'].isin(['Sales (H)', 'Sales (M)', 'Sales (HP)']),'Median_CTC']=pd.merge(df,market_value,on='Grade',how='left')

df['compa ratio']=df['CTC']/df['Median_CTC']

df['compa ratio'].fillna(1,inplace=True)

tr=training[training.status=="Attended"]
tr['training_days']=(tr.edate-tr.sdate).dt.days
tr['training_days']=tr['training_days'].apply(lambda x: max(x,1))
tr['training_hours']=(tr.time_to.astype(str).str[:2].astype(int)+(tr.time_to.astype(str).str[3:5].astype(int)/60))-tr.time_from.astype(str).str[:2].astype(int)+(tr.time_from.astype(str).str[3:5].astype(int)/60)
tr['training_hours']=tr['training_hours']*tr['training_days']
tr=tr.groupby(['Emp.No.','Employee Name']).aggregate(sum)
df=pd.merge(df,tr,on='Emp.No.',how='left')
df['trainings_attended']=df['training_days'].apply(lambda x: "Yes" if x>=0 else "No" )
df['avg_training_days']=df.training_days/df.TVS_exp_in_Years.apply(lambda x: min((pd.to_datetime("2021-03-31").toordinal()-pd.to_datetime("2015-01-01").toordinal())/365,x))
df['avg_training_days']=df['avg_training_days'].fillna(0)
df['avg_training_hours']=df.training_hours/df.TVS_exp_in_Years.apply(lambda x: min((pd.to_datetime("2021-03-31").toordinal()-pd.to_datetime("2015-01-01").toordinal())/365,x))
df['avg_training_hours']=df['avg_training_hours'].fillna(0)



df=df[df['Type of employment'].isin({"Permanent","Probationery"})]


df.loc[df['Education'].str.contains('MBA|M.B.A.',na=False),'Education']="MBA"
df.loc[df['Education'].str.contains('DET.',na=False),'Education']="DET"
df.loc[df['Education'].str.contains('PGE',na=False),'Education']="PGE"
df.loc[(df['Education'].str.contains('GE',na=False)) & (df['Education'].str.contains('PGE',na=False)==False),'Education']="GE"
df.loc[df['Education'].str.contains('B.E|B Tech',na=False),'Education']="BE/B.Tech"
df.loc[df['Education'].str.contains('Diploma',na=False),'Education']="Diploma"
df.loc[df['Education'].str.contains('B.COM|B.Com',na=False),'Education']="B.Com"
df.loc[df['Education'].str.contains('M.C.A.|MCA',na=False),'Education']="MCA"
df.loc[df['Education'].str.contains('ICWA',na=False),'Education']="ICWA"
df.loc[df['Education'].str.contains('CA',na=False) & (df['Education'].str.contains('MCA',na=False)==False),'Education']="CA"
df.loc[df['Education'].str.contains('B.SC|B.Sc',na=False),'Education']="B.SC"

df=df.reset_index(drop=True)
df['latest_rating']=0
for i in range(0,len(df)):
    if pd.notnull(df['PR 20 - 21'][i]):
        df['latest_rating'][i]=df['PR 20 - 21'][i]
    elif pd.isnull(df['PR 20 - 21'][i]) and pd.notnull(df['PR 19 - 20'][i]):  
        df['latest_rating'][i]=df['PR 19 - 20'][i] 
    elif pd.isnull(df['PR 19 - 20'][i]) and pd.notnull(df['PR 18 - 19'][i]):  
        df['latest_rating'][i]=df['PR 18 - 19'][i] 
    elif pd.isnull(df['PR 18 - 19'][i]) and pd.notnull(df['PR 17 - 18'][i]):  
        df['latest_rating'][i]=df['PR 17 - 18'][i] 
    else:
        df['latest_rating'][i]=df['PR 16 - 17'][i]

df=df.reset_index(drop=True)
df['previous_rating']=0
for i in range(0,len(df)):
    if pd.notnull(df['PR 20 - 21'][i]):
        df['previous_rating'][i]=df['PR 19 - 20'][i]
    elif pd.isnull(df['PR 20 - 21'][i]) and pd.notnull(df['PR 19 - 20'][i]):  
        df['previous_rating'][i]=df['PR 18 - 19'][i] 
    elif pd.isnull(df['PR 19 - 20'][i]) and pd.notnull(df['PR 18 - 19'][i]):  
        df['previous_rating'][i]=df['PR 17 - 18'][i] 
    elif pd.isnull(df['PR 18 - 19'][i]) and pd.notnull(df['PR 17 - 18'][i]):  
        df['previous_rating'][i]=df['PR 16 - 17'][i] 

rating_scale={'VP':3, 'PM':2, 'EP':5, 'EE':4, 'ME':3, 'NM':1, 'DR':3}

df['latest_rating']=df['latest_rating'].map(rating_scale).fillna(0)
df['previous_rating']=df['previous_rating'].map(rating_scale).fillna(0)

df['rating_delta']=0
for j in range(0,len(df)):
    if df['previous_rating'][j]>0:
        df['rating_delta'][j]=pd.to_numeric(df['latest_rating'][j])-pd.to_numeric(df['previous_rating'][j])
    else: 
        df['rating_delta'][j]=0

potential_rating={'LP':1, 'MP':2, 'HP':3, '-':0,0:0}
df['Potential Rating']=df['Potential Rating'].fillna(0).map(potential_rating)

df['Role']=df['Role'].fillna("no_data")

df['Overall_exp_in_Years']=df.TVS_exp_in_Years + df.External_exp_in_Years

df['CTC_percentile']=df.CTC.rank(pct=True)
df['Age_percentile']=df.Age.rank(pct=True)
df['TVS_exp_in_Years_percentile']=df.TVS_exp_in_Years.rank(pct=True)
df['External_exp_in_Years_percentile']=df.External_exp_in_Years.rank(pct=True)
df['Overall_exp_in_Years_percentile']=df.Overall_exp_in_Years.rank(pct=True)
df['Years_in_Current_Grade_percentile']=df.Years_in_Current_Grade.rank(pct=True)
df['Years_in_Current_Role_percentile']=df.Years_in_Current_Role.rank(pct=True)
df['rating_delta_percentile']=df.rating_delta.rank(pct=True)

progression=progression[['Emp.No.','Progression Grade']]
progression=progression.groupby('Emp.No.').aggregate('count').reset_index()
progression.rename(columns={'Progression Grade':'No of Grade changes'},inplace=True)

df=pd.merge(df,progression,on='Emp.No.',how='left')
df['No of Grade changes'].fillna(0,inplace=True)
df['Grade change per year']=df['No of Grade changes']/df['TVS_exp_in_Years']
df['Grade change per year'].fillna(0,inplace=True)

recognition=recognition.reset_index()
recognition=recognition[['EmployeeCode','Date']]
recognition.rename(columns={'EmployeeCode':'Emp.No.'},inplace=True)
recognition=recognition.groupby('Emp.No.').aggregate('count').reset_index()
recognition.rename(columns={'Date':'Recognitions'},inplace=True)

df=pd.merge(df,recognition,on='Emp.No.',how='left')
df['Recognitions'].fillna(0,inplace=True)

recognition=pd.read_excel("../input/recognition.xlsx",index_col=0).reset_index()
recognition=recognition[['EmployeeCode','Date']]
recognition=recognition.groupby('EmployeeCode').aggregate(max).reset_index()
recognition.rename(columns={'EmployeeCode':'Emp.No.','Date':'Last recognition date'},inplace=True)

df=pd.merge(df,recognition,on='Emp.No.',how='left')


df['Time since last recognition']=0
df.loc[df['Attrition']=="Yes",'Time since last recognition']=((df['Resignation Date']-df['Last recognition date'].astype(np.datetime64))/ np.timedelta64(1, 'D'))/365

from datetime import datetime
df.loc[df['Attrition']=="No",'Time since last recognition']=((datetime.today()-df['Last recognition date'].astype(np.datetime64))/ np.timedelta64(1, 'D'))/365
df.loc[df['Last recognition date'].isna(),'Time since last recognition']=df['TVS_exp_in_Years']

inc=pd.merge(incentive,df[['Emp.No.','Resignation Date']],on='Emp.No.',how='left')
inc3_=inc.sort_values('BEGDA').groupby('Emp.No.').tail(3).reset_index()
inc6_=inc.sort_values('BEGDA').groupby('Emp.No.').tail(6).reset_index()
inc3=inc3_[['Emp.No.','incentive']].groupby('Emp.No.').aggregate('mean').reset_index().rename(columns={'incentive':'last three months avg incentive'})
inc6=inc6_[['Emp.No.','incentive']].groupby('Emp.No.').aggregate('mean').reset_index().rename(columns={'incentive':'last six months avg incentive'})
inc3_sum=inc3_[['Emp.No.','incentive']].groupby('Emp.No.').aggregate('sum').reset_index().rename(columns={'incentive':'last three months incentive'})
inc6_sum=inc6_[['Emp.No.','incentive']].groupby('Emp.No.').aggregate('sum').reset_index().rename(columns={'incentive':'last six months incentive'})

df=pd.merge(df,inc3,on='Emp.No.',how='left')
df=pd.merge(df,inc6,on='Emp.No.',how='left')
df=pd.merge(df,inc3_sum,on='Emp.No.',how='left')
df=pd.merge(df,inc6_sum,on='Emp.No.',how='left')

df['last three months avg incentive'].fillna(0,inplace=True)
df['last six months avg incentive'].fillna(0,inplace=True)

df['last three months incentive utilization']=df['last three months incentive']/df['CTC']
df['last six months incentive utilization']=df['last six months incentive']/df['CTC']

df['last three months incentive utilization'].fillna(0,inplace=True)
df['last six months incentive utilization'].fillna(0,inplace=True)

df=df[df['Department'].isin(['Sales (H)', 'Sales (M)', 'Sales (HP)'])].reset_index(drop=True)

df['Role']=df.Role.replace({'Territory Sales Manager':'Territory Manager (Sales)','Territory Manager - Sales':'Territory Manager (Sales)'})

df['Grade']=df.Grade.map({'TM':0,'A1':1,'A2':2,'A3':3,'B1':4,'B2':5,'B3':6,'C1':7,'C2':8,'C3':9,'D1':10,'D2':11,'D3':12,'D4':13,'D5':14,'E1':15,'E2':16,'E3':17,'E4':18,'E5':19,'E6':20,'E7':21,
'M1':22,'M2':23,'M3':24,'M4':25,'M5':26,'M6':27})

df.to_csv("../input/created/input_sales.csv",index=False)

prediction_input=df[df.Attrition=='No']

prediction_input.to_csv("../input/created/sales_prediction_input.csv",index=False)

prediction_input = prediction_input.sample(frac=1).reset_index(drop=True) 


#Input for out of time validation

oot=[7684,9178,11488,11789,12336,12811,12815,12447, 12345, 12784, 13106, 12443,  7285, 12986, 10623, 12213,
       12420, 10855, 12064, 12337, 11493, 12783]
oot=pd.DataFrame(oot).rename(columns={0:"Emp.No."})
oot=pd.merge(oot,attrition[['Emp.No.','Leaving Date']],on='Emp.No.',how='left')
oot['Resignation Date']=oot['Leaving Date']-pd.DateOffset(months=3)
oot['Resignation Date'].fillna("2021-09-01",inplace=True)
oot=oot[['Emp.No.','Resignation Date']]
oot.to_excel("../input/oot.xlsx")



