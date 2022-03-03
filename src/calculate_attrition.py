import pandas as pd
import numpy as np
import datetime
import warnings
warnings.filterwarnings(action = 'ignore')
import matplotlib.pyplot as plt
import seaborn as sns

active_employees=pd.read_excel("../input/active_employees.xlsx")
attrition=pd.read_excel("../input/attrition.xlsx")

active_employees.rename(columns={'Masked Emp No':'Emp.No.','Masked CTC in lakhs':'CTC'},inplace=True)
attrition.rename(columns={'Masked Emp ID':'Emp.No.','Masked CTC in lakhs':'CTC'},inplace=True)


dac=active_employees[['Emp.No.','Date of Join', 'Present Grade','Department','Location','Gender','Contract Type']]
dac.rename(columns={'Present Grade':'Grade','Contract Type':'Type of employment'},inplace=True)
dac['Attrition']="No"

dat=attrition[['Emp.No.','Date of Join', 'Grade','Department','Location','Gender','Leaving Date','Reason',"Status.1"]]
dat.rename(columns={"Status.1":"Type of employment"},inplace=True)
dat['Attrition']="Yes"
dat=dat[dat['Reason'].str.contains("Resignation",na=False)]

df=pd.concat([dac,dat])
df=df[df['Type of employment'].isin({"Permanent","Probationery"})]



consolidated=pd.DataFrame()
def attrition_calculate(df,team):

        consolidated=pd.DataFrame()
        for starting_fy_year in range(2015,2022):
            ending_fy_year=starting_fy_year+1
            
            #Opening headcount
            APR_op=len(df[df['Date of Join']<f"{starting_fy_year}-04-1"])-len(df[df['Leaving Date']<f"{starting_fy_year}-04-1"])
            MAY_op=len(df[df['Date of Join']<f"{starting_fy_year}-05-1"])-len(df[df['Leaving Date']<f"{starting_fy_year}-05-1"])
            JUN_op=len(df[df['Date of Join']<f"{starting_fy_year}-06-1"])-len(df[df['Leaving Date']<f"{starting_fy_year}-06-1"])
            JUL_op=len(df[df['Date of Join']<f"{starting_fy_year}-07-1"])-len(df[df['Leaving Date']<f"{starting_fy_year}-07-1"])
            AUG_op=len(df[df['Date of Join']<f"{starting_fy_year}-08-1"])-len(df[df['Leaving Date']<f"{starting_fy_year}-08-1"])
            SEP_op=len(df[df['Date of Join']<f"{starting_fy_year}-09-1"])-len(df[df['Leaving Date']<f"{starting_fy_year}-09-1"])
            OCT_op=len(df[df['Date of Join']<f"{starting_fy_year}-10-1"])-len(df[df['Leaving Date']<f"{starting_fy_year}-10-1"])
            NOV_op=len(df[df['Date of Join']<f"{starting_fy_year}-11-1"])-len(df[df['Leaving Date']<f"{starting_fy_year}-11-1"])
            DEC_op=len(df[df['Date of Join']<f"{starting_fy_year}-12-1"])-len(df[df['Leaving Date']<f"{starting_fy_year}-12-1"])
            JAN_op=len(df[df['Date of Join']<f"{ending_fy_year}-01-1"])-len(df[df['Leaving Date']<f"{ending_fy_year}-01-1"])
            FEB_op=len(df[df['Date of Join']<f"{ending_fy_year}-02-1"])-len(df[df['Leaving Date']<f"{ending_fy_year}-02-1"])
            MAR_op=len(df[df['Date of Join']<f"{ending_fy_year}-03-1"])-len(df[df['Leaving Date']<f"{ending_fy_year}-03-1"])

            opening_hc=pd.DataFrame([APR_op,MAY_op,JUN_op,JUL_op,AUG_op,SEP_op,OCT_op,NOV_op,DEC_op,JAN_op,FEB_op,MAR_op]).rename(columns={0:"Opening_Headcount"})

            #Closing headcount
            APR_cl=MAY_op
            MAY_cl=JUN_op
            JUN_cl=JUL_op
            JUL_cl=AUG_op
            AUG_cl=SEP_op
            SEP_cl=OCT_op
            OCT_cl=NOV_op
            NOV_cl=DEC_op
            DEC_cl=JAN_op
            JAN_cl=FEB_op
            FEB_cl=MAR_op
            MAR_cl=len(df[df['Date of Join']<f"{ending_fy_year}-04-1"])-len(df[df['Leaving Date']<f"{ending_fy_year}-04-1"])

            closing_hc=pd.DataFrame([APR_cl,MAY_cl,JUN_cl,JUL_cl,AUG_cl,SEP_cl,OCT_cl,NOV_cl,DEC_cl,JAN_cl,FEB_cl,MAR_cl]).rename(columns={0:"Closing_Headcount"})

            #Exits
            APR_ex=len(df[(df['Leaving Date']>=f"{starting_fy_year}-04-01") & (df['Leaving Date']<=f"{starting_fy_year}-04-30")])
            MAY_ex=len(df[(df['Leaving Date']>=f"{starting_fy_year}-05-01") & (df['Leaving Date']<=f"{starting_fy_year}-05-31")])
            JUN_ex=len(df[(df['Leaving Date']>=f"{starting_fy_year}-06-01") & (df['Leaving Date']<=f"{starting_fy_year}-06-30")])
            JUL_ex=len(df[(df['Leaving Date']>=f"{starting_fy_year}-07-01") & (df['Leaving Date']<=f"{starting_fy_year}-07-31")])
            AUG_ex=len(df[(df['Leaving Date']>=f"{starting_fy_year}-08-01") & (df['Leaving Date']<=f"{starting_fy_year}-08-31")])
            SEP_ex=len(df[(df['Leaving Date']>=f"{starting_fy_year}-09-01") & (df['Leaving Date']<=f"{starting_fy_year}-09-30")])
            OCT_ex=len(df[(df['Leaving Date']>=f"{starting_fy_year}-10-01") & (df['Leaving Date']<=f"{starting_fy_year}-10-31")])
            NOV_ex=len(df[(df['Leaving Date']>=f"{starting_fy_year}-11-01") & (df['Leaving Date']<=f"{starting_fy_year}-11-30")])
            DEC_ex=len(df[(df['Leaving Date']>=f"{starting_fy_year}-12-01") & (df['Leaving Date']<=f"{starting_fy_year}-12-31")])
            JAN_ex=len(df[(df['Leaving Date']>=f"{ending_fy_year}-01-01") & (df['Leaving Date']<=f"{ending_fy_year}-01-31")])
            FEB_ex=len(df[(df['Leaving Date']>=f"{ending_fy_year}-02-01") & (df['Leaving Date']<=f"{ending_fy_year}-02-28")])
            MAR_ex=len(df[(df['Leaving Date']>=f"{ending_fy_year}-03-01") & (df['Leaving Date']<=f"{ending_fy_year}-03-31")])

            exits=pd.DataFrame([APR_ex,MAY_ex,JUN_ex,JUL_ex,AUG_ex,SEP_ex,OCT_ex,NOV_ex,DEC_ex,JAN_ex,FEB_ex,MAR_ex]).rename(columns={0:"Exits"})


            #Cumulative Exits
            APR_ex_cum=len(df[(df['Leaving Date']>=f"{starting_fy_year}-04-01") & (df['Leaving Date']<=f"{starting_fy_year}-04-30")])
            MAY_ex_cum=len(df[(df['Leaving Date']>=f"{starting_fy_year}-05-01") & (df['Leaving Date']<=f"{starting_fy_year}-05-31")])+APR_ex_cum
            JUN_ex_cum=len(df[(df['Leaving Date']>=f"{starting_fy_year}-06-01") & (df['Leaving Date']<=f"{starting_fy_year}-06-30")])+MAY_ex_cum
            JUL_ex_cum=len(df[(df['Leaving Date']>=f"{starting_fy_year}-07-01") & (df['Leaving Date']<=f"{starting_fy_year}-07-31")])+JUN_ex_cum
            AUG_ex_cum=len(df[(df['Leaving Date']>=f"{starting_fy_year}-08-01") & (df['Leaving Date']<=f"{starting_fy_year}-08-31")])+JUL_ex_cum
            SEP_ex_cum=len(df[(df['Leaving Date']>=f"{starting_fy_year}-09-01") & (df['Leaving Date']<=f"{starting_fy_year}-09-30")])+AUG_ex_cum
            OCT_ex_cum=len(df[(df['Leaving Date']>=f"{starting_fy_year}-10-01") & (df['Leaving Date']<=f"{starting_fy_year}-10-31")])+SEP_ex_cum
            NOV_ex_cum=len(df[(df['Leaving Date']>=f"{starting_fy_year}-11-01") & (df['Leaving Date']<=f"{starting_fy_year}-11-30")])+OCT_ex_cum
            DEC_ex_cum=len(df[(df['Leaving Date']>=f"{starting_fy_year}-12-01") & (df['Leaving Date']<=f"{starting_fy_year}-12-31")])+NOV_ex_cum
            JAN_ex_cum=len(df[(df['Leaving Date']>=f"{ending_fy_year}-01-01") & (df['Leaving Date']<=f"{ending_fy_year}-01-31")])+DEC_ex_cum
            FEB_ex_cum=len(df[(df['Leaving Date']>=f"{ending_fy_year}-02-01") & (df['Leaving Date']<=f"{ending_fy_year}-02-28")])+JAN_ex_cum
            MAR_ex_cum=len(df[(df['Leaving Date']>=f"{ending_fy_year}-03-01") & (df['Leaving Date']<=f"{ending_fy_year}-03-31")])+FEB_ex_cum

            cumulative_exits=pd.DataFrame([APR_ex_cum,MAY_ex_cum,JUN_ex_cum,JUL_ex_cum,AUG_ex_cum,SEP_ex_cum,OCT_ex_cum,NOV_ex_cum,DEC_ex_cum,JAN_ex_cum,FEB_ex_cum,MAR_ex_cum]).rename(columns={0:"Cumulative_Exits"})



            df1=opening_hc
            df1['Month']=pd.DataFrame(['APR','MAY','JUN','JUL','AUG','SEP','OCT','NOV','DEC','JAN','FEB','MAR'])
            df1['Year']=pd.DataFrame([starting_fy_year,starting_fy_year,starting_fy_year,starting_fy_year,starting_fy_year,starting_fy_year,starting_fy_year,starting_fy_year,starting_fy_year,ending_fy_year,ending_fy_year,ending_fy_year])
            df1['Closing_Headcount']=closing_hc
            df1['Average_Headcount']=(df1['Closing_Headcount']+df1['Opening_Headcount'][0])/2
            df1['Exits']=exits
            df1['Cumulative_Exits']=cumulative_exits
            df1=df1[['Year','Month','Opening_Headcount','Closing_Headcount','Average_Headcount','Exits','Cumulative_Exits']]
            df1['Month_count']=[1,2,3,4,5,6,7,8,9,10,11,12]
            df1['Annualized_Attrition_Percentage']=(df1['Cumulative_Exits']/df1['Average_Headcount'])*(12/df1['Month_count'])
            df1['FY']=f"{starting_fy_year}-{ending_fy_year}"
            
            
            consolidated=consolidated.append(df1)
            consolidated['Team']=team
        consolidated.to_excel(f"../output/attrition_calculation/{team}_attrition.xlsx",index=False)

        
def attrition_summary(data,team,value):
    data=pd.read_excel(f"../output/attrition_calculation/{data}_attrition.xlsx",index_col=0)
    data=pd.pivot_table(data,index="FY",columns="Month",values=value)
    data=data[['APR', 'MAY','JUN','JUL', 'AUG','SEP','OCT', 'NOV', 'DEC', 'JAN','FEB','MAR']]
    data.loc['2021-2022','NOV']="-"
    data.loc['2021-2022','DEC']="-"
    data.loc['2021-2022','JAN']="-"
    data.loc['2021-2022','FEB']="-"
    data.loc['2021-2022','MAR']="-"
    data.to_excel(f"../output/attrition_calculation/{team}_attrition_{value}.xlsx")
    return data

        
sales=df[df['Department'].isin(['Sales (H)', 'Sales (M)', 'Sales (HP)'])].reset_index(drop=True)

attrition_calculate(sales,'sales')
attrition_summary('sales','sales','Annualized_Attrition_Percentage')
attrition_summary('sales','sales','Exits')
attrition_summary('sales','sales','Closing_Headcount')