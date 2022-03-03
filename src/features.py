features = ['Designation','Grade','Role','Type of employment','Mode of Selection','Education','Age','TVS_exp_in_Years','External_exp_in_Years','Overall_exp_in_Years','Years_in_Current_Role','Potential Rating',
'CTC','latest_rating','previous_rating','avg_training_days', 'avg_training_hours','No of Grade changes', 'Grade change per year','Recognitions','last three months avg incentive',
       'last six months avg incentive','last three months incentive utilization','last six months incentive utilization','Time since last recognition','compa ratio','Attrition','kfold']
ohe_features = ['Designation','Role','Type of employment','Mode of Selection','Education']
labelencode_features= ['Designation','Role','Type of employment','Mode of Selection','Education']
numerical_features = ['Age', 'TVS_exp_in_Years', 'External_exp_in_Years','Overall_exp_in_Years','Years_in_Current_Role','Potential Rating',
'CTC','latest_rating','previous_rating','Grade','avg_training_days', 'avg_training_hours','No of Grade changes', 'Grade change per year','Recognitions','last three months avg incentive',
       'last six months avg incentive','last three months incentive utilization','last six months incentive utilization','Time since last recognition','compa ratio']
target_feature=['Attrition']





# ###Excluding confidential
# features = ['Designation','Grade','Role','Type of employment','Mode of Selection','Education','Age','TVS_exp_in_Years','External_exp_in_Years','Overall_exp_in_Years','Years_in_Current_Role',
# 'avg_training_days', 'avg_training_hours','No of Grade changes', 'Grade change per year','Recognitions','last three months avg incentive',
#        'last six months avg incentive','Attrition','kfold']
# ohe_features = ['Designation','Role','Type of employment','Mode of Selection','Education']
# labelencode_features= ['Designation','Role','Type of employment','Mode of Selection','Education']
# numerical_features = ['Age', 'TVS_exp_in_Years', 'External_exp_in_Years','Overall_exp_in_Years','Years_in_Current_Role','Grade','avg_training_days', 'avg_training_hours','No of Grade changes', 'Grade change per year','Recognitions','last three months avg incentive',
#        'last six months avg incentive']
# target_feature=['Attrition']


