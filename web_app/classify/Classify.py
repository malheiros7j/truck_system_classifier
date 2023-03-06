import pickle
import numpy as np
import pandas as pd


class ClassifyMaintenance(object):
    

    def __init__(self):

        self.home_path = ''
        self.standard_scaler = pickle.load(open(self.home_path + 'features/standard_scaler_train.pkl','rb'))

    def data_cleaning(self,df1):

        df1 = df1.replace(np.nan,0)
        df1 = df1.astype({col: float for col in df1.columns[1:]})

        # print(df1.columns)



        return df1


    def class_feature(self,df2):

        # df2['class'] = df2['class'].apply(lambda x: 1 if x == 'pos' else 0)
        return df2



    def data_preparation(self,df3):


        # Standardization -> Transforma em array
        df_scalar = self.standard_scaler.transform(df3)

        df3 = pd.DataFrame(df_scalar,columns=df3.columns)

        # print(df3)
        # print(df3.columns)

        cols_selected = ['ci_000', 'bj_000', 'ag_002', 'ay_004', 'aa_000', 'cs_001', 'ay_001',
       'ay_002', 'ag_006', 'ee_005', 'ag_001', 'al_000', 'am_0', 'ay_008',
       'ay_009', 'ay_005', 'ee_007', 'ad_000', 'ay_000', 'dg_000', 'cn_004',
       'cs_005', 'cn_000', 'ai_000', 'dn_000', 'ck_000', 'cn_006', 'aq_000',
       'an_000', 'cl_000', 'cu_000', 'dr_000', 'ao_000', 'az_001', 'ay_006',
       'cc_000', 'ah_000', 'ay_003', 'cj_000']

       # printf(df3[cols_selected].columns)
    
        # Return Only Colums Selected
        return df3[cols_selected]



    def get_prediction(self,model,original_data,test_data):

        # Prediction
        pred = model.predict_proba(test_data)

        # Join Predict into the original data
        original_data['score_prediction'] = pred[:,1]
        original_data = original_data.sort_values('score_prediction',ascending=False)
        
        return original_data.to_json(orient='records',date_format='iso')

