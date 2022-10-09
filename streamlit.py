# importing libraries required for our model
from keras.models import Sequential, load_model
from tensorflow.keras.models import Sequential, model_from_json
from keras.layers import LSTM,Dense,Dropout
from keras.callbacks import ModelCheckpoint
from keras.callbacks import EarlyStopping
import tensorflow as tf
import streamlit as st
import pandas as pd
import numpy as np

@st.cache(allow_output_mutation=True)
def Load_model():
    # load json and create model
    #json_file = open('LSTM_model.json', 'r')
    #loaded_model_json = json_file.read()
    #json_file.close()
    #loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    #loaded_model.load_weights("LSTM_model_weights.h5")
    #print("Loaded model from disk")
    #model._make_predict_function()
    #model.summary()  # included to make it visible when model is reloaded
    #session = K.get_session()
    #return model, session
    
    #Reading the model from JSON file
    with open('LSTM_model.json', 'r') as json_file:
        json_savedModel= json_file.read()
    #load the model architecture 
    model_j = tf.keras.models.model_from_json(json_savedModel)
    model_j.load_weights('LSTM_model_weights.h5')
    print("Loaded model from disk")
    #Compiling the model
    model_j.compile(loss = 'mse',optimizer = 'adam', metrics = ['mean_squared_error'])
    print("Model complied....")
    model_j.summary()    # included to make it visible when model is reloaded
    return model_j
st.title('Predict Total Sales')
st.header('Enter the shop id and item id')
item = st.number_input('Item Id', min_value=0, max_value=22169, value=1)
shop = st.number_input('Shop Id', min_value=0, max_value=59, value=1)

# Calling DataFrame constructor  
data = {'shop_id': [shop], 'item_id': [item]}
df = pd.DataFrame(data)  
if st.button('Predict Sales'):
    my_model=Load_model()
    my_data=pd.read_pickle(r"FINAL_DATA.pkl")
    if  ((my_data['shop_id'] == shop) & (my_data['item_id'] == item)).any() :
        df1=pd.merge(df, my_data, how='left', on=['shop_id', 'item_id'])
        X_test=df1.drop(['shop_id','item_id'],axis=1)    
        y_pred = my_model.predict(X_test)
        y_test_pred=pd.DataFrame(y_pred)
        y_test_pred.rename(columns = {0:'item_cnt_month'}, inplace = True)
        y_test_pred['item_cnt_month']=y_test_pred['item_cnt_month'].clip(0., 20.)
        val=y_test_pred['item_cnt_month'].values[0]
        st.success(f'The predicted price of the diamond is ${val:.2f} USD')
    else:
        def_val=0.18365039469587
        st.success(f'The predicted price of the diamond is ${def_val:.2f} USD')
    
        
        
    
    
