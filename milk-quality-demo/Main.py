import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import Functions
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import accuracy_score
from login_page import login_page
from register_page import register_page
from user_page import user_page
import plotly.express as px
import plotly.graph_objects as go
import joblib
import os
from copy import deepcopy
import pickle
#import streamlit_authenticator as sta
from pathlib import Path
import membership as ms
import shutil
from sklearn.feature_selection import r_regression
from sklearn.metrics import mean_squared_error

# connected = False
# while not connected:
#     try:
        
#         connected = True
#     except Exception as E:
#         print(E)

ms.connection()
n_ys = 3
st.set_page_config(page_title='Milk Quality Prediction', page_icon=None, layout="wide", initial_sidebar_state="collapsed", menu_items=None)

@st.cache_resource
def load_data(data):
    return pd.read_excel(data,index_col=0)

#@st.cache_resource
def display_result():
    for i in range(n_ys):
        with st.expander('View Results for '+y_names[i]):
            #st.write(i)
            st.dataframe(list_result_table[i],use_container_width=True)
            st.plotly_chart(list_result_fig[i],use_container_width=True)
            st.plotly_chart(list_result_weights[i],use_container_width=True)
            with st.container():
                list_cols = st.columns(3)
                with list_cols[0]:
                    st.success('Correlation : '+str(list_result_corr[i]))
                with list_cols[1]:
                    st.success('R-square : '+str(list_result_score[i]))
                with list_cols[2]:
                    st.success('RMSE : '+str(list_result_rmse[i]))

#@st.cache_resource
def train_model(n_components):
    global y_test,y_pred
    #print(2)
    for y_name in y_names:
        #st.text('Number of components :'+str(n_components))
        model = deepcopy(PLSRegression(n_components=n_components))
        #st.write(x_train)
        model.fit(training_data_dict.get(y_name)[0],training_data_dict.get(y_name)[2])
        model_dict[y_name] = model

        #with st.expander('View Results for '+y_name):
        y_pred = model.predict(training_data_dict.get(y_name)[1])
        #compare values
        scaler = training_data_dict.get(y_name)[4]
        compare = pd.DataFrame(columns=['y_test','y_pred'])
        compare['y_test'] = training_data_dict.get(y_name)[3].reshape(len(y_test),)
        compare['y_pred'] = y_pred.reshape(len(y_pred),)
        list_result_corr.append(compare['y_test'].corr(compare['y_pred']))
        list_result_rmse.append(mean_squared_error(training_data_dict.get(y_name)[3],y_pred,squared=False))
        y_test = scaler.inverse_transform(training_data_dict.get(y_name)[3])
        y_pred = scaler.inverse_transform(y_pred)
        compare['y_test'] = y_test.reshape(len(y_test),)
        compare['y_pred'] = y_pred.reshape(len(y_pred),)
        #st.write(compare)
        list_result_table.append(compare)

        compare_fig = go.Figure()
        for col in compare.columns:
            compare_fig.add_trace(go.Scatter(x=compare.index, y=compare[col],
                                mode='lines+markers',
                                name=col))
            compare_fig.update_layout(title='Compare Plot',
                                xaxis_title='Records',
                                yaxis_title='Value')
        #st.write(compare_fig)
        list_result_fig.append(compare_fig)
        score = model.score(training_data_dict.get(y_name)[0],training_data_dict.get(y_name)[2])
        list_result_score.append(score)
        weights = model.coef_[0]
        weights_fig = go.Figure()
        weights_fig.add_trace(go.Scatter(x=np.arange(0,len(weights),1), y=weights.reshape(len(weights),),
                                mode='lines+markers',
                                name='Weights'))
        weights_fig.update_layout(title='Regression Coefficients Plot',
                                xaxis_title='Parameter',
                                yaxis_title='Coefficient')
        #st.write(weights)
        
        list_result_weights.append(weights_fig)

def save_model(model_name, y_name):
    model_name_y_name = model_name+'_'+y_name
    scaler_name = model_name+'_scaler'
    scaler_name_y_name = scaler_name+'_'+y_name
    model_filename = os.path.join(dirname, model_name+'/'+model_name_y_name+'.joblib')
    scaler_filename = os.path.join(dirname, model_name+'/'+scaler_name_y_name+'.joblib')
    model = model_dict.get(y_name)
    scaler = training_data_dict.get(y_name)[4]
    joblib.dump(model,model_filename)
    joblib.dump(scaler,scaler_filename)
    global all_models
    #st.write(model_name)
    return None


def save_all_model(model_name):
    #global model_name
    #update_model_name()
    global y_names
    try:
        st.session_state['save'] = True
        folder_dir = os.path.join(dirname,model_name)
        #st.write(folder_dir)
        os.mkdir(folder_dir)
        #st.write(model_name)
    except:
        pass
    for y_name in y_names:
        save_model(model_name,y_name)

def clear_cache_resource():
    global train_now
    train_now = False
    #st.cache_resource.clear()

def delete_model(mode,name):
    global all_models
    if mode == 'Single':
        shutil.rmtree('Models/'+name)
    else:
        for md in all_models:
            shutil.rmtree('Models/'+md)

def logout():
    st.session_state['login'] = False
    st.session_state['logout'] = True
    print(st.session_state['login'])


dirname = os.path.dirname(__file__)
dirname = os.path.join(dirname,'Models')
all_models = [name for name in os.listdir(dirname) if name != '.DS_Store']

def refresh_models_list():
    return [name for name in os.listdir(dirname) if name != '.DS_Store']

#authenticate
#ms.connection() #initiate sql connection

wait_for_process_text = 'Waiting For Data'
hide_streamlit_style = """
                <style>
                #MainMenu {visibility: hidden;}
                footer {visibility: hidden;}
                </style>
                """
st.markdown(hide_streamlit_style, unsafe_allow_html=True) 

if 'uploaded' not in st.session_state:
    st.session_state['uploaded'] = False
    st.session_state['process'] = False
    st.session_state['train'] = False
    st.session_state['reload'] = True 
    st.session_state['login'] = False
    st.session_state['register'] = False
    st.session_state['user'] = False
    st.session_state['save'] = False
    st.session_state['logout'] = False
    st.session_state['save_button'] = False
    train_now = False
    st.cache_data.clear()
    st.cache_resource.clear()
    #st.write(st.session_state)

#print(st.session_state['save'])

if st.session_state['save'] or st.session_state['train'] == True:
    train_now = True
    #st.write(True)
else:
    train_now = False

#train_now = False
#is_user = True

#st.session_state['login'] = True

### Login Page ###
if st.session_state['login'] != True and st.session_state['register'] == False:
    login_page()

### Register Page ###
elif st.session_state['login'] != True and st.session_state['register'] == True:
    register_page()

### User Side Page ###
elif st.session_state['login'] == True and st.session_state['user']:
    user_page()

### Admin Side Page ###
elif st.session_state['login'] == True and not st.session_state['user']:
    st.experimental_set_query_params(code='logged_in')
    ### Side Bar ###
    with st.sidebar:
        st.header('Model Management')
        #tf = open("current_model.txt", "r")
        #st.text('Current Model : '+tf.read())
        to_deploy_model = st.selectbox('Select Model To Deploy',all_models)
        if st.button('Deploy'):
            with open('current_model.txt','r+') as f:
                f.seek(0)
                f.write(to_deploy_model)
                f.truncate() 
                tf = to_deploy_model
                f.close()
                st.success('Deployed Successfully')
        to_delete_model = st.selectbox('Select Model To Delete',all_models)
        with st.container():
            col1,col2 = st.columns([0.075,0.2])
            with col1:
                if all_models == []:
                    st.button('Delete',disabled=True)
                else:
                    st.button('Delete',on_click=delete_model,kwargs={'mode':'Single','name':to_delete_model},key='del1')
            with col2:
                if all_models == []:
                    st.button('Delete all',disabled=True)
                else:
                    st.button('Delete',on_click=delete_model,kwargs={'mode':'All','name':to_delete_model},key='delall')
                    
    with st.container():
        col1,col2 = st.columns([0.9,0.1])
    with col1:
        st.title('Milk Quality Prediction')
    with col2:
        logout_button = st.button('Logout',type='primary',use_container_width=True,on_click=logout)

    # Data
    data = st.file_uploader('Upload Dataset',type=['xlsx'])
    filter_ratio = [0.1,0.5,0.4]

    # Load Data
    if data is not None:
        data = pd.read_excel(data,index_col=0)
        sample_data = data.sample(frac=0.3, random_state=42)
        #data = data.iloc[:100,:]
        y_names = data.columns[0:n_ys]
        x_names = data.columns[n_ys:]
        x_names = [x for x in x_names if x > 900 and x < 1663]

        data_dict = Functions.split_data_corr_y(data,y_names)
        st.session_state['uploaded'] = True
        st.success('Uploaded Successfully')
        st.subheader('Raw Data')
        st.dataframe(data.iloc[:50,:])
        with st.container():
            col1,col2 = st.columns(2,gap='small')
            with col1:
                with st.expander('View Data Statistic'):
                    st.text('Data Statistic')
                    st.dataframe(data.describe().loc[['count','mean','std','min','max'],y_names],use_container_width=True)
            with col2:
                with st.expander('View Spectrum Plot'):
                    raw_fig = Functions.plot_spectrum(sample_data,x_names)
                    st.plotly_chart(raw_fig,use_container_width=True)

    # Preprocess button
    st.subheader('Preprocess')
    if st.session_state['uploaded'] == True:

        #process_button = st.button('Process')
        with st.container():
            col1,col2 = st.columns([0.2,0.8])
        with col1:
            process_and_train_button = st.button('Process')
        with col2:
            skip_check_box = st.checkbox('Skip Preprocess',on_change=clear_cache_resource)
        if process_and_train_button:
            try:
                st.session_state['process'] = True
                train_now = False
            except:
                st.warning('Duplicated model name. Please Use another name.')
                st.session_state['process'] = False
        # elif process_button:
        #     try:
        #         st.session_state['process'] = True
        #         train_now = False
        #         folder_dir = os.path.join(dirname,model_name)
        #         os.mkdir(folder_dir)
        #     except:
        #         st.warning('Duplicated model name. Please Use another name.')
        #         st.session_state['process'] = False

    #st.write(data.columns)
    if st.session_state['reload'] == False:
        st.session_state['process'] == True

    # Preprocess
    with st.container():
        p1,p2,p3 = st.columns(3,gap='medium')
        with p1:
            with st.container():
                col0,col1,col2 = st.columns(filter_ratio)
            with col0:
                st.write(1)
            with col1:
                st.text('Savitzky-Golay Filter')
            first_input_dev,first_input_ponm,first_input_smp = Functions.sav_tuning_1()
            if st.session_state['process'] == True:
                with col2:
                    if not skip_check_box:
                        st.success('Success')
                    else:
                        st.success('Skipped')
                if st.session_state['reload'] == True :
                    #try: 
                    if not skip_check_box:
                        #Functions.update_data(data)
                        sv1_data = deepcopy(Functions.perform_savgol(data.copy(), list(x_names), first_input_ponm, first_input_smp//2, first_input_dev))
                        #Functions.update_data(sv1_data.loc[:,x_names])
                        sv1_fig = Functions.plot_spectrum(sv1_data.sample(frac=0.3, random_state=42).copy(), x_names)
                    # except:
                    #     st.error('Error')
            else:
                with col2:
                    st.info(wait_for_process_text)

        with p2:
            with st.container():
                col0,col1,col2 = st.columns(filter_ratio)
            with col0:
                st.write(2)
            with col1:
                st.text('MSC')
            if st.session_state['process'] == True:
                with col2:
                    if not skip_check_box:
                        st.success('Success')
                    else:
                        st.success('Skipped')
                st.write(st.session_state['reload'])
                if st.session_state['reload'] == True:
                    if not skip_check_box:
                        msc_data = sv1_data
                    try:
                        if not skip_check_box:
                            msc_data.loc[:,x_names] = deepcopy(Functions.msc(sv1_data[x_names].copy()))
                            #Functions.update_data(msc_data)
                            msc_fig = Functions.plot_spectrum(msc_data.sample(frac=0.3, random_state=42).copy(),x_names)
                    except Exception as E:
                        st.error(E)
                        st.error('Error')
            else:
                with col2:
                    st.info(wait_for_process_text)

        with p3:
            with st.container():
                col0,col1,col2 = st.columns(filter_ratio)
            with col0:
                st.write(3)
            with col1:
                st.text('Savitzky-Golay Filter')
            second_input_dev,second_input_ponm,second_input_smp = Functions.sav_tuning_2()
            if st.session_state['process'] == True:
                with col2:
                    if not skip_check_box:
                        st.success('Success')
                    else:
                        st.success('Skipped')
                if st.session_state['reload'] == True:
                    try:
                        if not skip_check_box:
                            sv2_data = deepcopy(Functions.perform_savgol2(msc_data.copy(), list(x_names), second_input_ponm, second_input_smp//2, second_input_dev))
                            #Functions.update_data(sv2_data)
                            sv2_fig = Functions.plot_spectrum(sv2_data.sample(frac=0.3, random_state=42).copy(),x_names)
                    except:
                        st.error('Error')
            else:
                with col2:
                    st.info(wait_for_process_text)

        if st.session_state['process'] == True and not skip_check_box:
            with st.container():
                pd1,pd2,pd3 = st.columns(3,gap='medium')
                with pd1:
                    with st.expander('View Spectrum Plot'):
                        st.plotly_chart(sv1_fig,use_container_width=True)
                with pd2:
                    with st.expander('View Spectrum Plot'):
                        st.plotly_chart(msc_fig,use_container_width=True)
                with pd3:
                    with st.expander('View Spectrum Plot'):
                        st.plotly_chart(sv2_fig,use_container_width=True)
        
        if st.session_state['process'] == True:
            if st.session_state['reload'] == True:
                training_data_dict = {}
                for y_name in y_names:
                    scaler = deepcopy(StandardScaler())
                    #st.write(sv2_data)
                    if skip_check_box:
                        sv2_data = data
                    x_train,x_test,y_train,y_test = Functions.tt_split(sv2_data,x_names,y_name)
                    #normalize y
                    y_train, y_test,scaler = Functions.normalize_y(y_train,y_test,scaler)
                    training_data_dict[y_name] = deepcopy((x_train,x_test,y_train,y_test,scaler))


    # if st.session_state['save'] == True:
    #     st.session_state['train'] = True

    # Training
    st.subheader('Training')
    if st.button('Train'):
        train_now = True
        #st.cache_resource.clear()
    list_result_table = []
    list_result_fig = []
    list_result_score = []
    list_result_weights = []
    list_result_corr = []
    list_result_rmse = []

    # Training 
    if st.session_state['process'] == True:
        if train_now:
            st.session_state['train'] = True
            #st.session_state['reload'] = False
            model_dict = {}
            n_components = 20
            #print(1)
            train_model(n_components)
                #st.success('Score :'+str(score))
                # for y_name in y_names:
                #     save_model(model_name,y_name)
            def update_model_name():
                global model_name
                model_name = model_name
                #st.experimental_rerun()
            
            #displaying
            display_result()
            st.success('Trained Successfully')
            with st.form(key='model_save'):
                model_name = st.text_input('Enter the model name')
                save_button = st.form_submit_button(label='Save',type='primary')
                if save_button:
                    if model_name:
                        #st.write(model_name)
                        save_all_model(model_name)
                        st.success('Saved Successfully')
            # model_name = st.text_input('Enter the model name',on_change=update_model_name)
            # if model_name == '':
            #     save_button = st.button('Save Model',on_click=save_all_model,type='primary', disabled=True)
            # else:
            #     save_button = st.button('Save Model',on_click=save_all_model,type='primary')
            if st.session_state['save'] == True:
                st.session_state['save'] = False
    else:
        st.info('Waiting For Pre-processed Data')

