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
from pathlib import Path
import membership as ms
import shutil
from sklearn.feature_selection import r_regression
from sklearn.metrics import mean_squared_error
import yaml
import zipfile
import io

ms.connection()
n_ys = 3
st.set_page_config(page_title='Milk Quality Prediction', page_icon=None, layout="wide", initial_sidebar_state="collapsed", menu_items=None)

@st.cache_resource
def load_data(data):
    return pd.read_excel(data,index_col=0)


def display_result(train_labels: list = None):
    for i in range(len(train_labels)):
        with st.expander('View Results for '+train_labels[i]):
            st.dataframe(list_result_table[i].astype(float),use_container_width=True)
            st.plotly_chart(list_result_fig[i],use_container_width=True)
            st.plotly_chart(list_result_weights[i],use_container_width=True)
            with st.container():
                list_cols = st.columns(7)
                with list_cols[0]:
                    st.success('Correlation :\n ' + str(round(list_result_corr[i], 7)))
                with list_cols[1]:
                    st.success('R-square :\n' + str(round(list_result_score[i], 7)))
                with list_cols[2]:
                    st.success('RMSE :\n' + str(round(list_result_rmse[i], 7)))
                with list_cols[3]:
                    st.success('Train Bias :\n' + str(round(list_train_bias[i], 7)))
                with list_cols[4]:
                    st.success('Test Bias :\n' + str(round(list_test_bias[i], 7)))
                with list_cols[5]:
                    st.success('Train SEP :\n' + str(round(list_train_sep[i], 7)))
                with list_cols[6]:
                    st.success('Test SEP :\n' + str(round(list_test_sep[i], 7)))


def train_model(n_components, train_labels: list =  None):
    global y_test,y_pred
    for y_name in train_labels:
        model = deepcopy(PLSRegression(n_components=n_components))
        model.fit(training_data_dict.get(y_name)[0],training_data_dict.get(y_name)[2])
        model_dict[y_name] = model

        y_train_pred = model.predict(training_data_dict.get(y_name)[0])
        y_pred = model.predict(training_data_dict.get(y_name)[1])

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
        list_result_table.append(compare)

        train_bias = np.mean(y_train_pred - y_train)
        test_bias = np.mean(y_pred - y_test)
        list_train_bias.append(train_bias)
        list_test_bias.append(test_bias)

        train_sep = np.sqrt(np.mean((y_train_pred - y_train - train_bias) ** 2))
        test_sep = np.sqrt(np.mean((y_pred - y_test - test_bias) ** 2))
        list_train_sep.append(train_sep)
        list_test_sep.append(test_sep)

        compare_fig = go.Figure()
        for col in compare.columns:
            compare_fig.add_trace(go.Scatter(x=compare.index, y=compare[col],
                                mode='lines+markers',
                                name=col))
            compare_fig.update_layout(title='Compare Plot',
                                xaxis_title='Records',
                                yaxis_title='Value')
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
        
        list_result_weights.append(weights_fig)


def save_model(model_name, y_name):
    model_name_y_name = model_name+'_'+y_name
    scaler_name = model_name+'_scaler'
    scaler_name_y_name = scaler_name+'_'+y_name
    save_dir = os.path.join(dirname, y_name, model_name)
    print(save_dir)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    model_filename = os.path.join(save_dir, model_name_y_name+'.joblib')
    print(model_filename)
    scaler_filename = os.path.join(save_dir, scaler_name_y_name+'.joblib')
    model = model_dict.get(y_name)
    scaler = training_data_dict.get(y_name)[4]
    joblib.dump(model,model_filename)
    joblib.dump(scaler,scaler_filename)
    global all_models
    return None


def save_all_model(model_name, train_labels: list = None):
    global y_names
    try:
        st.session_state['save'] = True
        for label in train_labels:
            folder_dir = os.path.join(dirname,label)
            os.mkdir(folder_dir)
    except:
        pass
    for y_name in train_labels:
        save_model(model_name,y_name)


def create_zip(csv_dict: dict):
    """
    Create an in-memory ZIP file containing multiple CSVs from the provided URLs.
    """
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        for name, df in csv_dict.items():
            try:
                csv_data = df.to_csv(index=False)
                zf.writestr(f"{name}.csv", csv_data)
            except Exception as e:
                st.warning(f"Failed to download from {name}: {e}")
    zip_buffer.seek(0)
    return zip_buffer


def clear_cache_resource():
    global train_now
    train_now = False


def delete_model(label: str, name: str):
    with open('current_model.yaml', 'r') as file:
        data = yaml.safe_load(file)
        if name != data[label.lower()]:
            shutil.rmtree(os.path.join('Models',label,name))
        else:
            st.warning('This model is currently on production!')


def logout():
    st.session_state['login'] = False
    st.session_state['logout'] = True


dirname = os.path.dirname(__file__)
dirname = os.path.join(dirname,'Models')


def get_models_by_labels(label: str):
    global dirname
    return list(os.listdir(os.path.join(dirname, label)))
    

def refresh_models_list():
    return [name for name in os.listdir(dirname) if name != '.DS_Store']


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
    st.session_state['user'] = False #
    st.session_state['save'] = False
    st.session_state['logout'] = False
    st.session_state['save_button'] = False
    train_now = False
    st.cache_data.clear()
    st.cache_resource.clear()


if st.session_state['save'] or st.session_state['train'] == True:
    train_now = True
else:
    train_now = False


if st.session_state['login'] != True and st.session_state['register'] == False:
    login_page()


elif st.session_state['login'] != True and st.session_state['register'] == True:
    register_page()


elif st.session_state['login'] == True and st.session_state['user']:
    user_page()


elif st.session_state['login'] == True and not st.session_state['user']:
    st.experimental_set_query_params(code='logged_in')
    with st.sidebar:
        st.header('Model Management')
        st.subheader('Deploy')
        scc_deploy_model = st.selectbox('SCC',get_models_by_labels('SCC'), key='deploy_scc')
        fat_deploy_model = st.selectbox('Fat',get_models_by_labels('Fat'), key='deploy_fat')
        prt_deploy_model = st.selectbox('Prt',get_models_by_labels('Prt'), key='deploy_prt')
        if st.button('Deploy'):
            with open('current_model.yaml', 'r') as file:
                data = yaml.safe_load(file)
            data['scc'] = scc_deploy_model
            data['fat'] = fat_deploy_model
            data['prt'] = prt_deploy_model
            with open('current_model.yaml', 'w') as file:
                yaml.safe_dump(data, file)
                st.success('Deployed Successfully')
        st.subheader('Delete')       
        scc_delete_model = st.selectbox('SCC',get_models_by_labels('SCC'), key='del_scc')
        if st.button('Delete', disabled=len(get_models_by_labels('SCC')) == 0, key='scc_del_button', on_click=delete_model, kwargs={'label':'SCC','name':scc_delete_model}):
            pass
        fat_delete_model = st.selectbox('Fat',get_models_by_labels('Fat'), key='del_fat')
        if st.button('Delete', disabled=len(get_models_by_labels('Fat')) == 0, key='fat_del_button', on_click=delete_model, kwargs={'label':'Fat','name':fat_delete_model}):
            pass
        prt_delete_model = st.selectbox('Prt',get_models_by_labels('Prt'), key='del_prt')
        if st.button('Delete', disabled=len(get_models_by_labels('Prt')) == 0, key='prt_del_button', on_click=delete_model, kwargs={'label':'Prt','name':prt_delete_model}):
            pass

    with st.container():
        col1,col2 = st.columns([0.9,0.1])
    with col1:
        st.title('Milk Quality Prediction')
    with col2:
        logout_button = st.button('Logout',type='primary',use_container_width=True,on_click=logout)

    data = st.file_uploader('Upload Dataset',type=['xlsx'])
    filter_ratio = [0.1,0.5,0.4]

    if data is not None:
        data = pd.read_excel(data,index_col=0)
        sample_data = data.sample(frac=0.3, random_state=42)
        y_names = data.columns[0:n_ys]
        x_names = data.columns[n_ys:]

        data_dict = Functions.split_data_corr_y(data,y_names)
        st.session_state['uploaded'] = True
        st.success('Uploaded Successfully')
        st.subheader('Raw Data')
        st.dataframe(data.iloc[:50,:])
        with st.container():
            col1,col2 = st.columns(2, gap='small')
            with col1:
                with st.expander('View Data Statistic'):
                    st.text('Data Statistic')
                    st.dataframe(data.describe().loc[['count','mean','std','min','max'],y_names],use_container_width=True)
                    st.success(f"Number of feature: {len(x_names)}")
            with col2:
                with st.expander('View Spectrum Plot'):
                    raw_fig = Functions.plot_spectrum(sample_data, x_names)
                    st.plotly_chart(raw_fig,use_container_width=True)

    st.subheader('Preprocess')
    if st.session_state['uploaded'] == True:
        with st.container():
            col1,col2 = st.columns([0.2,0.8])
        with col1:
            process_and_train_button = st.button('Process')
        with col2:
            skip_check_box: bool = st.checkbox('Skip Preprocess',on_change=clear_cache_resource)
            plot_preprocess_result: bool = st.checkbox('Plot Result')
        if process_and_train_button:
            try:
                st.session_state['process'] = True
                train_now = False
            except:
                st.warning('Duplicated model name. Please Use another name.')
                st.session_state['process'] = False

    if st.session_state['reload'] == False:
        st.session_state['process'] == True

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
                    if not skip_check_box:
                        sv1_data = deepcopy(Functions.perform_savgol(data.copy(), list(x_names), first_input_ponm, first_input_smp//2, first_input_dev))
                        sv1_fig = Functions.plot_spectrum(sv1_data.sample(frac=0.3, random_state=42).copy(), x_names)
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
                if st.session_state['reload'] == True:
                    if not skip_check_box:
                        msc_data = sv1_data
                    try:
                        if not skip_check_box:
                            msc_data.loc[:,x_names] = deepcopy(Functions.msc(sv1_data[x_names].copy()))
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
                            sv2_fig = Functions.plot_spectrum(sv2_data.sample(frac=0.3, random_state=42).copy(),x_names)
                    except:
                        st.error('Error')
            else:
                with col2:
                    st.info(wait_for_process_text)

        if st.session_state['process'] == True and not skip_check_box and plot_preprocess_result:
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
                    if skip_check_box:
                        sv2_data = data
                    x_train,x_test,y_train,y_test = Functions.tt_split(sv2_data,x_names,y_name)
                    y_train, y_test,scaler = Functions.normalize_y(y_train,y_test,scaler)
                    training_data_dict[y_name] = deepcopy((x_train,x_test,y_train,y_test,scaler))


    st.subheader('Training')
    train_labels = []
    with st.form(key='model_train'):
        with st.container():
            p1,p2,p3= st.columns(3, gap='small')
        with p1:
            with st.container():
               c1,c2,c3,c4 = st.columns(4, gap='small')
            with c1:
                train_scc = st.checkbox(label='SCC')
                train_fat = st.checkbox(label='Fat')
                train_prt = st.checkbox(label='Prt')
            with c2:
                n_components = st.slider(label='Number of Components', min_value=1, max_value=200, step=1, value=72)
        train_button = st.form_submit_button(label='Train',type='primary')
        if train_button:
            train_now = True
            if not (train_scc or train_fat or train_prt):
                st.warning('Please select as least one target to train!')
                train_now = False
    if train_scc: train_labels.append('SCC')
    if train_fat: train_labels.append('Fat')
    if train_prt: train_labels.append('Prt')
    list_result_table = []
    list_result_fig = []
    list_result_score = []
    list_result_weights = []
    list_result_corr = []
    list_result_rmse = []
    list_train_bias = []
    list_test_bias = []
    list_train_sep = []
    list_test_sep = []
    if st.session_state['process'] == True:
        if train_now and 'train_labels' in globals():
            st.session_state['train'] = True
            model_dict = {}
            train_model(n_components, train_labels=train_labels)
            def update_model_name():
                global model_name
                model_name = model_name
            summary_df = pd.DataFrame(
                {
                    "Attribute": train_labels,
                    "Correlation": list_result_corr,
                    "R-Square": list_result_score,
                    "RMSE": list_result_rmse,
                    "Train Bias": list_train_bias,
                    "Test Bias": list_test_bias,
                    "Train SEP": list_train_sep,
                    "Test SEP": list_test_sep
                },
                index=train_labels
            )
            with st.container():
                st.dataframe(summary_df, use_container_width=True)
            display_result(train_labels=train_labels)
            param_df = pd.DataFrame({
                "pol_1": [first_input_ponm],
                "dev_1": [first_input_dev],
                "smoothing_point_1": [first_input_smp],
                "pol_2": [second_input_ponm],
                "dev_2": [second_input_dev],
                "smoothing_point_2": [second_input_smp],
                "n_components": [n_components]
            })

            with st.container():
                c1, c2 = st.columns([0.2,0.8], gap='small')
                with c1:
                    try:
                        zip_buffer = create_zip(csv_dict={
                            "parameters": param_df,
                            "Result Summary": summary_df,
                            **{
                                train_labels[i]: list_result_table[i] for i in range(len(train_labels))
                            }
                        })
                        st.download_button(
                            label="Download Results (ZIP)",
                            data=zip_buffer,
                            file_name='training_artifacts.zip',
                            mime='application/zip',
                            use_container_width=True
                        )
                    except Exception as e:
                        st.error(f"Error creating ZIP: {e}")
                with c2:
                    st.success('Trained Successfully')
            with st.form(key='model_save'):
                model_name = st.text_input('Enter the model name')
                save_button = st.form_submit_button(label='Save',type='primary')
                if save_button:
                    if model_name:
                        save_all_model(model_name, train_labels=train_labels)
                        st.success('Saved Successfully')
            if st.session_state['save'] == True:
                st.session_state['save'] = False
    else:
        st.info('Waiting For Pre-processed Data')

