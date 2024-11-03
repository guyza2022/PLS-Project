import streamlit as st
import Functions
import joblib
import pandas as pd
import os
import yaml

dirname = os.path.dirname(__file__)
dirname = os.path.join(dirname,'Models')

@st.cache_resource
def load_data(data):
    return pd.read_excel(data,index_col=0)

def user_page():
    st.experimental_set_query_params(code='logged_in')
    st.title('Milk Quality Prediction')
    data = st.file_uploader('Upload Dataset',type=['xlsx'])
    if data is not None:
        data = load_data(data)
        y_names = ['SCC','Fat','Prt']
        x_names = data.columns
        #x_names = [x for x in x_names if x > 900 and x < 1663]
        sample_data = data.sample(frac=0.3, random_state=42)
        #data_dict = Functions.split_data_corr_y(data,y_names)
        st.session_state['uploaded'] = True
        st.success('Uploaded Successfully')
    st.subheader('Raw Data')
    if st.session_state['uploaded'] == True:
        #Show only top 50 rows
        st.dataframe(data.iloc[:50,:])
        with st.expander('View Data Statistic'):
            st.text('Data Statistic')
            st.write(data.describe().loc[['count','mean','std','min','max'],x_names])
            st.success(f"Number of feature: {len(x_names)}")
        #st.text('Spectrum Plot')
        with st.expander('View Spectrum Plot'):
            raw_fig = Functions.plot_spectrum(sample_data, x_names)
            st.write(raw_fig)
    else:
        st.info('Waiting For Data')
    st.subheader('Result')
    if st.session_state['uploaded'] == True:
        sv1_data = Functions.perform_savgol(data, x_names, 1, 3, 0)
        #sv1_fig = Functions.plot_spectrum(sv1_data,x_names)
        #st.write(sv1_fig)
        msc_data = sv1_data
        msc_data.loc[:, x_names] = Functions.msc(sv1_data[x_names])
        #msc_fig = Functions.plot_spectrum(msc_data,x_names)
        #st.write(msc_fig)
        sv2_data = Functions.perform_savgol(msc_data, x_names, 1, 3, 1)
        #sv2_fig = Functions.plot_spectrum(sv2_data,x_names)
        #st.write(sv2_fig)

        numpy_data = sv2_data.to_numpy()
        

        #st.write(numpy_data)
        if st.button('Predict'):
            file = open('current_model.yaml', 'r')
            model_dict = yaml.safe_load(file)
            for y_name in y_names:
                with st.expander('View '+y_name+' Prediction Results'):
                    current_model_path = os.path.join(dirname,y_name,model_dict[y_name.lower()],model_dict[y_name.lower()]+'_'+y_name+'.joblib')
                    #st.write(current_model_path)
                    current_scaler_path = os.path.join(dirname,y_name,model_dict[y_name.lower()],model_dict[y_name.lower()]+'_'+'scaler'+'_'+y_name+'.joblib')
                    model = joblib.load(current_model_path)
                    scaler = joblib.load(current_scaler_path)
                    result_numpy = model.predict(numpy_data)
                    #st.write(result)
                    result_numpy = scaler.inverse_transform(result_numpy)
                    result_frame = pd.DataFrame({'Name':data.index,'Result':result_numpy.reshape(-1,)})
                    st.dataframe(result_frame)
            st.success('Predicted Successfully')
        else:
            st.success('Model is Ready to Predict')
    else:
        st.info('Waiting For Data')