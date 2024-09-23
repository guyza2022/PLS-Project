import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import accuracy_score
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

data:pd.DataFrame = None

def update_data(new_data):
    global data
    data = new_data

@st.cache_data
def split_data_corr_y(data,_y_names):
    dataset_dict = {}
    for name in _y_names:
        temp_data = data
        y_data = data[name]
        temp_data = temp_data.drop(columns=_y_names)
        temp_data[name] = y_data
        dataset_dict[name] = temp_data
    return dataset_dict

#@st.cache_data(hash_funcs={pd.DataFrame: lambda _: None}, persist='disk')
@st.cache_data(persist='disk')
def perform_savgol(data: pd.DataFrame, _x_names: list, pol, wl, dvt):
    def simple_moving_average(col, window_length):
        return col.rolling(window=window_length*2+1, min_periods=1, center=False).mean()

    data = data.copy()
    
    if pol == 0:
        st.warning("If the polynomial order is set to 0, an SMA (Simple Moving Average) will be used instead (equivalently).")
        data[_x_names] = data[_x_names].apply(lambda col: simple_moving_average(col, wl), axis=1)
    else:
        data[_x_names] = data[_x_names].apply(lambda col: savgol_filter(col, polyorder=pol, window_length=wl, deriv=dvt, mode='nearest'))
    
    return data

#@st.cache_data(hash_funcs={pd.DataFrame: lambda _: None}, persist='disk')
@st.cache_data(persist='disk')
def perform_savgol2(data: pd.DataFrame, _x_names: list, pol, wl, dvt):
    def simple_moving_average(col, window_length):
        return col.rolling(window=window_length, min_periods=1, center=False).mean()

    data = data.copy()
    
    if pol == 0:
        st.warning("If the polynomial order is set to 0, an SMA (Simple Moving Average) will be used instead (equivalently).")
        data[_x_names] = data[_x_names].apply(lambda col: simple_moving_average(col, wl), axis=1)
    else:
        data[_x_names] = data[_x_names].apply(lambda col: savgol_filter(col, polyorder=pol, window_length=wl, deriv=dvt, mode='nearest'))
    
    return data

@st.cache_data
def plot_spectrum(data, _x_names):
    # Create a list of Scatter objects
    traces = [go.Scatter(x=_x_names, y=row[_x_names],
                         mode='lines', name=index)
              for index, row in data.iterrows()]
    
    # Create the figure and add all traces at once
    fig = go.Figure(data=traces)
    
    # Update layout
    fig.update_layout(title='Spectrum Plot',
                      xaxis_title='Spectrum',
                      yaxis_title='Value',
                      showlegend=False)
    
    return fig

@st.cache_data
def plot_spectrum2(data: pd.DataFrame, _x_names):
    df_t = data[_x_names].T
    print(df_t.columns)
    df_t.index = _x_names
    df_t = df_t.reset_index()
    fig = px.line(df_t, x='index', y=df_t.columns[1:], title='Spectra Plot')

    # Update the layout of the plot
    fig.update_layout(
        xaxis_title='X-axis',
        yaxis_title='Intensity',
        showlegend=False
    )

    # Show the figure
    #fig.show()
    return fig

#@st.cache_data(hash_funcs={pd.DataFrame: lambda _: None}, persist='disk')
@st.cache_data(persist='disk')
def msc(input_data):
    """
    Perform Multiplicative Scatter Correction (MSC) on spectral data.
    
    Parameters:
    input_data (pandas.DataFrame): DataFrame where each row is a spectrum and each column is a wavelength.
    
    Returns:
    pandas.DataFrame: MSC-corrected spectral data.
    """
    # Convert input DataFrame to numpy array
    data_array = input_data.values
    
    # Mean spectrum of the input data
    mean_spectrum = np.mean(data_array, axis=0)
    
    # Initialize array to store corrected spectra
    corrected_data = np.zeros_like(data_array)
    
    # Apply MSC to each spectrum
    for i in range(data_array.shape[0]):
        spectrum = data_array[i, :]
        # Perform least squares linear regression
        fit = np.polyfit(mean_spectrum, spectrum, 1, full=True)
        slope = fit[0][0]
        intercept = fit[0][1]
        
        # Correct the spectrum
        corrected_spectrum = (spectrum - intercept) / slope
        corrected_data[i, :] = corrected_spectrum
    
    # Convert corrected data array back to DataFrame
    corrected_df = pd.DataFrame(corrected_data, index=input_data.index, columns=input_data.columns)
    
    return corrected_df

def msc2(input_data, reference=None):
    ''' Perform Multiplicative scatter correction'''

    # Baseline correction
    for i in range(input_data.shape[0]):
        input_data[i,:] -= input_data[i,:].mean()

    # Get the reference spectrum. If not given, estimate from the mean    
    if reference is None:    
        # Calculate mean
        matm = np.mean(input_data, axis=0)
    else:
        matm = reference

    # Define a new data matrix and populate it with the corrected data    
    output_data = np.zeros_like(input_data)
    for i in range(input_data.shape[0]):
        # Run regression
        fit = np.polyfit(matm, input_data[i,:], 1, full=True)
        # Apply correction
        output_data[i,:] = (input_data[i,:] - fit[0][1]) / fit[0][0] 

    return output_data

#@st.cache_data
def tt_split(data,_x_names,_to_predict_label):
    #data_to_train = data_dict.get(to_predict_label) #use specific feature to train
    y = data.loc[:,_to_predict_label].to_numpy()
    X = data.loc[:,_x_names].to_numpy()
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return x_train,x_test,y_train,y_test

@st.cache_data
def normalize_y(y_train,y_test,_scaler):
    y_train = _scaler.fit_transform(y_train.reshape(-1,1))
    y_test = _scaler.transform(y_test.reshape(-1,1))
    return y_train,y_test,_scaler

#@st.cache_data(experimental_allow_widgets=True)
def sav_tuning_1():
    with st.container():
        first_input_smp = st.slider(label='Smoothing Points',min_value=3,max_value=201,step=2,value=3,key='s_first',on_change=clear_cache)
        col1,col2= st.columns(2)
        with col1:
            first_input_ponm = st.selectbox(label='Polynomial Order',options=list(range(0,first_input_smp//2)),key='p_first',index=0,on_change=clear_cache)
        with col2:
            if first_input_ponm != 0:
                first_input_dev = st.selectbox(label='Derivative',options=[x for x in range(0,first_input_ponm+1)],key='d_first',index=1,on_change=clear_cache)
            else:
                first_input_dev = st.selectbox(label='Derivative',options=['Not Available'],key='d_first',index=0,disabled=True, on_change=clear_cache)
    return first_input_dev,first_input_ponm,first_input_smp

#@st.cache_data(experimental_allow_widgets=True)
def sav_tuning_2():
    with st.container():
        second_input_smp = st.slider(label='Smoothing Points',min_value=3,max_value=201,step=2,value=147,key='s_second',on_change=clear_cache)
        col1,col2 = st.columns(2)
        with col1:
            second_input_ponm = st.selectbox(label='Polynomial Order',options=list(range(0,second_input_smp//2)),key='p_second',index=2,on_change=clear_cache)
        with col2:
            if second_input_ponm != 0:
                second_input_dev = st.selectbox(label='Derivative',options=[x for x in range(0,second_input_ponm+1)],key='d_second',index=1,on_change=clear_cache)
            else:
                second_input_dev = st.selectbox(label='Derivative',options=['Not Available'],key='d_second',index=0,disabled=True, on_change=clear_cache)
    return second_input_dev,second_input_ponm,second_input_smp

def clear_cache():
    #st.write('ON CHANGE!!!')
    #st.cache_data.clear()
    pass