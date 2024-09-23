import streamlit as st
import membership as ms

def register_callback():
    if st.session_state['register'] == False:
        st.session_state['register'] = True
    else:
        st.session_state['register'] = False

    try:
        if st.experimental_get_query_params()['code'][0] == 'logged_in' and not st.session_state['logout']:
            st.session_state['login'] = True
            #print(6)
        else:
            st.session_state['login'] = False
    except:
        print('Error regist')
        pass

def register_page():
    with st.form(key='register_form'):
        st.title('Register')
        fname = st.text_input('',placeholder='Firstname')
        lname = st.text_input('',placeholder='Lastname')
        username = st.text_input('',placeholder='Username')
        password = st.text_input('',placeholder='Password',type='password')
        register_button = st.form_submit_button('Register')
        if register_button and len(fname) != 0 and len(lname) != 0 and len(username) != 0 and len(password) != 0:
            ms.register(fname,lname,username,password)
    st.text('Already have an acoount? Log in.')
    if st.button('Go to Login Page',on_click=register_callback):
        st.session_state['register'] = False