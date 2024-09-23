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
        print('Error login')
        pass


def login_page():
    st.title('Welcome to milk quality predictor!')
    st.experimental_set_query_params()
    st.session_state['logout'] = False
    with st.form(key='login_form'):
        st.subheader('Login')
        username = st.text_input('',placeholder='Username')
        password = st.text_input('',placeholder='Password',type='password')
        login_buton = st.form_submit_button('Login')
        if login_buton and len(username) !=0 and len(password) != 0:
            succ = ms.login(username,password)
            #st.write(succ)
            if succ:
                st.session_state['login'] = True
                st.experimental_rerun()
    st.text('Don\'t Have account? Sign up.')
    if st.button('Go to Register Page',on_click=register_callback):
        st.session_state['register'] = True