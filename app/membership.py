import streamlit as st
import pymysql
import bcrypt

# Connect

def connection():
    global conn,c
    conn = pymysql.connect(host="localhost",user="root",passwd="root",database="milk_quality_prediction",port=5555)        
    c = conn.cursor()
    
def register(name,surname,username,password):
    num = c.execute('SELECT username FROM user WHERE username = %s',(username))
    bytes = password.encode('utf-8')
    salt = bcrypt.gensalt()
    encoded_password = bcrypt.hashpw(bytes, salt)
    if num == 0:
        c.execute("INSERT INTO user (name,surname,username,password,user_type) VALUES (%s,%s,%s,%s,%s)",(name,surname,username,encoded_password,'user'))
        conn.commit()
        st.success('Registered Successfully')
    else:
        st.warning('Username Duplicated')

def login(username,password):
    successful = False
    global is_user
    c.execute('SELECT username,password,user_type FROM user WHERE username = %s',(username))
    #bytes = password.encode('utf-8')
    user_data = c.fetchone()
    encoded_password = password.encode('utf-8')
    real_password = user_data[1].encode('utf-8')
    #st.write(real_password)
    user_type = user_data[2]
    #st.write(bytes)
    #st.write(bcrypt.checkpw(encoded_password, real_password))
    if bcrypt.checkpw(encoded_password, real_password):
        st.session_state['login'] = True
        if user_type == 'admin':
            st.session_state['user'] = False
        else:
            st.session_state['user'] = True
        st.success('Logged-in Successfully')
        successful = True
    else:
        st.warning('Incorrect Username/Password')
    return successful
    #st.warning('Please Fill The Entry')