import streamlit as st
import pymysql
import bcrypt
import dotenv
import os
import Functions
import json
import logging
logging.basicConfig(level=logging.INFO)

dotenv.load_dotenv()
secret_name = "prod/milk-quality-prediction/mysql"
region_name = "us-east-1"
try: 
    sql_keys = json.loads(Functions.get_secret(secret_name=secret_name, region_name=region_name))
    user = sql_keys["SQL_USER"]
    pw = sql_keys["SQL_PASSWORD"]
    db_name = sql_keys["DB_NAME"]
    db_port = int(sql_keys["DB_PORT"])
except Exception as e:
    user = os.environ["SQL_USER"]
    pw = os.environ["SQL_PASSWORD"]
    db_name = os.environ["DB_NAME"]
    db_port = int(os.environ["DB_PORT"])

production = True
host = "107.20.196.80" if production else "localhost"

def connection():
    global conn,c
    logging.info(host + user + db_name + str(db_port))
    conn = pymysql.connect(host=host,user=user,passwd=pw,database=db_name,port=db_port)        
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
    user_data = c.fetchone()
    encoded_password = password.encode('utf-8')
    real_password = user_data[1].encode('utf-8')
    user_type = user_data[2]
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