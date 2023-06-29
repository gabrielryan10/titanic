#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import streamlit as st
import pickle
import pandas as pd
filename = 'xgb_titanic_model.sav'
load_model = pickle.load(open(filename, 'rb'))


# In[3]:


def titanic_pipe(pclass,title,sibsp,parch):
    
    if title=='Mrs':
        title_num=1
    elif title=='Miss':
        title_num=2
    elif title=='Master':
        title_num=3
    elif title=='Mr':
        title_num=4
    else:
        title_num=-999
    
    if sibsp=='Yes':
        sibsp_num=1
    else:
        sibsp_num=2
    
    if parch=='Yes':
        parch_num=1
    else:
        parch_num=2
        
    return pclass,title_num,sibsp_num,parch_num

def titanic_pred(model_titanic,pclass,title_num,sibsp_num,parch_num):
    df=pd.DataFrame()
    df['pclass']=[pclass]
    df['title_num']=[title_num]
    df['sibsp_num']=[sibsp_num]
    df['parch_num']=[parch_num]
    
    pred=model_titanic.predict_proba(df)[:,1]
    
    if pred<=0.131:
        score=5
    elif pred>0.131 and pred<=0.30:
        score=4
    elif pred>0.30 and pred<=0.55:
        score=3
    elif pred>0.55 and pred<=0.91:
        score=2
    else:
        score=1
    
    return score, pred


# In[ ]:


def main():
    st.title("Titanic Survival Score Prediction")

    pclass = st.selectbox("Class of Passenpger", [1, 2, 3])
    title = st.selectbox("Name Title of Passenger", ["Mr", "Miss", "Mrs","Master"])
    sibsp = st.selectbox("Passenger has at least one spouse/sibling", ["No","Yes"])
    parch = st.selectbox("Passenger has at least one child/parent", ["No","Yes"])

    x1,x2,x3,x4=titanic_pipe(pclass,title,sibsp,parch)
    score=titanic_pred(load_model,x1,x2,x3,x4)[0]

    st.write('<p style="font-size:30px; color:red;">***Result:***</p>',unsafe_allow_html=True)
    st.write(f'<p style="font-size:30px; color:red;">Survival Score is *{score}*</p>',unsafe_allow_html=True)
    st.write('<p style="font-size:16px; color:black;">Score 1: highest chance of survival</p>',unsafe_allow_html=True)
    st.write('<p style="font-size:16px; color:black;">Score 2: high chance of survival</p>',unsafe_allow_html=True)
    st.write('<p style="font-size:16px; color:black;">Score 3: medium chance of survival</p>',unsafe_allow_html=True)
    st.write('<p style="font-size:16px; color:black;">Score 4: low chance of survival</p>',unsafe_allow_html=True)
    st.write('<p style="font-size:16px; color:black;">Score 5: lowest chance of survival</p>',unsafe_allow_html=True)


if __name__ == "__main__":
    main()

