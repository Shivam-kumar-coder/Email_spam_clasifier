import pandas as pd  
import streamlit as st
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
df=pd.read_csv("spam.csv")
df['spam']=df['Category'].apply(lambda x:1 if x=='spam' else 0)
v=CountVectorizer()
x_train,x_test,y_train,y_test=train_test_split(df['Message'],df['spam'])
x_train_c=v.fit_transform(x_train.values)
x_train_c.toarray()
model=MultinomialNB()
model.fit(x_train_c,y_train)
#mail=["i am shivam kumar"]


st.title("Spam Classifier")
mail=st.text_input("Enter The  Content")
c=st.button("Check")
if c:
    #mai=list(mail)
    #st.write(mail)
    mail_c=v.transform([mail])
    a=model.predict(mail_c)
    #st.write(a)
    if (a==0):
        st.balloons()
        st.success("This content is not spam")
    elif (a==1):
        st.info("this content is spam")
        st.warning('This is a spam', icon="⚠️")
                
hide_menu = """
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}
</style>
"""
st.markdown(hide_menu, unsafe_allow_html=True)
