import streamlit as st
import pandas as pd
import numpy as np
import nltk
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline
import joblib
#nltk.download('stopwords')
st.title('Sentimental Analysis of Customer Feedback ')
#punctuation = ["""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""]
    
#@st.cache
#def remove_punct_stopwords(message):
 #   form_str = [char for char in message if char not in punctuation]
  #  form_str_join = ''.join(form_str)
   # words_stop = nltk.corpus.stopwords.words('english')
   # form_str_stop = [word for word in form_str_join.split() if word.lower() not in words_stop]
  #  return form_str_stop


sent_model = joblib.load('summary_svm_model.joblib')
vectorizer = joblib.load('CountVectorizer_summary.joblib')
inp_text = st.text_area('Feedback of item you bought',height=200)

vectorised_text = vectorizer.transform([inp_text])
pred = ''
# add a placeholder

def sent_predict(inp_text):
    prediction = sent_model.predict(inp_text)
    if prediction == 0:
        pred = 'Positive'
    else:
        pred = 'Negative'
    return pred
if st.button('Submit'):
    st.write('Feedback is:',sent_predict(vectorised_text))
   
