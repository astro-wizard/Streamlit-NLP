# import core pkg
import streamlit as st
#EDA pkg
import pandas as pd
import numpy as np
#util pkg
import joblib

# Function
pipe_lr = joblib.load(open("models/emotion_clf.pkl","rb"))
def predict_emotions(docx):
    results = pipe_lr.predict([docx])
    return results[0]

#def get_prediction_proba([docx]):
 #   results = pipe_lr.predict_proba([docx])
  #  return results
    
    
def main():
    menu = ["Home","Monitor","About"]
    choice = st.sidebar.selectbox("Menu",menu)
    
    if choice == "Home":
        st.subheader("Home-Emotion In Text")
        
        with st.form(key='emotion_clf_form'):
            raw_text = st.text_area("Type Here")
            submit_text = st.form_submit_button(label='Submit')
            
        if submit_text:
            col1,col2 = st.columns(2)
            # Apply function here
            prediction = predict_emotions(raw_text)
            #probability = get_prediction_proba(raw_text)
            
            with col1:
                st.success("Original Text")
                st.write(raw_text)
                
                st.success("Prediction")
                st.write(prediction)
            with col2:
                st.success("Prediction Probability")
             #   st.write(probability)
    elif choice == "Monitor":
        st.subheader("Monitor App")
    else:
        st.subheader("About")
        
if __name__ =='__main__':
    main()
    
