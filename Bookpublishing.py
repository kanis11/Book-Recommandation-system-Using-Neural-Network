# [pandas]
import pandas as pd
import numpy as np
from sklearn import model_selection
from sklearn.calibration import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

#[Data visualizaion Libraries]
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras 

import streamlit as st
import re
import PIL
from PIL import Image
# [SQL libraries]
import mysql.connector
import joblib

# -------------------------------This is the configuration page for our Streamlit Application---------------------------

st.set_page_config(page_title="Publishing Industry Application",page_icon="📚",
                   layout="wide",
                   )
# -------------------------------This is the main tabs in a Streamlit application, helps in navigation--------------------

st.write("""
<div style='text-align:center'>
    <h1 style='color:#009999;'>Welcome to Online Book Store</h1>
</div>
""", unsafe_allow_html=True)
tab1,tab2,tab3,tab4 = st.tabs(["ABOUT","Book prediction", "Show Result","Book Analysis"]) 


# -----------------------------------------------About Project Section--------------------------------------------------

with tab1:
    im1 = Image.open("book5.jpg")
    st.image(im1, width=1200)

    col1, col2 = st.columns(2)

    with col1:
        im = Image.open("book4.png")
        st.markdown("")
        st.image(im)

    with col2:

        st.markdown("Learning: Books are a source of information that can teach us new things and help us understand the world.")
        st.markdown("Personal growth: Books can help with personal development and can be companions that nurture the mind.")
        st.markdown("Academic performance: Reading books can enhance academic performance.")


        st.markdown("Stress relief: Books can provide comfort and a sense of solace during stressful times.")
        st.header(":rainbow[Online Bookstores can offer a Number of Benefits:]",divider=True)
       
        st.markdown("Convenience: Customers can shop for books from home, without having to visit a physical store")
        st.markdown("Price: Online bookstores can sometimes offer lower prices than traditional bookstores")
        st.markdown("Selection: Online bookstores can offer a wider selection of books than traditional bookstores")
        
        
## ====================================================   /     Book prediction    /   ================================================= #   
         
with tab2:
    # Load dataset (this is an example, you should load your actual data)
    data = pd.read_csv("C:/Users/91790/Music/book.csv")

    # Simple preprocessing: remove duplicates, handle missing values
    data = data.dropna(subset=['author_name', 'language_name', 'title'])

# Function to get book recommendations
    def get_recommendations(author_name, language_name):
            # Filter data based on input values
            filtered_books = data[(data['author_name'].str.contains(author_name, case=False)) & 
                           (data['language_name'].str.contains(language_name, case=False))]
    
            if filtered_books.empty:
                return "No books found for this author and language."

            # Using a simple approach, recommend books based on the author's and language's popularity
            recommended_books = filtered_books[['title', 'author_name', 'language_name']].head(10)
            return recommended_books

            # Streamlit UI
    st.title("Book Recommendation System")
    st.header("Enter author and language to get book recommendations")

    author_input = st.text_input("Enter Author Name:")
    language_input = st.text_input("Enter Language:")


    if st.button("Get Recommendations"):
            recommendations = get_recommendations(author_input, language_input)
            if isinstance(recommendations, str):
                st.write(recommendations)
            else:
                st.write("Recommended Books:")
                st.dataframe(recommendations)
                
## ====================================================   /     Show Result  /   ================================================= #   

with tab3:       
# Streamlit app
        st.title("Result")
    # Load dataset
        file = pd.read_csv("C:/Users/91790/Music/book1.csv")
       # Show the DataFrame in Streamlit
        st.write(" Result form AWS")
        st.dataframe(file)  # Display the dataframe as an interactive table

    # Optionally, show some basic statistics
        st.write("Summary of the data:")
        st.write(file.describe())  # Displays basic statistics about the dataset
        # Display the original data in Streamlit
        st.subheader("Original Data")
        st.write(file)

        # Apply Label Encoding to 'author_name' and 'language_name'
        label_encoder_author = LabelEncoder()
        label_encoder_language = LabelEncoder()

        # Transform the categorical columns
        file['author_encoded'] = label_encoder_author.fit_transform(file['output'])
        file['language_encoded'] = label_encoder_language.fit_transform(file['input'])

        # Display the encoded data
        st.subheader("Encoded Data")
        st.write(file)


        # Adding labels for better clarity
        for i in range(len(file)):
            plt.text(file['author_encoded'][i], file['language_encoded'][i], f'({file["output"][i]}, {file["input"][i]})', fontsize=9)

        
## ====================================================   /     Book Analysis    /   ================================================= #   
        

with tab4:
# Title of the app
        st.title("Book Dataset Visualization")

# File uploader widget to upload CSV file
        uploaded_file = st.file_uploader("C:/Users/91790/Music/book.csv", type="csv")

        if uploaded_file is not None:
        # Read the uploaded CSV file into a Pandas DataFrame
            df = pd.read_csv(uploaded_file)

        # Show the raw data
            st.write("### Data from CSV file:")
            st.table(df)  # Display the dataframe as an interactive table

        # Display the number of unique authors, languages, and publishers
            st.write("### Number of Unique Authors:")
            st.write(df['author_name'].nunique())  # Show the number of unique authors

            st.write("### Number of Unique Languages:")
            st.write(df['language_name'].nunique())  # Show the number of unique languages

            st.write("### Number of Unique Publishers:")
            st.write(df['publisher_name'].nunique())  # Show the number of unique publishers

        # Visualizations
            st.write("### Visualizations:")

        # Bar chart for the number of books per author
            author_count = df['author_name'].value_counts()
            st.write("#### Number of Books per Author:")
            st.bar_chart(author_count)

        # Bar chart for the number of books per language
            language_count = df['language_name'].value_counts()
            st.write("#### Number of Books per Language:")
            st.bar_chart(language_count)

        # Bar chart for the number of books per publisher
            publisher_count = df['publisher_name'].value_counts()
            st.write("#### Number of Books per Publisher:")
            st.bar_chart(publisher_count)

        # Optional: Pie charts for categorical distribution
            st.write("#### Pie Chart for Authors Distribution:")
            fig, ax = plt.subplots()
            author_count.plot.pie(ax=ax, autopct='%1.1f%%', startangle=90)
            ax.set_ylabel('')  # To hide the label
            st.pyplot(fig)

            st.write("#### Pie Chart for Languages Distribution:")
            fig, ax = plt.subplots()
            language_count.plot.pie(ax=ax, autopct='%1.1f%%', startangle=90)
            ax.set_ylabel('')
            st.pyplot(fig)

            st.write("#### Pie Chart for Publishers Distribution:")
            fig, ax = plt.subplots()
            publisher_count.plot.pie(ax=ax, autopct='%1.1f%%', startangle=90)
            ax.set_ylabel('')
            st.pyplot(fig)

st.write( f'<h6 style="color:rgb(0, 153, 153,0.35);">App Created by kani</h6>', unsafe_allow_html=True )  
