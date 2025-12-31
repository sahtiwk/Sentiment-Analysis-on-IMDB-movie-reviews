import streamlit as st
import joblib
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer

st.set_page_config(
    page_title="Sentiment Analyzer",
    page_icon="🎬",
    layout="centered"
)

hide_st_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            .stTextArea textarea {
                font-size: 16px;
                border-radius: 10px;
            }
            .stButton>button {
                width: 100%;
                border-radius: 10px;
                height: 3em;
                background-color: #4CAF50; 
                color: white;
            }
            </style>
            """
st.markdown(hide_st_style, unsafe_allow_html=True)
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

@st.cache_resource
def load_resources():
    try:
        model = joblib.load('svm_rbf_sentiment_model.pkl')
        vectorizer = joblib.load('tfidf_vectorizer.pkl')
        return model, vectorizer
    except FileNotFoundError:
        return None, None

model, vectorizer = load_resources()

def preprocess_text(text):
    text = re.sub(r'<.*?>', '', text)
    text = text.lower()
    text = re.sub(r'(!)', r' \1 ', text)
    text = re.sub(r'[^a-z\s!]', '', text)
    words = text.split()
    stop_words = set(stopwords.words('english'))
    stemmer = SnowballStemmer('english')
    cleaned_words = [stemmer.stem(w) for w in words if w not in stop_words or w == '!']
    return ' '.join(cleaned_words)

if model is None:
    st.error("Error: Model files not found. Please place .pkl files in the app folder.")
    st.stop()

st.title("Movie Sentiment Analysis")
st.markdown("Enter a review below to analyze its emotional tone.")

st.write("")

# The input box
user_input = st.text_area(
    label="Review Text", 
    placeholder="Type your thoughts here...", 
    height=150,
    label_visibility="collapsed" 
)

st.write("")
if st.button("Analyze"):
    if user_input.strip() == "":
        st.caption("⚠️ Please enter text to analyze.")
    else:
        clean_text = preprocess_text(user_input)
        vec_input = vectorizer.transform([clean_text])
        prediction = model.predict(vec_input)[0]
        st.divider()
        if prediction == 1:
            st.markdown(
                """
                <div style="background-color: #d4edda; padding: 20px; border-radius: 10px; text-align: center;">
                    <h3 style="color: #155724; margin: 0;">Positive Sentiment</h3>
                </div>
                """, 
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                """
                <div style="background-color: #f8d7da; padding: 20px; border-radius: 10px; text-align: center;">
                    <h3 style="color: #721c24; margin: 0;">Negative Sentiment</h3>
                </div>
                """, 
                unsafe_allow_html=True
            )