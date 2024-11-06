import streamlit as st
from helper_functions import *  



st.set_page_config(page_title="Sentiment Analysis", layout="wide")

# Title and description
#st.title("Customer Product Reviews Sentiment Analysis App")
# app design
set_bg_hack('Picture1.png')
st.title("Team Members 🧑‍💻")

st.markdown("""
<style>
.team-members {
  background-color: #f2f2f2;
  padding: 20px;
  border-radius: 10px;
}

h2 {
  color: #333;
  text-align: center;
}

ul {
  list-style-type: none;
  padding: 0;
  margin: 0;
}

li {
  margin-bottom: 10px;
  font-size: 18px;
  font-weight: bold;
  color: #555;
}
</style>

<div class="team-members">
  <ul>
    <li>Mohamed Talaat Abo Elftouh</li>
    <li>Mohamed Mostafa Abdelhamed</li>
    <li>Moaz Mohamed Tawfik</li>
    <li>Amr Khaled Mostafa</li>
    <li>Mahmoud Mohammed Abdelmawgoud</li>
    <li>Mohamed Alaa Elsayad</li>
  </ul>
</div>
""", unsafe_allow_html=True)


st.sidebar.header("About App")
st.sidebar.info("A Customer Sentiment analysis Project which collect data of reviews of different domains. The reviews will then be used to determine the Sentiments of those reviews. \
                The different Visualizations will help us get a feel of the overall exploration of reviews")
st.sidebar.text("Built with Streamlit")

st.sidebar.header("For Any Queries/Suggestions Please reach out at :")
st.sidebar.info("medotalaat20177@gmail.com")