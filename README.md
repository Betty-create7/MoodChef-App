# MoodChef App

## Overview
"MoodChef is a web-based application that recommends cuisines and music based on the user's current mood and uses geolocation to find nearby restaurants serving those meals."

## Features
**ML-Powered Recommendations:** Uses a trained Random Forest Classifier to suggest food and music tailored to the user's selected emotional state.
**Geolocation Integration:** Finds nearby restaurants serving the recommended cuisine.
**Spotify Integration:** Generates mood-matched music playlists dynamically.
**Secure Configuration:** Uses environment variables (`.env`) to securely manage API keys.

## Tech Stack
**Framework:** Python (Streamlit)
**Machine Learning:** Scikit-Learn (Random Forest Classifier), Joblib
**Data Handling:** Pandas, Requests
**APIs Used:** Spoonacular API(food), Foursquare API(location) and Spotify API(music)

## How to Run Locally
1. Clone the repository: `git clone https://github.com/bettycreate-7/MoodChef-App.git`
2. Navigate to the folder: `cd [Moodchef]`
3. Install dependencies: `pip install -r requirements.txt OR npm install`
4. Run the app: `python app.py OR npm start`

## 📝 Status & Future Improvements
This project is currently a work in progress. Future updates will include deploying to a live server, adding user reviews, etc.
