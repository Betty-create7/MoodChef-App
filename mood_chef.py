# Import necessary libraries for the application
import streamlit as st  # Web app framework for creating the UI
import requests  # For making HTTP requests to APIs
import os  # For interacting with the operating system, like file paths
import time  # For handling time-related tasks, like sleep for retries
import pandas as pd  # Data manipulation library, used for training data
from dotenv import load_dotenv  # For loading environment variables from .env file
from sklearn.ensemble import RandomForestClassifier  # Machine learning model for mood prediction
import joblib  # For saving and loading ML models
import spotipy  # Spotify API client
from spotipy.oauth2 import SpotifyClientCredentials  # OAuth for Spotify authentication
from typing import Dict, List, Any, Optional  # Type hints for better code readability
import logging  # For logging messages and errors

# Configure logging to display INFO level messages and above
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)  # Create a logger for this module

# Load API Keys from .env file for secure configuration
load_dotenv()

# Define a Config class to manage API keys and validation
class Config:
    """Configuration management"""
    # Retrieve API keys from environment variables
    SPOONACULAR_API_KEY = os.getenv("SPOONACULAR_API_KEY")
    FOURSQUARE_API_KEY = os.getenv("FOURSQUARE_API_KEY") 
    SPOTIFY_CLIENT_ID = os.getenv("SPOTIFY_CLIENT_ID")
    SPOTIFY_CLIENT_SECRET = os.getenv("SPOTIFY_CLIENT_SECRET")
    
    @classmethod
    def validate_keys(cls) -> Dict[str, bool]:
        """Validate that all required API keys are present"""
        # Check if each API key is set and return a dictionary of booleans
        return {
            "spoonacular": bool(cls.SPOONACULAR_API_KEY),
            "foursquare": bool(cls.FOURSQUARE_API_KEY),
            "spotify": bool(cls.SPOTIFY_CLIENT_ID and cls.SPOTIFY_CLIENT_SECRET)
        }

# Define a class for predicting cuisines based on mood, using a mapping and ML model
class MoodCuisinePredictor:
    """Improved mood to cuisine mapping with API-friendly search terms"""
    
    def __init__(self):
        # Initialize a dictionary mapping moods (with emojis) to lists of cuisine suggestions
        # These are general terms suitable for API queries
        self.mood_cuisine_map = {
            "😊 happy": ["Rice dishes", "Italian pasta", "Mediterranean"],
            "😢 sad": ["Soup", "Italian comfort food", "Chinese noodles"],
            "😴 tired": ["Quick meals", "American fast food", "Asian noodles"],
            "😰 stressed": ["Salad", "Mediterranean healthy", "Grilled fish"],
            "😌 relaxed": ["Barbecue", "French cuisine", "Japanese"],
            "⚡ energetic": ["Spicy food", "Mexican spicy", "Korean spicy"],
            "💕 romantic": ["Fine dining", "French romantic", "Italian romantic"],
            "🌟 adventurous": ["African", "Ethiopian", "Moroccan fusion"]
        }
        
        # Set the file path for saving/loading the trained ML model
        self.model_path = "improved_mood_model.pkl"
        # Load or train the model upon initialization
        self.model = self._load_or_train_model()
    
    # Private method to create training data for the ML model
    def _create_training_data(self) -> pd.DataFrame:
        """Create more comprehensive training data"""
        training_data = []
        
        # Loop through the mood-cuisine map to create data points
        # For each mood, extract the clean mood name (without emoji) and pair with each cuisine
        for mood_key, cuisines in self.mood_cuisine_map.items():
            clean_mood = mood_key.split()[-1]  # Extract mood name without emoji
            for cuisine in cuisines:
                training_data.append({"mood": clean_mood, "cuisine": cuisine})
        
        # Convert the list of dictionaries to a pandas DataFrame
        return pd.DataFrame(training_data)
    
    # Private method to load an existing model or train a new one if loading fails
    def _load_or_train_model(self):
        """Load existing model or train new one"""
        # Check if the model file exists
        if os.path.exists(self.model_path):
            try:
                # Attempt to load the model using joblib
                return joblib.load(self.model_path)
            except Exception as e:
                # Log a warning if loading fails and proceed to train a new model
                logger.warning(f"Failed to load model: {e}. Training new model.")
        
        # If no model exists or loading failed, train a new one
        return self._train_model()
    
    # Private method to train the RandomForestClassifier model
    def _train_model(self):
        """Train the mood prediction model"""
        # Create the training data
        data = self._create_training_data()
        # One-hot encode the moods for input features
        X = pd.get_dummies(data["mood"])
        # Set the target variable as the cuisine
        y = data["cuisine"]
        
        # Initialize the RandomForestClassifier with 100 estimators and a fixed random state for reproducibility
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        # Fit the model to the data
        model.fit(X, y)
        
        # Save the trained model to disk using joblib
        joblib.dump(model, self.model_path)
        # Log success message
        logger.info("Model trained and saved successfully")
        
        # Return the trained model
        return model
    
    # Public method to predict cuisines for a given mood
    def predict_cuisines(self, mood: str, n_suggestions: int = 3) -> List[str]:
        """Predict multiple cuisine suggestions for a mood"""
        # Handle moods that may include emojis by cleaning them
        clean_mood = mood
        # First, check if the mood matches directly in the map
        if mood in self.mood_cuisine_map:
            return self.mood_cuisine_map[mood][:n_suggestions]
        
        # If not, try matching the text part of the mood
        for mood_key in self.mood_cuisine_map.keys():
            if mood.split()[-1] in mood_key or mood_key.split()[-1] == mood:
                return self.mood_cuisine_map[mood_key][:n_suggestions]
        
        # If no direct match, fall back to ML model prediction
        try:
            # Clean the mood for model input (remove emoji if present)
            clean_mood = mood.split()[-1] if " " in mood else mood
            # Create a one-hot encoded DataFrame for the input mood
            mood_df = pd.get_dummies(pd.Series([clean_mood]))
            # Get all possible mood columns from the map for reindexing
            all_moods = [key.split()[-1] for key in self.mood_cuisine_map.keys()]
            all_cols = pd.get_dummies(pd.Series(all_moods)).columns
            # Reindex to match training data structure, filling missing with 0
            mood_df = mood_df.reindex(columns=all_cols, fill_value=0)
            
            # Predict the cuisine using the model
            prediction = self.model.predict(mood_df)[0]
            # Return as a list with one suggestion
            return [prediction]
        except Exception as e:
            # Log error if prediction fails
            logger.error(f"Prediction failed: {e}")
            # Fallback to a default cuisine
            return ["Italian"]  # Safe fallback

# Define a class for handling API requests with caching and error handling
class APIClient:
    """Centralized API client with improved error handling"""
    
    # Static method to fetch recipes from Spoonacular API, cached for 1 hour
    @staticmethod
    @st.cache_data(ttl=3600)  # Cache for 1 hour
    def fetch_recipes(query: str, allergies: str = "", limit: int = 5) -> Dict[str, Any]:
        """Fetch recipes with improved error handling and debugging"""
        # Check if API key is configured
        if not Config.SPOONACULAR_API_KEY:
            return {"error": "Spoonacular API key not configured"}
        
        # Set the base URL for Spoonacular's complex search endpoint
        base_url = "https://api.spoonacular.com/recipes/complexSearch"
        # Prepare parameters for the API request
        params = {
            "query": query,
            "number": min(limit, 12),  # Spoonacular free tier limit
            "addRecipeInformation": True,
            "fillIngredients": True,
            "instructionsRequired": True,
            "apiKey": Config.SPOONACULAR_API_KEY
        }
        
        # Add intolerances parameter if allergies are provided
        if allergies.strip():
            params["intolerances"] = allergies.strip()
        
        # Retry the request up to 3 times in case of failures
        for attempt in range(3):
            try:
                # Log the query being fetched
                logger.info(f"Fetching recipes for: {query}")
                # Make the GET request with a 30-second timeout
                response = requests.get(base_url, params=params, timeout=30)
                
                # Log the response status
                logger.info(f"Spoonacular API Response Status: {response.status_code}")
                
                # Handle specific HTTP error codes
                if response.status_code == 401:
                    return {"error": "Invalid Spoonacular API key - please check your API key"}
                elif response.status_code == 402:
                    return {"error": "Spoonacular API quota exceeded - upgrade your plan"}
                elif response.status_code == 403:
                    return {"error": "Spoonacular API access forbidden - check your API key permissions"}
                elif response.status_code == 404:
                    return {"error": "Spoonacular API endpoint not found"}
                elif response.status_code != 200:
                    # Get a snippet of error text if available
                    error_text = response.text[:200] if response.text else "Unknown error"
                    # Log the error
                    logger.error(f"Spoonacular API error: {response.status_code} - {error_text}")
                    return {"error": f"Spoonacular API error {response.status_code}: {error_text}"}
                
                try:
                    # Parse the JSON response
                    data = response.json()
                except ValueError as e:
                    # Handle invalid JSON
                    return {"error": f"Invalid JSON response from Spoonacular: {str(e)}"}
                
                # Log the number of recipes found
                logger.info(f"Found {len(data.get('results', []))} recipes")
                
                # Initialize list to hold processed recipes
                recipes = []
                # Process each recipe in the results
                for r in data.get("results", []):
                    try:
                        # Build list of ingredients from extendedIngredients
                        ingredients = []
                        for ingredient in r.get("extendedIngredients", []):
                            if ingredient.get("original"):
                                ingredients.append(ingredient["original"])
                        
                        # Build list of instructions from analyzedInstructions
                        instructions = []
                        for inst_group in r.get("analyzedInstructions", []):
                            for step in inst_group.get("steps", []):
                                if step.get("step"):
                                    instructions.append({
                                        "number": step.get("number", len(instructions) + 1),
                                        "step": step["step"]
                                    })
                        
                        # Create a dictionary for the recipe details
                        recipe = {
                            "title": r.get("title", "Unknown Recipe"),
                            "image": r.get("image", ""),
                            "ready_in_minutes": r.get("readyInMinutes", "N/A"),
                            "servings": r.get("servings", "N/A"),
                            "source_url": r.get("sourceUrl", r.get("spoonacularSourceUrl", "")),
                            "ingredients": ingredients,
                            "instructions": instructions,
                            "summary": r.get("summary", "")
                        }
                        
                        # Add the recipe to the list
                        recipes.append(recipe)
                        
                    except Exception as e:
                        # Log warning if processing a recipe fails
                        logger.warning(f"Error processing recipe: {e}")
                        continue
                
                # Return the list of recipes
                return {"recipes": recipes}
                
            except requests.exceptions.Timeout:
                # Log timeout and retry if attempts remain
                logger.warning(f"Timeout on attempt {attempt + 1}")
                if attempt == 2:
                    return {"error": "Request timeout - Spoonacular API took too long to respond"}
            except requests.exceptions.ConnectionError:
                # Log connection error and retry if attempts remain
                logger.warning(f"Connection error on attempt {attempt + 1}")
                if attempt == 2:
                    return {"error": "Cannot connect to Spoonacular API - check your internet connection"}
            except requests.exceptions.RequestException as e:
                # Log general request error
                logger.error(f"Request error: {e}")
                if attempt == 2:
                    return {"error": f"Network error: {str(e)}"}
            except Exception as e:
                # Log unexpected errors
                logger.error(f"Unexpected error: {e}")
                return {"error": f"Unexpected error: {str(e)}"}
                
            # Exponential backoff before next retry
            time.sleep(2 ** attempt)  # Exponential backoff
        
        # Return error if all attempts fail
        return {"error": "Failed to fetch recipes after multiple attempts"}
    
    # Static method to fetch music from Spotify API, cached for 30 minutes
    @staticmethod
    @st.cache_data(ttl=1800)  # Cache for 30 minutes
    def fetch_music(query: str, limit: int = 5) -> Dict[str, Any]:
        """Fetch music with improved error handling"""
        # Check if Spotify credentials are configured
        if not (Config.SPOTIFY_CLIENT_ID and Config.SPOTIFY_CLIENT_SECRET):
            return {"error": "Spotify credentials not configured"}
        
        try:
            # Set up Spotify authentication manager
            auth_manager = SpotifyClientCredentials(
                client_id=Config.SPOTIFY_CLIENT_ID,
                client_secret=Config.SPOTIFY_CLIENT_SECRET
            )
            # Create Spotify client
            sp = spotipy.Spotify(auth_manager=auth_manager)
            
            # Prepare queries for Afrobeats and international music
            afrobeats_query = f"afrobeats {query} nigerian music"
            international_query = f"{query} chill relaxing"
            
            # Search for playlists based on mood
            playlist_results = sp.search(q=f"{query} mood playlist", limit=1, type="playlist")
            # Search for Afrobeats tracks
            afrobeats_results = sp.search(q=afrobeats_query, limit=2, type="track")
            # Search for international tracks
            track_results = sp.search(q=international_query, limit=limit-3, type="track")
            
            # Initialize list for music recommendations
            recommendations = []
            
            # Add playlists if found, with error checking
            if playlist_results and playlist_results.get("playlists") and playlist_results["playlists"].get("items"):
                for playlist in playlist_results["playlists"]["items"]:
                    if playlist and playlist.get("name"):
                        # Append playlist details
                        recommendations.append({
                            "type": "playlist",
                            "name": playlist["name"],
                            "description": playlist.get("description", ""),
                            "url": playlist.get("external_urls", {}).get("spotify", ""),
                            "owner": playlist.get("owner", {}).get("display_name", "Unknown")
                        })
            
            # Add Afrobeats/Nigerian tracks if found, with error checking
            if afrobeats_results and afrobeats_results.get("tracks") and afrobeats_results["tracks"].get("items"):
                for track in afrobeats_results["tracks"]["items"]:
                    if track and track.get("name") and track.get("artists"):
                        # Append track details
                        recommendations.append({
                            "type": "track",
                            "name": track["name"],
                            "artist": track["artists"][0].get("name", "Unknown Artist") if track["artists"] else "Unknown Artist",
                            "url": track.get("external_urls", {}).get("spotify", ""),
                            "preview_url": track.get("preview_url"),
                            "genre": "Afrobeats/Nigerian"
                        })
            
            # Add international tracks if found, with error checking
            if track_results and track_results.get("tracks") and track_results["tracks"].get("items"):
                for track in track_results["tracks"]["items"]:
                    if track and track.get("name") and track.get("artists"):
                        # Append track details
                        recommendations.append({
                            "type": "track",
                            "name": track["name"],
                            "artist": track["artists"][0].get("name", "Unknown Artist") if track["artists"] else "Unknown Artist",
                            "url": track.get("external_urls", {}).get("spotify", ""),
                            "preview_url": track.get("preview_url"),
                            "genre": "International"
                        })
            
            # Return the list of recommendations
            return {"music": recommendations}
            
        except Exception as e:
            # Log and return Spotify API errors
            logger.error(f"Spotify API error: {e}")
            return {"error": f"Spotify error: {str(e)}"}
    
    # Static method to fetch restaurants from Foursquare API, cached for 30 minutes
    @staticmethod
    @st.cache_data(ttl=1800)  # Cache for 30 minutes
    def fetch_restaurants(cuisine: str, location: str, limit: int = 5) -> Dict[str, Any]:
        """Fetch restaurants using the new Foursquare Places API"""
        # Check if API key is configured
        if not Config.FOURSQUARE_API_KEY:
            return {"error": "Foursquare API key not configured"}
        
        # Set the API endpoint URL
        url = "https://places-api.foursquare.com/places/search"
        # Prepare headers with authorization
        headers = {
            "Authorization": f"Bearer {Config.FOURSQUARE_API_KEY}",
            "Accept": "application/json",
            "X-Places-Api-Version": "2025-06-17"
        }
        
        # Create a search query combining cuisine with "restaurant"
        search_query = f"{cuisine} restaurant"
        
        # Prepare parameters for the request
        params = {
            "query": search_query,
            "near": location,
            "limit": limit,
            "categories": "13065"  # Restaurant category ID
        }
        
        # Retry the request up to 3 times
        for attempt in range(3):
            try:
                # Log the search details
                logger.info(f"Searching for {search_query} near {location}")
                # Make the GET request with timeout
                response = requests.get(url, headers=headers, params=params, timeout=15)
                
                # Handle specific HTTP error codes
                if response.status_code == 401:
                    return {"error": "Invalid Foursquare API key"}
                elif response.status_code == 403:
                    return {"error": "Foursquare API access forbidden - check your API key permissions"}
                elif response.status_code == 429:
                    return {"error": "Foursquare API rate limit exceeded"}
                elif response.status_code != 200:
                    # Log error response
                    logger.error(f"Foursquare API error: {response.status_code} - {response.text}")
                    return {"error": f"API error: {response.status_code}"}
                
                # Parse the JSON response
                data = response.json()
                # Initialize list for restaurants
                restaurants = []
                
                # Parse each result in the new API structure
                for r in data.get("results", []):
                    location_info = r.get("location", {})
                    
                    # Use formatted_address if available, otherwise build from components
                    address = location_info.get("formatted_address", "")
                    if not address:
                        address_parts = []
                        if location_info.get("address"):
                            address_parts.append(location_info["address"])
                        if location_info.get("locality"):
                            address_parts.append(location_info["locality"])
                        if location_info.get("region"):
                            address_parts.append(location_info["region"])
                        address = ", ".join(address_parts) if address_parts else "Address not available"
                    
                    # Extract categories from the result
                    categories = []
                    for cat in r.get("categories", []):
                        if cat.get("name"):
                            categories.append(cat["name"])
                    
                    # Create a dictionary for restaurant details
                    restaurant = {
                        "name": r.get("name", "Unknown Restaurant"),
                        "address": address,
                        "distance": r.get("distance"),
                        "rating": None,  # Not in the provided response structure
                        "price": None,   # Not in the provided response structure
                        "categories": categories,
                        "phone": r.get("tel", ""),
                        "website": r.get("website", ""),
                        "fsq_place_id": r.get("fsq_place_id", "")
                    }
                    # Add to the list
                    restaurants.append(restaurant)
                
                # Log the number found
                logger.info(f"Found {len(restaurants)} restaurants")
                # Return the list
                return {"restaurants": restaurants}
                
            except requests.exceptions.Timeout:
                # Log timeout and retry if attempts remain
                logger.warning(f"Timeout on attempt {attempt + 1}")
                if attempt == 2:
                    return {"error": "Request timeout - Foursquare API took too long to respond"}
            except requests.exceptions.ConnectionError:
                # Log connection error and retry if attempts remain
                logger.warning(f"Connection error on attempt {attempt + 1}")
                if attempt == 2:
                    return {"error": "Cannot connect to Foursquare API - check your internet connection"}
            except requests.exceptions.RequestException as e:
                # Log general request error
                logger.error(f"Request error: {e}")
                if attempt == 2:
                    return {"error": f"Network error: {str(e)}"}
            except Exception as e:
                # Log unexpected errors
                logger.error(f"Unexpected error: {e}")
                return {"error": f"Unexpected error: {str(e)}"}
                
            # Exponential backoff
            time.sleep(2 ** attempt)  # Exponential backoff
        
        # Return error if all attempts fail
        return {"error": "Failed to fetch restaurants after multiple attempts"}

# Function to create the Streamlit user interface with a chat-like flow
def create_ui():
    """Create the chat-like interface UI"""
    # Set page configuration for the Streamlit app
    st.set_page_config(
        page_title="M😊😋D·Chef",
        page_icon="🍽️",
        layout="centered",  # Changed to centered for chat-like feel
        initial_sidebar_state="collapsed"
    )

    # Initialize session state variables for managing chat flow and user inputs
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'step' not in st.session_state:
        st.session_state.step = 0
    if 'mood' not in st.session_state:
        st.session_state.mood = None
    if 'location' not in st.session_state:
        st.session_state.location = None
    if 'allergies' not in st.session_state:
        st.session_state.allergies = None
    if 'num_suggestions' not in st.session_state:
        st.session_state.num_suggestions = None
        
    # Ensure all messages in session state have the required keys (type and content)
    for message in st.session_state.messages:
        if 'type' not in message:
            message['type'] = 'bot'
        if 'content' not in message:
            message['content'] = ''
    
    # Inject custom CSS for styling the chat interface, buttons, cards, etc.
    st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

/* Main App Background */
.main {
    background: #f0f2f5;
    color: #000000;
    font-family: 'Inter', sans-serif;
    max-width: 900px;
    margin: 0 auto;
    padding: 1rem;
}

/* make emoji render with normal color/emoji font */
.main-header .emoji {
  -webkit-text-fill-color: initial !important;
  background: none !important;
  background-clip: initial !important;
  font-family: "Apple Color Emoji", "Segoe UI Emoji", "Noto Color Emoji", sans-serif;
  /* optional spacing */
  margin: 0 0.1rem;
}

/* Chat Container */
.chat-container {
    display: flex;
    flex-direction: column;
    gap: 1rem;
    margin-bottom: 2rem;
    max-height: 500px;
    overflow-y: auto;
    padding-right: 8px;
}

/* Custom scrollbar for chat container */
.chat-container::-webkit-scrollbar {
    width: 6px;
}

.chat-container::-webkit-scrollbar-track {
    background: #f1f1f1;
    border-radius: 3px;
}

.chat-container::-webkit-scrollbar-thumb {
    background: #ff8c00;
    border-radius: 3px;
}

.chat-container::-webkit-scrollbar-thumb:hover {
    background: #e07b00;
}

/* Chat Messages */
.chat-message {
    display: flex;
    align-items: flex-start;
    margin: 0.5rem 0;
    animation: fadeInUp 0.3s ease-out;
}

@keyframes fadeInUp {
    from {
        opacity: 0;
        transform: translateY(20px);
    }
    to {
        opacity: 1;
        transform: translateY(0);
    }
}

.chat-message.bot {
    justify-content: flex-start;
}

.chat-message.user {
    justify-content: flex-end;
}

.message-content {
    max-width: 70%;
    padding: 1rem 1.5rem;
    border-radius: 18px;
    font-size: 1rem;
    line-height: 1.5;
    word-wrap: break-word;
}

.bot .message-content {
    background: #f8f9fa;
    border: 1px solid #e9ecef;
    margin-right: 30%;
    border-bottom-left-radius: 6px;
    position: relative;
}

.bot .message-content::before {
    content: '🤖';
    position: absolute;
    left: -35px;
    top: 0;
    font-size: 1.2rem;
}

.user .message-content {
    background: linear-gradient(135deg, #ff8c00, #ffae42);
    color: white;
    margin-left: 30%;
    border-bottom-right-radius: 6px;
    box-shadow: 0 2px 8px rgba(255, 140, 0, 0.3);
}

/* Chat Input Area */
.chat-input-area {
    border-top: 1px solid #e1e5e9;
    padding-top: 1.5rem;
    margin-top: 1.5rem;
}

/* Mood Options Grid */
.mood-options {
    display: grid;
    grid-template-columns: repeat(2, 1fr);
    gap: 0.8rem;
    margin: 1.5rem 0;
    padding: 0;
}

/* Mood Button */
.mood-button {
    background: linear-gradient(135deg, #ffffff, #f8f9fa);
    border: 2px solid #ff8c00;
    color: #ff8c00;
    padding: 1rem;
    border-radius: 15px;
    cursor: pointer;
    font-size: 1rem;
    font-weight: 500;
    transition: all 0.3s ease;
    text-align: center;
    width: 100%;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
}

.mood-button:hover {
    background: linear-gradient(135deg, #ff8c00, #ffae42);
    color: white;
    transform: translateY(-2px);
    box-shadow: 0 4px 12px rgba(255, 140, 0, 0.3);
}

/* Chat Input */
.chat-input {
    background: #f8f9fa;
    border: 2px solid #e9ecef;
    border-radius: 25px;
    padding: 1rem 1.5rem;
    margin-top: 1rem;
    width: 100%;
    font-size: 1rem;
    transition: all 0.2s ease;
}

.chat-input:focus {
    border-color: #ff8c00;
    outline: none;
    box-shadow: 0 0 0 3px rgba(255, 140, 0, 0.1);
}

/* Header */
.main-header {
    text-align: center;
    color: #000000;
    font-size: 3.5rem;
    font-weight: 700;
    margin: 1rem 0;
    background: linear-gradient(135deg, #ff8c00, #ffae42);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}

.subtitle {
    text-align: center;
    color: #6c757d;
    font-size: 1.2rem;
    margin-bottom: 2rem;
    font-weight: 400;
}

/* Status Indicator */
.chat-status {
    text-align: center;
    padding: 0.5rem;
    background: #e8f5e8;
    border: 1px solid #4caf50;
    border-radius: 10px;
    margin-bottom: 1rem;
    font-size: 0.9rem;
    color: #2e7d32;
}

/* Cards */
.card,
.recipe-card,
.restaurant-card {
    background: linear-gradient(135deg, #f5f5f5, #ebebeb);
    border: 1px solid #dddddd;
    border-radius: 12px;
    padding: 1.5rem;
    margin: 1rem 0;
    transition: border-color 0.2s, box-shadow 0.2s;
}

.recipe-card:hover,
.restaurant-card:hover {
    border-color: #cccccc;
    box-shadow: 0 4px 12px rgba(0,0,0,0.08);
}

/* Cuisine Tags */
.cuisine-tag {
    background: linear-gradient(135deg, #ff8c00, #ffae42);
    color: #ffffff;
    padding: 0.8rem 1.2rem;
    border-radius: 8px;
    text-align: center;
    font-weight: 500;
    border: 1px solid #e07b00;
    margin: 0.5rem 0;
}

/* Form Elements */
.stSelectbox > div > div,
.stTextInput > div > div > input {
    background: linear-gradient(135deg, #f5f5f5, #eaeaea);
    border: 1px solid #cccccc;
    border-radius: 8px;
    color: #000000;
}

.stTextInput > div > div > input::placeholder {
    color: #777777;
}

.stSlider > div > div > div {
    background: linear-gradient(90deg, #ff8c00, #ffae42);
    border-radius: 8px;
}

/* Button */
.stButton > button {
    background: linear-gradient(90deg, #ff8c00, #ffae42);
    color: white;
    border: none;
    border-radius: 12px;
    padding: 0.75rem 2rem;
    font-size: 1rem;
    font-weight: 500;
    width: 100%;
    transition: all 0.3s ease;
    box-shadow: 0 2px 8px rgba(255, 140, 0, 0.3);
}

.stButton > button:hover {
    background: linear-gradient(90deg, #e07b00, #ff9911);
    transform: translateY(-2px);
    box-shadow: 0 4px 16px rgba(255, 140, 0, 0.4);
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    background: linear-gradient(135deg, #f5f5f5, #ebebeb);
    border-radius: 8px;
    gap: 4px;
    padding: 4px;
}

.stTabs [data-baseweb="tab"] {
    background-color: transparent;
    color: #555555;
    border-radius: 6px;
    font-weight: 500;
    padding: 0.5rem 1rem;
}

.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #ff8c00, #ffae42) !important;
    color: white !important;
}

/* Sidebar */
.css-1d391kg {
    background: linear-gradient(135deg, #ffffff, #f2f2f2);
    border-right: 1px solid #dddddd;
}

/* Text Colors */
h1, h2, h3, h4, h5, h6 {
    color: #000000 !important;
}

p, div, span {
    color: #000000;
}

/* Metrics */
.metric-card {
    background: linear-gradient(135deg, #f5f5f5, #ebebeb);
    border-radius: 8px;
    padding: 1rem;
    text-align: center;
    border: 1px solid #cccccc;
}

/* Links */
a {
    color: #ff8c00;
    text-decoration: none;
}

a:hover {
    color: #e07b00;
    text-decoration: underline;
}

/* Messages with Gradients */
.stError {
    background: linear-gradient(135deg, #fff0f0, #ffe5e5);
    border: 1px solid #cc4444;
    border-radius: 8px;
}

.stWarning {
    background: linear-gradient(135deg, #fff8e1, #ffedc2);
    border: 1px solid #ffae42;
    border-radius: 8px;
}

.stSuccess {
    background: linear-gradient(135deg, #e6f4ea, #d0ebd7);
    border: 1px solid #28a745;
    border-radius: 8px;
}

.stInfo {
    background: linear-gradient(135deg, #e7f0fa, #d2e5f5);
    border: 1px solid #4488cc;
    border-radius: 8px;
}

/* Reset/New Chat Button */
.reset-button {
    background: linear-gradient(135deg, #6c757d, #5a6268);
    color: white;
    border: none;
    border-radius: 12px;
    padding: 0.75rem 2rem;
    font-size: 0.9rem;
    font-weight: 500;
    width: 100%;
    margin-top: 2rem;
    transition: all 0.3s ease;
}

.reset-button:hover {
    background: linear-gradient(135deg, #5a6268, #495057);
    transform: translateY(-1px);
}
</style>
    """, unsafe_allow_html=True)

    
    # Display the app header with emoji
    st.markdown('<h1 class="main-header">M<span class="emoji">😊😋</span>D·Chef</h1>', unsafe_allow_html=True)
    # Display subtitle
    st.markdown('<p class="subtitle">Your AI-powered mood-based food & entertainment companion</p>', unsafe_allow_html=True)
    
    # Sidebar section for settings and API status
    with st.sidebar:
        st.header("⚙️ Settings")
        # Validate API keys and display warning if any are missing
        api_status = Config.validate_keys()
        if not all(api_status.values()):
            st.warning("Some API keys are missing. Check your .env file.")
    
    # Conditionally display the chat box if there are messages or the chat has started
    if st.session_state.messages or st.session_state.step > 0:
        # Start the chat box container
        st.markdown('<div class="chat-box">', unsafe_allow_html=True)
        
        # Display chat status if in early steps
        if st.session_state.step < 4:
            st.markdown('<div class="chat-status">🟢 Chat Active</div>', unsafe_allow_html=True)
        
        # Display chat container if there are messages
        if st.session_state.messages:
            st.markdown('<div class="chat-container">', unsafe_allow_html=True)
            
            # Loop through and display each message in the chat history
            for message in st.session_state.messages:
                st.markdown(f"""
                    <div class="chat-message {message['type']}">
                        <div class="message-content">{message['content']}</div>
                    </div>
                """, unsafe_allow_html=True)
            
            st.markdown('</div>', unsafe_allow_html=True)  # End chat-container
        
        # Start the chat input area
        st.markdown('<div class="chat-input-area">', unsafe_allow_html=True)
    else:
        # Display a welcome message if no chat is active
        st.markdown("""
        <div style="text-align: center; padding: 2rem; margin: 2rem 0;">
            <p style="font-size: 1.1rem; color: #6c757d; margin-bottom: 1rem;">
                Ready to discover your perfect meal? Let's start by sharing how you're feeling! 😊
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    
    # Manage the chat flow based on the current step in session state
    if st.session_state.step == 0:
        # Step 0: Greet and ask for mood
        if len(st.session_state.messages) == 0:
            # Add initial bot message
            st.session_state.messages = [{
                "type": "bot",
                "content": "Hey there! 👋 How are you feeling today?"
            }]
            # Rerun to update the UI
            st.rerun()
        
        # Display mood options in a grid layout
        st.markdown('<div class="mood-options">', unsafe_allow_html=True)
        cols = st.columns(2)
        # List of available moods with emojis
        moods = ["😊 happy", "😢 sad", "😴 tired", "😰 stressed", 
                "😌 relaxed", "⚡ energetic", "💕 romantic", "🌟 adventurous"]
        
        # Create buttons for each mood
        for idx, mood in enumerate(moods):
            with cols[idx % 2]:
                if st.button(mood, key=f"mood_{idx}"):
                    # Save selected mood and update messages
                    st.session_state.mood = mood
                    st.session_state.messages.append({
                        "type": "user",
                        "content": mood
                    })
                    st.session_state.messages.append({
                        "type": "bot",
                        "content": "Where are you located? This helps me find restaurants near you! 📍"
                    })
                    # Advance to next step
                    st.session_state.step = 1
                    st.rerun()
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    elif st.session_state.step == 1:
        # Step 1: Ask for location
        location = st.text_input("Enter your location", key="location_input", 
                               placeholder="e.g., Lagos, Nigeria")
        if location:
            # Save location and update messages
            st.session_state.location = location
            st.session_state.messages.append({
                "type": "user",
                "content": location
            })
            st.session_state.messages.append({
                "type": "bot",
                "content": "Do you have any allergies or dietary restrictions I should know about? 🍽️"
            })
            # Advance to next step
            st.session_state.step = 2
            st.rerun()
    
    elif st.session_state.step == 2:
        # Step 2: Ask for allergies using a form for submit on enter
        with st.form(key="allergies_form", clear_on_submit=False):
            allergies = st.text_input("Enter any allergies", key="allergies_input",
                                    placeholder="e.g., nuts, dairy, gluten (or press Enter to skip)")
            submitted = st.form_submit_button("Submit")
            
        if submitted:
            # Save allergies (or empty string)
            st.session_state.allergies = allergies
            response_text = allergies if allergies else "No allergies"
            st.session_state.messages.append({
                "type": "user",
                "content": response_text
            })
            st.session_state.messages.append({
                "type": "bot",
                "content": "How many suggestions would you like? (Enter a number between 1-10) 🔢"
            })
            # Advance to next step
            st.session_state.step = 3
            st.rerun()
    
    elif st.session_state.step == 3:
        # Step 3: Ask for number of suggestions using a form
        with st.form(key="num_suggestions_form", clear_on_submit=False):
            num = st.number_input("Number of suggestions", min_value=1, max_value=10, value=5, key="num_input")
            submitted = st.form_submit_button("Submit")
            
        if submitted:
            # Save number and update messages
            st.session_state.num_suggestions = num
            st.session_state.messages.append({
                "type": "user",
                "content": f"{num} suggestions"
            })
            st.session_state.messages.append({
                "type": "bot",
                "content": "Perfect! Let me get your personalized recommendations ready... ✨"
            })
            # Advance to final step
            st.session_state.step = 4
            st.rerun()
    
    if st.session_state.step == 4:
        # Step 4: Generate and display recommendations
        get_recommendations(
            st.session_state.mood,
            st.session_state.allergies,
            st.session_state.location,
            st.session_state.num_suggestions
        )
        
        # Add a button to reset the chat
        if st.button("🔄 Start New Chat", key="reset_chat"):
            # Clear all session state keys to reset
            for key in st.session_state.keys():
                del st.session_state[key]
            st.rerun()

# Function to generate recommendations based on user inputs
def get_recommendations(mood: str, allergies: str, location: str, num_suggestions: int):
    """Generate and display recommendations"""
    # Create predictor instance and get suggested cuisines
    predictor = MoodCuisinePredictor()
    suggested_cuisines = predictor.predict_cuisines(mood, 2)
    
    # Display header for recommendations
    st.markdown(f"### Your {mood} mood recommendations:")
    
    # Define emojis for different cuisines
    cuisine_emojis = {
        "Nigerian rice dishes": "🍚", "Nigerian soup": "🍲", "Quick Nigerian meals": "⚡",
        "Nigerian salad": "🥗", "Nigerian barbecue": "🔥", "Spicy Nigerian": "🌶️",
        "Nigerian fine dining": "✨", "West African": "🌍", "Italian pasta": "🍝",
        "Mediterranean": "🫒", "Italian comfort food": "🍝", "Chinese noodles": "🥢",
        "American fast food": "🍔", "Asian noodles": "🍜", "Mediterranean healthy": "🥗",
        "Grilled fish": "🐟", "French cuisine": "🥐", "Japanese": "🍱",
        "Mexican spicy": "🌮", "Korean spicy": "🍜", "French romantic": "🥐",
        "Italian romantic": "🍝", "Ethiopian": "🫓", "Moroccan fusion": "🍯",
        "Rice dishes": "🍚", "Soup": "🍲", "Quick meals": "⚡",
        "Salad": "🥗", "Barbecue": "🔥", "Spicy food": "🌶️",
        "Fine dining": "✨", "African": "🌍"
    }
    
    # Display suggested cuisines in columns with emojis
    cols = st.columns(len(suggested_cuisines))
    for i, cuisine in enumerate(suggested_cuisines):
        with cols[i]:
            emoji = cuisine_emojis.get(cuisine, "🍽️")
            st.markdown(f"""
            <div class="cuisine-tag">
                {emoji} <strong>{cuisine}</strong>
            </div>
            """, unsafe_allow_html=True)
    
    # Create tabs for different recommendation types
    tab1, tab2, tab3 = st.tabs(["🍲 Recipes", "🎵 Music", "🏙️ Restaurants"])
    
    with tab1:
        # Display recipes in the first tab
        display_recipes(suggested_cuisines, allergies, num_suggestions)
    
    with tab2:
        # Display music in the second tab
        display_music(mood, num_suggestions)
    
    with tab3:
        # Display restaurants in the third tab if location is provided
        if location:
            display_restaurants(suggested_cuisines, location, num_suggestions)
        else:
            st.info("Please enter your location to see restaurant recommendations.")

# Function to display recipes in the UI
def display_recipes(cuisines: List[str], allergies: str, limit: int):
    """Display recipes with clean styling"""
    # Loop through each suggested cuisine
    for cuisine in cuisines:
        st.markdown(f"## 🍽️ {cuisine} Recipes")
        # Fetch recipes for the cuisine
        recipes_data = APIClient.fetch_recipes(cuisine, allergies, limit//len(cuisines))
        
        # Handle errors
        if "error" in recipes_data:
            st.error(recipes_data["error"])
            continue
        
        # Handle no recipes found
        if not recipes_data.get("recipes"):
            st.info(f"No recipes found for {cuisine}. Try a different cuisine or check your API key.")
            continue
        
        # Display each recipe in a card
        for recipe in recipes_data["recipes"]:
            st.markdown(f"""
            <div class="recipe-card">
            """, unsafe_allow_html=True)
            
            # Create columns for image and details
            col1, col2 = st.columns([1, 2])
            
            with col1:
                # Display image if available
                if recipe["image"]:
                    st.image(recipe["image"], width=200)
            
            with col2:
                # Display recipe title
                st.markdown(f"### {recipe['title']}")
                
                # Display metrics in columns
                metrics_cols = st.columns(3)
                with metrics_cols[0]:
                    st.markdown(f"""
                    <div class="metric-card">
                        <strong>⏱️ {recipe['ready_in_minutes']} min</strong>
                    </div>
                    """, unsafe_allow_html=True)
                with metrics_cols[1]:
                    st.markdown(f"""
                    <div class="metric-card">
                        <strong>👥 {recipe['servings']} servings</strong>
                    </div>
                    """, unsafe_allow_html=True)
                with metrics_cols[2]:
                    if recipe['source_url']:
                        st.markdown(f"""
                        <div class="metric-card">
                            <a href="{recipe['source_url']}" target="_blank"><strong>🔗 Full Recipe</strong></a>
                        </div>
                        """, unsafe_allow_html=True)
            
            # Display ingredients list
            if recipe['ingredients']:
                st.markdown("**📋 Ingredients:**")
                for ingredient in recipe['ingredients'][:6]:
                    st.write(f"• {ingredient}")
                # Expander for more ingredients if many
                if len(recipe['ingredients']) > 6:
                    with st.expander(f"Show {len(recipe['ingredients']) - 6} more ingredients"):
                        for ingredient in recipe['ingredients'][6:]:
                            st.write(f"• {ingredient}")
            
            # Display instructions list
            if recipe['instructions']:
                st.markdown("**👩‍🍳 Instructions:**")
                for instruction in recipe['instructions'][:3]:
                    st.write(f"**Step {instruction['number']}:** {instruction['step']}")
                # Expander for more steps if many
                if len(recipe['instructions']) > 3:
                    with st.expander(f"Show {len(recipe['instructions']) - 3} more steps"):
                        for instruction in recipe['instructions'][3:]:
                            st.write(f"**Step {instruction['number']}:** {instruction['step']}")
            
            # End the recipe card
            st.markdown("</div>", unsafe_allow_html=True)

# Function to display music recommendations in the UI
def display_music(mood: str, limit: int):
    """Display music recommendations"""
    # Clean the mood for query (remove emoji)
    clean_mood = mood.split()[-1] if " " in mood else mood
    # Fetch music data
    music_data = APIClient.fetch_music(clean_mood, limit)
    
    # Handle errors
    if "error" in music_data:
        st.error(music_data["error"])
        return
    
    # Display each music item in a card
    for item in music_data.get("music", []):
        st.markdown(f"""
        <div class="card">
        """, unsafe_allow_html=True)
        
        if item["type"] == "playlist":
            # Display playlist details
            st.markdown(f"""
            ### 🎵 [{item['name']}]({item['url']})
            **Playlist** • {item.get('description', 'No description')}  
            Created by {item['owner']}
            """)
        else:
            # Display track details
            preview_text = "🎧 Preview available" if item.get("preview_url") else ""
            genre_text = f"**{item.get('genre', '')}**" if item.get('genre') else ""
            st.markdown(f"""
            ### 🎵 [{item['name']}]({item['url']})
            **{item['artist']}** • {genre_text}  
            {preview_text}
            """)
        
        # End the card
        st.markdown("</div>", unsafe_allow_html=True)

# Function to display restaurant recommendations in the UI
def display_restaurants(cuisines: List[str], location: str, limit: int):
    """Display restaurant recommendations"""
    # Loop through each cuisine
    for cuisine in cuisines:
        st.markdown(f"## 🏪 {cuisine} Restaurants")
        # Fetch restaurants for the cuisine and location
        restaurants_data = APIClient.fetch_restaurants(cuisine, location, limit//len(cuisines))
        
        # Handle errors
        if "error" in restaurants_data:
            st.error(f"Error: {restaurants_data['error']}")
            continue
        
        # Get the list of restaurants
        restaurants = restaurants_data.get("restaurants", [])
        # Handle no restaurants found
        if not restaurants:
            st.info(f"No {cuisine} restaurants found in {location}.")
            continue
        
        # Display each restaurant in a card
        for restaurant in restaurants:
            st.markdown(f"""
            <div class="restaurant-card">
            """, unsafe_allow_html=True)
            
            # Create columns for details and metrics
            col1, col2 = st.columns([2, 1])
            
            with col1:
                # Display name and address
                st.markdown(f"### {restaurant['name']}")
                st.write(f"📍 {restaurant['address']}")
                
                # Display categories if available
                if restaurant['categories']:
                    categories = " • ".join(restaurant['categories'][:3])
                    st.write(f"🏷️ {categories}")
                
                # Collect contact info
                contact_info = []
                if restaurant.get('phone'):
                    contact_info.append(f"📞 {restaurant['phone']}")
                if restaurant.get('website'):
                    contact_info.append(f"[🌐 Website]({restaurant['website']})")
                
                # Display contact info if any
                if contact_info:
                    st.markdown(" • ".join(contact_info))
            
            with col2:
                # Display distance if available
                if restaurant.get('distance'):
                    distance_km = restaurant['distance'] / 1000
                    if distance_km < 1:
                        st.markdown(f"""
                        <div class="metric-card">
                            <strong>📏 {restaurant['distance']}m</strong>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class="metric-card">
                            <strong>📏 {distance_km:.1f}km</strong>
                        </div>
                        """, unsafe_allow_html=True)
            
            # End the restaurant card
            st.markdown("</div>", unsafe_allow_html=True)

# Main entry point: Run the UI creation function if script is executed directly
if __name__ == "__main__":
    create_ui()