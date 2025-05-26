import streamlit as st
from datetime import datetime, time
import pandas as pd
pip install joblib
import joblib
from utils.data_processing import prepare_data, enhance_features
from googleapiclient.discovery import build
from urllib.parse import urlparse, parse_qs

# MUST be the first Streamlit command
st.set_page_config(
    page_title="YouTube Trending Predictor",
    page_icon="▶️",
    layout="wide"
)

# YouTube API Setup
API_KEY = "AIzaSyDdwP2h9VY2c5TxtOpYGz1X9FabDlGdLuw"
youtube = build('youtube', 'v3', developerKey=API_KEY)
# Change the load_model() function to this:

@st.cache_resource
def load_model():
    try:
        # Use your specific dated filenames
        model = joblib.load('models/youtube_trending_model_20250516.joblib')
        features = joblib.load('models/youtube_features_20250516.joblib')
        return model, features
    except FileNotFoundError as e:
        st.error(f"Model loading error: {e}")
        # Try default filenames as fallback
        try:
            model = joblib.load('models/youtube_trending_model.joblib')
            features = joblib.load('models/youtube_features.joblib')
            return model, features
        except FileNotFoundError:
            st.error("Could not find model files. Please ensure:")
            st.error("1. The 'models' directory exists")
            st.error("2. Files are named either:")
            st.error("   - youtube_trending_model_20250516.joblib")
            st.error("   - youtube_features_20250516.joblib")
            st.error("   OR the non-dated versions")
            return None, None

# Load model and features
model, features = load_model()

@st.cache_data(ttl=3600)
def get_video_details(video_url_or_id):
    """Fetch video details from YouTube API"""
    try:
        if 'youtube.com' in video_url_or_id or 'youtu.be' in video_url_or_id:
            if 'youtu.be' in video_url_or_id:
                video_id = video_url_or_id.split('/')[-1]
            else:
                url_data = urlparse(video_url_or_id)
                query = parse_qs(url_data.query)
                video_id = query['v'][0]
        else:
            video_id = video_url_or_id
        
        request = youtube.videos().list(
            part="snippet,statistics,contentDetails",
            id=video_id
        )
        response = request.execute()
        
        if not response['items']:
            return None
            
        item = response['items'][0]
        return {
            'title': item['snippet']['title'],
            'channel': item['snippet']['channelTitle'],
            'views': int(item['statistics'].get('viewCount', 0)),
            'likes': int(item['statistics'].get('likeCount', 0)),
            'comments': int(item['statistics'].get('commentCount', 0)),
            'published_at': item['snippet']['publishedAt']
        }
    except Exception as e:
        st.error(f"Error fetching video details: {e}")
        return None

# Main App
st.title("YouTube Trending Probability Predictor")

with st.sidebar:
    st.title("Video Details")
    video_input = st.text_input("Enter YouTube URL or Video ID", "")
    
    if video_input:
        with st.spinner("Fetching video details..."):
            video_details = get_video_details(video_input)
            
        if video_details:
            title = st.text_input("Video Title", video_details['title'])
            channel = st.text_input("Channel Name", video_details['channel'])
            views = st.number_input("Views", min_value=0, value=video_details['views'])
            likes = st.number_input("Likes", min_value=0, value=video_details['likes'])
            comments = st.number_input("Comments", min_value=0, value=video_details['comments'])
            
            publish_datetime = datetime.strptime(
                video_details['published_at'], 
                "%Y-%m-%dT%H:%M:%SZ"
            )
            st.markdown(f"**Published Date:** {publish_datetime.date()}")
            st.markdown(f"**Published Time:** {publish_datetime.time()}")
        else:
            st.warning("Could not fetch video details. Please enter manually.")
            title = st.text_input("Video Title", "")
            channel = st.text_input("Channel Name", "")
            views = st.number_input("Views", min_value=0, value=100000)
            likes = st.number_input("Likes", min_value=0, value=5000)
            comments = st.number_input("Comments", min_value=0, value=1000)
            publish_datetime = datetime.now()
    else:
        title = st.text_input("Video Title", "")
        channel = st.text_input("Channel Name", "")
        views = st.number_input("Views", min_value=0, value=100000)
        likes = st.number_input("Likes", min_value=0, value=5000)
        comments = st.number_input("Comments", min_value=0, value=1000)
        publish_datetime = datetime.now()
    
    region = st.selectbox("Region", ["US", "IN", "GB", "JP", "CA"])
    
    video_data = {
        'Title': title,
        'Channel': channel,
        'Views': views,
        'Likes': likes,
        'Comments': comments,
        'Published At': publish_datetime,
        'Region': region
    }

# Prediction Section
if model and features:
    if st.button("Predict Trending Probability"):
        input_df = pd.DataFrame([video_data])
        processed_df = prepare_data(input_df)
        enhanced_df = enhance_features(processed_df)
        
        for feature in features:
            if feature not in enhanced_df.columns:
                enhanced_df[feature] = 0
        
        probability = model.predict_proba(enhanced_df[features])[0][1]
        
        st.subheader("Prediction Results")
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Trending Probability", f"{probability:.1%}")
            if probability > 0.7:
                st.success("🔥 Very likely to trend!")
            elif probability > 0.4:
                st.warning("🤔 Might trend")
            else:
                st.error("❌ Unlikely to trend")
        
        with col2:
            st.write("**Key Factors:**")
            st.write(f"- Title: {video_data['Title'][:30]}...")
            st.write(f"- Channel: {video_data['Channel']}")
            st.write(f"- Engagement: {video_data['Likes']/video_data['Views']:.2%} like/view ratio")
else:
    st.error("""
    Model not loaded. Please ensure:
    1. You've trained the model (run training notebook)
    2. Created a 'models' directory with:
       - youtube_trending_model.joblib
       - youtube_features.joblib
    """)

st.markdown("---")
st.subheader("How to Improve")
st.write("""
1. Increase video engagement (likes/comments)
2. Publish during peak hours (2-4 PM local time)
3. Use compelling titles with proper keywords
""")
