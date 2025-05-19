import pandas as pd
from datetime import datetime
from textblob import TextBlob
import re

def prepare_data(df):
    """
    Prepare raw YouTube data for modeling
    Args:
        df: Raw DataFrame from YouTube API
    Returns:
        Processed DataFrame with basic features
    """
    # Convert to datetime - remove timezone if present
    df['Published At'] = pd.to_datetime(df['Published At']).dt.tz_localize(None)
    
    # Basic features
    df['Upload_Hour'] = df['Published At'].dt.hour
    df['Upload_Weekday'] = df['Published At'].dt.weekday  # Monday=0
    df['Title_Length'] = df['Title'].apply(len)
    
    # Engagement ratios
    df['Like_View_Ratio'] = df['Likes'] / df['Views']
    df['Comment_View_Ratio'] = df['Comments'] / df['Views']
    df['Like_Comment_Ratio'] = df['Likes'] / (df['Comments'] + 1)  # Avoid division by zero
    
    # Target variable (top 20% by views)
    view_threshold = df['Views'].quantile(0.80)
    df['Is_Trending'] = (df['Views'] >= view_threshold).astype(int)
    
    return df

def enhance_features(df):
    """
    Create advanced predictive features
    Args:
        df: Preprocessed DataFrame from prepare_data()
    Returns:
        DataFrame with enhanced features
    """
    # Title analysis
    df['Title_Has_Emoji'] = df['Title'].apply(lambda x: bool(re.search(r'[^\w\s]', str(x)))).astype(int)
    df['Title_Sentiment_Score'] = df['Title'].apply(lambda x: TextBlob(str(x)).sentiment.polarity)
    df['Title_Word_Count'] = df['Title'].apply(lambda x: len(str(x).split()))
    
    # Corrected line - added missing parenthesis
    df['Title_Caps_Ratio'] = df['Title'].apply(
        lambda x: sum(1 for c in str(x) if c.isupper()) / max(1, len(str(x)))
    )
    
    df['Has_Official'] = df['Title'].str.contains('official', case=False).astype(int)
    
    # Channel popularity
    df['Channel_Frequency'] = df.groupby('Channel')['Channel'].transform('count')
    
    # Temporal features
    now = datetime.now()
    df['Hours_Since_Upload'] = (now - df['Published At']).dt.total_seconds() / 3600
    
    # Combined engagement metrics
    df['Engagement_Score'] = (df['Likes'] * 0.6 + df['Comments'] * 0.4) / df['Views']
    
    return df