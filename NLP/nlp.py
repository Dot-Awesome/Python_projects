import os
import pandas as pd
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import google.generativeai as genai
from dotenv import load_dotenv

# Download necessary NLTK data
nltk.download('vader_lexicon')

class CustomerFeedbackAnalyzer:
    def __init__(self, gemini_api_key=None):
        """Initialize sentiment analyzer and Gemini model"""
        # Set up sentiment analyzer
        self.sentiment_analyzer = SentimentIntensityAnalyzer()
        
        # Set up Gemini API
        if gemini_api_key:
            genai.configure(api_key=gemini_api_key)
        else:
            # Try to load API key from environment
            load_dotenv()
            api_key = os.getenv("GEMINI_API_KEY")
            if api_key:
                genai.configure(api_key=api_key)
            else:
                raise ValueError("No Gemini API key provided. Set GEMINI_API_KEY environment variable or pass it as a parameter.")
        
        # Configure Gemini model
        self.model = genai.GenerativeModel('gemini-2.0-flash')
    
    def analyze_sentiment(self, text):
        """Analyze sentiment using NLTK's VADER"""
        scores = self.sentiment_analyzer.polarity_scores(text)
        compound = scores['compound']
        
        if compound >= 0.05:
            sentiment = 'positive'
        elif compound <= -0.05:
            sentiment = 'negative'
        else:
            sentiment = 'neutral'
            
        return {
            'sentiment': sentiment,
            'score': compound,
            'details': scores
        }
    
    def generate_summary(self, text):
        """Generate a summary of likes and dislikes using Gemini"""
        prompt = f"""
        Please analyze the following customer feedback and generate a concise summary that highlights:
        1. What the customer liked (positive points)
        2. What the customer disliked or had concerns about (negative points)
        
        Feedback: {text}
        
        Format your response as:
        {{
            "liked": ["point 1", "point 2", ...],
            "disliked": ["point 1", "point 2", ...]
        }}
        """
        
        response = self.model.generate_content(prompt)
        return response.text
    
    def analyze_feedback(self, text):
        """Complete feedback analysis with sentiment and summary generation"""
        # Get sentiment
        sentiment_result = self.analyze_sentiment(text)
        
        # Generate summary
        summary = self.generate_summary(text)
        
        return {
            'sentiment_analysis': sentiment_result,
            'summary': summary
        }

# Example usage
def main():
    analyzer = CustomerFeedbackAnalyzer()
    
    # Get customer feedback from user input
    feedback = input("Please enter the customer feedback: ")
    
    # Analyze the feedback
    result = analyzer.analyze_feedback(feedback)
    
    # Print results
    print(f"\nSentiment: {result['sentiment_analysis']['sentiment']} (Score: {result['sentiment_analysis']['score']:.2f})")
    print("\nSummary of Customer Feedback:")
    print(result['summary'])

if __name__ == "__main__":
    main()