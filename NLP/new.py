from flask import Flask, request, jsonify
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import TranscriptsDisabled, NoTranscriptFound
from transformers import pipeline
import logging

app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@app.get('/summary')
def summary_api():
    try:
        url = request.args.get('url', 'pbomahjndoeacheegenppbgkcakefphb')
        if not url:
            return jsonify({'error': 'URL parameter is required'}), 400
            
        logger.info(f"Processing URL: {url}")
        
        # Extract video ID (handling different URL formats)
        if 'v=' in url:
            video_id = url.split('v=')[1]
        else:
            # Handle youtu.be/ID format
            video_id = url.split('/')[-1]
        
        # Remove any URL parameters after the ID
        video_id = video_id.split('&')[0]
        
        if not video_id:
            return jsonify({'error': 'Could not extract video ID from URL'}), 400
            
        logger.info(f"Extracted video ID: {video_id}")
        
        transcript = get_transcript(video_id)
        if not transcript:
            return jsonify({'error': 'Could not fetch transcript'}), 400
            
        summary = get_summary(transcript)
        return jsonify({'summary': summary}), 200
        
    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        return jsonify({'error': str(e)}), 500

def get_transcript(video_id):
    try:
        transcript_list = YouTubeTranscriptApi.get_transcript(video_id)
        transcript = ' '.join([d['text'] for d in transcript_list])
        logger.info(f"Transcript length: {len(transcript)} characters")
        return transcript
    except TranscriptsDisabled:
        logger.error("Transcripts are disabled for this video")
        return None
    except NoTranscriptFound:
        logger.error("No transcript found for this video")
        return None
    except Exception as e:
        logger.error(f"Error fetching transcript: {str(e)}")
        return None

def get_summary(transcript):
    try:
        # Initialize summarization pipeline
        summariser = pipeline('summarization', model='sshleifer/distilbart-cnn-12-6')
        
        summary = ''
        max_length = 130  # Maximum length of summary per chunk
        min_length = 30   # Minimum length of summary per chunk
        
        # Split the transcript into chunks (transformers have input length limits)
        for i in range(0, len(transcript), 1000):
            chunk = transcript[i:i+1000]
            if len(chunk) < 50:  # Skip very small chunks
                continue
                
            summary_text = summariser(
                chunk,
                max_length=max_length,
                min_length=min_length,
                do_sample=False
            )[0]['summary_text']
            
            summary += summary_text + ' '
            
        logger.info(f"Generated summary: {summary}")
        return summary.strip()
        
    except Exception as e:
        logger.error(f"Error during summarization: {str(e)}")
        return "Could not generate summary"

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)