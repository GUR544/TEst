import os
import subprocess
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont
import json
import whisper
import requests
import argparse
import time
import textwrap
from typing import List, Dict, Any
import google.generativeai as genai
from collections import deque

# Free LLM API options - we'll use Google Gemini API with a rate limit for free tier
# Alternative options include HuggingFace Inference API or other free tier services
# You can swap this implementation for any other free LLM API
class LLMClipFinder:
    """Class to handle LLM API calls for identifying interesting clips"""

    def __init__(self, api_key=None, model="gemini-1.5-flash"):
        """Initialize with optional API key (for Google Gemini)"""
        self.api_key = api_key or os.getenv("AIzaSyCro919wAmveDqoPCvjxLv1ULdrtcSzEPU")
        self.model_name = model

        if not self.api_key:
            print("No Google Gemini API key found. Falling back to alternate method.")
            self.use_gemini = False
            return
            
        # Configure the Gemini API client
        try:
            genai.configure(api_key=self.api_key)
            self.model = genai.GenerativeModel(self.model_name)
            self.use_gemini = True
        except Exception as e:
            print(f"Failed to initialize Gemini API: {e}")
            self.use_gemini = False

    def find_interesting_moments(self, transcription_segments, min_clips=3, max_clips=10):
        """Use LLM to identify interesting moments from transcription segments"""
        
        # Format the transcription data for the LLM
        transcript_text = ""
        for i, segment in enumerate(transcription_segments):
            start_time = self._format_time(segment["start"])
            end_time = self._format_time(segment["end"])
            transcript_text += f"[{start_time} - {end_time}] {segment['text']}\n"
        
        # Create prompt for the LLM
        prompt = f"""
You are an expert video editor who finds the most compelling moments in videos.

Here's a transcript with timestamps:

{transcript_text}

Please identify {min_clips}-{max_clips} moments that would make great short clips (45-60 seconds each). Focus on:
1. Interesting statements or stories
2. Emotional moments
3. Surprising revelations or insights
4. Quotable or memorable segments
5. Self-contained moments that work well in isolation

Format your response as JSON with this structure:
{{
  "clips": [
    {{
      "start": "mm:ss",
      "end": "mm:ss",
      "reason": "brief explanation",
      "caption": "suggested caption"
    }},
    ...
  ]
}}
"""

        if self.use_gemini:
            return self._call_gemini_api(prompt)
        else:
            return self._fallback_extraction(transcription_segments)
            
    def _call_gemini_api(self, prompt):
        """Call Gemini API with proper error handling"""
        try:
            response = self.model.generate_content(prompt)
            content = response.text
            
            # Try to parse the JSON from the response
            try:
                # Find JSON in the response if it's not pure JSON
                import re
                json_match = re.search(r'\{[\s\S]*\}', content)
                if json_match:
                    content = json_match.group(0)
                
                import json
                clip_data = json.loads(content)
                return clip_data
            except json.JSONDecodeError:
                print("Failed to parse JSON from LLM response. Using manual extraction.")
                return self._manually_extract_clips(content)
                
        except Exception as e:
            print(f"Error calling Gemini API: {str(e)}")
            return self._fallback_extraction(self.transcription_segments)
    def _manually_extract_clips(self, content):
        """Manually extract clip information if JSON parsing fails"""
        clips = []
        
        # Try to find and extract clip information using regex
        import re
        
        # Look for patterns like "Start: 01:23" or "Start time: 01:23"
        start_times = re.findall(r'start(?:\s+time)?:\s*(\d+:\d+)', content, re.IGNORECASE)
        end_times = re.findall(r'end(?:\s+time)?:\s*(\d+:\d+)', content, re.IGNORECASE)
        
        # Extract everything between "Reason:" and the next section as the reason
        reasons = re.findall(r'reason:\s*(.*?)(?=\n\s*(?:caption|start|end|clip|\d+\.)|\Z)', 
                             content, re.IGNORECASE | re.DOTALL)
        
        # Extract captions
        captions = re.findall(r'caption:\s*(.*?)(?=\n\s*(?:reason|start|end|clip|\d+\.)|\Z)', 
                              content, re.IGNORECASE | re.DOTALL)
        
        # Match up the extracted information
        for i in range(min(len(start_times), len(end_times))):
            clip = {
                "start": start_times[i],
                "end": end_times[i],
                "reason": reasons[i].strip() if i < len(reasons) else "Interesting moment",
                "caption": captions[i].strip() if i < len(captions) else "Check out this moment!"
            }
            clips.append(clip)
        
        return {"clips": clips}
    
    def _fallback_extraction(self, transcription_segments):
        """Simple fallback method if all API calls fail"""
        clips = []
        
        # Group segments into potential clips (simple approach)
        # This is a very basic fallback that just picks evenly spaced segments
        total_segments = len(transcription_segments)
        num_clips = min(5, total_segments // 3)  # Create up to 5 clips
        
        if num_clips == 0 and total_segments > 0:
            num_clips = 1
        
        for i in range(num_clips):
            idx = (i * total_segments) // num_clips
            segment = transcription_segments[idx]
            
            # Calculate clip start/end (aim for 45-60 second clips)
            clip_mid = (segment["start"] + segment["end"]) / 2
            clip_start = max(0, clip_mid - 25)
            clip_end = min(clip_mid + 25, segment["end"] + 30)
            
            clip = {
                "start": self._format_time(clip_start),
                "end": self._format_time(clip_end),
                "reason": "Potentially interesting segment",
                "caption": segment["text"][:100] + "..." if len(segment["text"]) > 100 else segment["text"]
            }
            clips.append(clip)
        
        return {"clips": clips}
    
    def _format_time(self, seconds):
        """Format seconds to mm:ss format"""
        minutes = int(seconds // 60)
        seconds = int(seconds % 60)
        return f"{minutes:02d}:{seconds:02d}"


def extract_audio(video_path, output_path="temp_audio.wav"):
    """Extract audio from video file"""
    command = f"ffmpeg -i {video_path} -ab 160k -ac 2 -ar 44100 -vn {output_path} -y"
    subprocess.call(command, shell=True)
    return output_path


def transcribe_audio(audio_path, whisper_model_size="base"):
    """Transcribe audio using Whisper and return segments"""
    print("Loading Whisper model...")
    model = whisper.load_model(whisper_model_size)
    
    print(f"Transcribing audio file: {audio_path}")
    result = model.transcribe(audio_path, word_timestamps=True)
    
    # Extract segments from the result
    segments = []
    for segment in result["segments"]:
        segments.append({
            "start": segment["start"],
            "end": segment["end"],
            "text": segment["text"],
            "words": segment.get("words", [])
        })
    
    # Process segments to create text lines for captions, similar to instaClips.py
    for segment in segments:
        clip_words = []
        for word in segment.get("words", []):
            clip_words.append({
                "text": word["word"],
                "start": word["start"],
                "end": word["end"]
            })
        
        # Create text lines for captions
        width = 1080  # Instagram width
        sample_text = "Sample Text for Calculation"
        font_size = 60  # Default font size

        font_path = "C:/Windows/Fonts/ARLRDBD.TTF"  # Adjust font path as needed
        font = font_path if os.path.exists(font_path) else "arial.ttf"
        
        # Create a temporary image to measure text size
        try:
            font = ImageFont.truetype(font_path, font_size)
        except:
            font = ImageFont.load_default()
            
        temp_img = Image.new('RGB', (1, 1))
        draw = ImageDraw.Draw(temp_img)
        
        # Calculate average char width
        bbox = draw.textbbox((0, 0), sample_text, font=font)
        sample_width = bbox[2] - bbox[0]
        char_width = sample_width / len(sample_text)
        
        # Calculate usable width (80% of screen width)
        usable_width = int(width * 0.8)
        chars_per_line = int(usable_width / char_width)
        
        # Create text lines based on words
        text_lines = []
        current_line = ""
        line_start_time = None
        
        for word in clip_words:
            word_text = word["text"].strip()
            if not word_text:
                continue
                
            # Start a new line if needed
            if line_start_time is None:
                line_start_time = word["start"]
            
            # Check if adding this word would exceed line length
            test_line = current_line + " " + word_text if current_line else word_text
            if len(test_line) > chars_per_line:
                # Add current line to text_lines
                if current_line:
                    text_lines.append({
                        "text": current_line,
                        "start": line_start_time,
                        "end": word["start"]
                    })
                
                # Start new line with current word
                current_line = word_text
                line_start_time = word["start"]
            else:
                # Add word to current line
                current_line = test_line
        
        # Add the last line if there is one
        if current_line:
            text_lines.append({
                "text": current_line,
                "start": line_start_time,
                "end": clip_words[-1]["end"] if clip_words else segment["end"]
            })
            
        # Add text_lines to segment
        segment["text_lines"] = text_lines
    
    return segments


def parse_timestamp(timestamp):
    """Convert 'mm:ss' timestamp to seconds"""
    parts = timestamp.split(":")
    if len(parts) == 2:
        minutes, seconds = parts
        return int(minutes) * 60 + int(seconds)
    elif len(parts) == 3:
        hours, minutes, seconds = parts
        return int(hours) * 3600 + int(minutes) * 60 + int(seconds)
    else:
        raise ValueError(f"Invalid timestamp format: {timestamp}")


def review_clips(clips, transcription_segments):
    """Allow user to review and edit clips before creating them"""
    approved_clips = []
    
    print("\n=== Clips to Review ===")
    for i, clip in enumerate(clips):
        print(f"\nClip {i+1}:")
        
        while True:  # Continue until user decides to approve or skip
            # Display current clip info
            print(f"  Time: {clip['start']} to {clip['end']}")
            print(f"  Reason: {clip['reason']}")
            print(f"  Caption: {clip['caption']}")
            
            # Display transcript for this time period
            start_time = parse_timestamp(clip["start"])
            end_time = parse_timestamp(clip["end"])
            
            print("\n  Transcript:")
            relevant_segments = []
            for j, segment in enumerate(transcription_segments):
                if segment["end"] >= start_time and segment["start"] <= end_time:
                    relevant_segments.append((j, segment))
            
            # Display segments with index for reference
            for seg_idx, (trans_idx, segment) in enumerate(relevant_segments):
                print(f"    [{seg_idx}] {segment['text']}")
            
            # Ask for user action
            action = input("\nActions: [a]pprove, [e]dit transcript, [t]rim times, [s]kip, [n]ext clip: ").lower()
            
            if action == 'a':
                # Add a second buffer to both start and end time
                clip = add_time_buffer(clip, buffer_seconds=1)
                approved_clips.append(clip)
                print("Clip approved!")
                break
                
            elif action == 'e':
                # Edit transcript
                if relevant_segments:
                    seg_to_edit = input("Enter segment number to edit [0, 1, ...] or 'all' for all segments: ")
                    
                    if seg_to_edit.lower() == 'all':
                        # Edit all segments
                        for seg_idx, (trans_idx, segment) in enumerate(relevant_segments):
                            current_text = segment['text']
                            print(f"\nEditing segment [{seg_idx}]: {current_text}")
                            new_text = input("Enter corrected text (leave empty to keep unchanged): ")
                            
                            if new_text:
                                # Update in transcription_segments
                                transcription_segments[trans_idx]['text'] = new_text
                                print(f"Updated segment [{seg_idx}]")
                                
                                # Update text_lines as well if they exist
                                if 'text_lines' in transcription_segments[trans_idx]:
                                    # Create a better word-level representation
                                    words = new_text.split()
                                    seg_duration = transcription_segments[trans_idx]["end"] - transcription_segments[trans_idx]["start"]
                                    word_duration = seg_duration / len(words) if words else 0
                                    
                                    new_words = []
                                    for i, word in enumerate(words):
                                        word_start = transcription_segments[trans_idx]["start"] + (i * word_duration)
                                        word_end = word_start + word_duration
                                        new_words.append({
                                            "word": word,
                                            "start": word_start,
                                            "end": word_end
                                        })
                                    
                                    # Update the words in the segment
                                    transcription_segments[trans_idx]['words'] = new_words
                                    
                                    # Update text_lines with evenly distributed words
                                    lines = textwrap.wrap(new_text, width=40)  # Basic line breaking
                                    line_count = len(lines)
                                    line_duration = seg_duration / line_count if line_count else 0
                                    
                                    new_text_lines = []
                                    for i, line in enumerate(lines):
                                        line_start = transcription_segments[trans_idx]["start"] + (i * line_duration)
                                        line_end = line_start + line_duration
                                        new_text_lines.append({
                                            "text": line,
                                            "start": line_start,
                                            "end": line_end
                                        })
    
                                        transcription_segments[trans_idx]['text_lines'] = new_text_lines
                    else:
                        try:
                            seg_idx = int(seg_to_edit)
                            if 0 <= seg_idx < len(relevant_segments):
                                trans_idx, segment = relevant_segments[seg_idx]
                                current_text = segment['text']
                                print(f"Current text: {current_text}")
                                new_text = input("Enter corrected text: ")
                                
                                if new_text:
                                    # Update in transcription_segments
                                    transcription_segments[trans_idx]['text'] = new_text
                                    print(f"Updated segment [{seg_idx}]")
                                    
                                    # Update text_lines as well if they exist
                                    if 'text_lines' in transcription_segments[trans_idx]:
                                        # Create a simple one-line text_line
                                        transcription_segments[trans_idx]['text_lines'] = [{
                                            "text": new_text,
                                            "start": transcription_segments[trans_idx]["start"],
                                            "end": transcription_segments[trans_idx]["end"]
                                        }]
                            else:
                                print("Invalid segment number.")
                        except ValueError:
                            print("Please enter a valid segment number or 'all'.")
                else:
                    print("No transcript segments available for this clip.")
                    
            elif action == 't':
                new_start = input(f"New start time (current: {clip['start']}, format mm:ss): ")
                if new_start:
                    clip["start"] = new_start
                
                new_end = input(f"New end time (current: {clip['end']}, format mm:ss): ")
                if new_end:
                    clip["end"] = new_end
                
                print(f"Clip timing updated: {clip['start']} to {clip['end']}")
                
            elif action == 's':
                print("Clip skipped.")
                break
                
            elif action == 'n':
                # Add a second buffer to both start and end time
                clip = add_time_buffer(clip, buffer_seconds=1)
                approved_clips.append(clip)
                print("Moving to next clip...")
                break
                
            else:
                print("Invalid action. Please try again.")
    
    return approved_clips, transcription_segments


def add_time_buffer(clip, buffer_seconds=1):
    """Add buffer time to clip start and end times"""
    # Parse current times
    start_time = parse_timestamp(clip["start"])
    end_time = parse_timestamp(clip["end"])
    
    # Add buffer (subtract from start, add to end)
    new_start_time = max(0, start_time - buffer_seconds)
    new_end_time = end_time + buffer_seconds
    
    # Format back to mm:ss
    clip["start"] = format_time(new_start_time)
    clip["end"] = format_time(new_end_time)
    
    return clip


def format_time(seconds):
    """Format seconds to mm:ss format"""
    minutes = int(seconds // 60)
    seconds = int(seconds % 60)
    return f"{minutes:02d}:{seconds:02d}"


def cv2_to_pil(cv2_img):
    """Convert CV2 image (BGR) to PIL image (RGB)"""
    return Image.fromarray(cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB))


def pil_to_cv2(pil_img):
    """Convert PIL image (RGB) to CV2 image (BGR)"""
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)


def draw_rounded_
