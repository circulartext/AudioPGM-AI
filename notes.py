import numpy as np
import librosa
import os
import soundfile as sf

def extract_note_segments(audio_file, output_dir='notes', segment_duration=0.5):
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load the audio file
    y, sr = librosa.load(audio_file)
    
    # Compute onset times
    onset_frames = librosa.onset.onset_detect(y=y, sr=sr)
    
    # Convert onset frames to time
    onset_times = librosa.frames_to_time(onset_frames, sr=sr)
    
    # Extract segments around onsets
    note_segments = []
    for i, onset_time in enumerate(onset_times):
        # Calculate start and end of segment
        start_time = max(0, onset_time - segment_duration/2)
        end_time = min(len(y)/sr, onset_time + segment_duration/2)
        
        # Convert times to samples
        start_sample = int(start_time * sr)
        end_sample = int(end_time * sr)
        
        # Extract segment
        segment = y[start_sample:end_sample]
        
        # Generate filename
        segment_filename = os.path.join(output_dir, f'note_segment_{i}.wav')
        
        # Save segment as WAV file
        sf.write(segment_filename, segment, sr)
        
        # Store segment info
        note_segments.append({
            'filename': f'note_segment_{i}.wav',
            'start_time': start_time,
            'duration': end_time - start_time
        })
    
    # Create a JSON file with segment metadata
    import json
    metadata_file = os.path.join(output_dir, 'note_segments_metadata.json')
    with open(metadata_file, 'w') as f:
        json.dump(note_segments, f, indent=4)
    
    print(f"Extracted {len(note_segments)} note segments")
    print(f"Metadata saved to {metadata_file}")
    
    return note_segments

def main():
    # Path to your audio file
    audio_file = 'sound1.wav'  # Replace with your audio file
    
    # Extract note segments
    note_segments = extract_note_segments(audio_file)

if __name__ == '__main__':
    main()