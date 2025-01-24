# AudioPGM-AI

AI-Driven Melody Generation and Matching
This application leverages artificial intelligence to generate musical patterns, match them to a target melody, and allows users to extract, modify, and utilize note segments from an audio file. It combines audio processing, machine learning, and interactive visualization to create a unique musical experience.

Table of Contents
Overview
Features
Required Files
System Requirements
Installation
Usage
1. Extracting Note Segments (notes.py)
2. Modifying Note Segments (modify.py)
3. Melody Matching and AI Optimization (Main Script)
File Structure
Dependencies
Notes
Credits
Overview
This application allows users to:

Extract note segments from an audio file containing a melody.
Visualize and modify these extracted notes (reorder or delete unwanted notes).
Use these notes to generate new musical sequences.
Employ an AI-driven optimization process to match generated melodies to a target melody.
Visualize and compare the generated melodies with the target melody.
Features
Note Extraction: Automatically detects and extracts individual notes from an audio file.
Interactive Note Modification: Provides a GUI to visualize, reorder, and delete extracted note segments.
AI Optimization: Uses machine learning to optimize matrices that represent musical patterns.
Audio Generation: Converts numerical matrices into audio sequences using the extracted note segments.
Melody Matching: Compares generated melodies to a target melody using audio analysis techniques.
Visualization: Generates waveforms and chromagrams for visual comparison.
Required Files
notes.py : Script to extract note segments from an audio file.
modify.py : GUI application to visualize and modify extracted note segments.
Main Script : The primary script that performs melody matching and AI optimization (provided earlier).
An audio file with a melody (e.g., sound1.wav) to extract notes from.
An audio file containing the target melody for matching.
A directory named notes to store the extracted note segments and metadata.
System Requirements
Python 3.6 or higher
Operating System: Windows, macOS, or Linux
Audio Playback Capability: Required for playing audio within the application.
Installation
Clone or Download the Repository

BASH

git clone <repository_url>
Install Required Python Packages

It's recommended to use a virtual environment.

BASH

pip install numpy librosa soundfile matplotlib tensorflow scikit-learn pandas sounddevice
Additional dependencies:

tkinter (should be included with Python on most systems)
scipy
pyaudio (for sound playback)
Usage
1. Extracting Note Segments (notes.py)
This script extracts individual note segments from an audio file containing a melody.

Steps:

Ensure you have an audio file (e.g., sound1.wav) from which you wish to extract notes. Place this file in your working directory.

Run the notes.py script:

BASH

python notes.py
Alternatively, if your audio file has a different name or path:

BASH

python notes.py path/to/your/audiofile.wav
What It Does:

Loads the specified audio file.
Detects note onsets using librosa.
Extracts segments around each onset.
Saves each note segment as a separate .wav file in the notes directory.
Creates a note_segments_metadata.json file containing metadata about each segment.
Expected Output:

Extracted note segments saved as note_segment_0.wav, note_segment_1.wav, etc., in the notes directory.
Metadata file note_segments_metadata.json saved in the notes directory.
2. Modifying Note Segments (modify.py)
This GUI application allows you to visualize, reorder, and delete extracted note segments.

Steps:

Run the modify.py script:

BASH

python modify.py
What It Does:

Displays a window showing all extracted note segments.
Allows you to play each note, delete unwanted notes, and reorder notes by moving them left or right.
Upon saving, updates the notes and metadata accordingly.
How to Use:

Play a Note: Click the "Play" button below a note's waveform to listen to it.
Delete a Note: Click the "Delete" button to remove a note segment.
Reorder Notes: Use the left (←) and right (→) arrow buttons to move notes within the sequence.
Save Changes: Click the "Save Order" button to apply your changes. This will overwrite the existing note files and metadata.
Notes:

The GUI displays waveforms of each note for visual reference.
Changes are saved to the notes directory, updating both the .wav files and the metadata.
3. Melody Matching and AI Optimization (Main Script)
This is the main script that uses the note segments to generate melodies and optimizes them to match a target melody.

Prerequisites:

Ensure you have a target melody audio file (e.g., sound1.wav) for matching.
Ensure the notes directory contains the note segments and metadata.
Steps:

Run the main script (replace main_script.py with the actual filename if different):

BASH

python main_script.py
What It Does:

Loads the target melody and note segments.
Initializes the AI optimization process.
Generates random matrices representing musical patterns.
Simulates the spread of numbers in the matrix to create audio sequences using the note segments.
Compares the generated melodies to the target melody using chromagram analysis.
Trains a neural network model to predict and improve similarity over iterations.
Saves the best matching melody and accompanying visualizations.
Expected Output:

Audio Files:

Generated melodies saved in the melody_matching_results directory as generated_melody_round_X.wav.
Visualizations:

Plots comparing the generated and target melodies saved in melody_matching_results as melody_match_round_X.png.
Results File:

A CSV file training_results.csv summarizing each round's results.
Notes:

The number of rounds for optimization can be adjusted in the script (default is set to 1000).
The process may take some time depending on the number of rounds and computational resources.
The AI model improves over time, enhancing the similarity of generated melodies to the target.
File Structure
notes.py: Script to extract note segments from an audio file.
modify.py: GUI application for visualizing and modifying note segments.
main_script.py: Main script for melody matching and AI optimization.
notes/: Directory containing note segment .wav files and metadata.
melody_matching_results/: Directory where results (audio and plots) are saved.
sound1.wav: Example audio file containing the target melody.
note_segments_metadata.json: JSON file containing metadata about the note segments.
training_results.csv: CSV file summarizing the results of each optimization round.
Dependencies
Ensure that the following Python packages are installed:

numpy
librosa
matplotlib
scipy
soundfile
sounddevice
tensorflow
scikit-learn
pandas
tkinter (usually included with Python)
json
os
shutil
re
Use the following command to install the required packages:

BASH

pip install numpy librosa matplotlib scipy soundfile sounddevice tensorflow scikit-learn pandas
Notes
Audio Files: The quality of the generated melodies depends on the clarity and quality of the input audio files.
Performance: Running the AI optimization may be resource-intensive. Using a machine with a good CPU/GPU is recommended.
Error Handling: The scripts include basic error handling; however, ensure that all files and directories are correctly placed to prevent issues.
Customization: Parameters like segment duration in notes.py and the number of rounds in the main script can be adjusted to suit your needs.
Credits
This application was developed to explore the intersection of artificial intelligence and music technology, showcasing innovative methods of melody generation and pattern matching through code.

Feel free to reach out if you have any questions or need assistance with the application!

Enjoy creating and exploring musical patterns with AI!




Audio Pattern Generation and Matching
link to Linkedin article
https://www.linkedin.com/pulse/bridging-ai-music-technical-dive-audio-pattern-generation-group-po1ac/?trackingId=yDziENFMR%2BihHsVov3qhcg%3D%3D

