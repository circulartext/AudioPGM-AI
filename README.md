# AI-Driven Melody Generation and Matching

## Project Description

This project combines artificial intelligence and audio processing to create and match musical patterns. It provides tools to:

- Extract note segments from an audio file containing a melody.
- Visualize and modify these extracted notes, allowing reordering or deletion of unwanted segments.
- Generate new musical sequences using the extracted notes.
- Employ AI-driven optimization to match generated melodies to a target melody.
- Visualize and compare the generated melodies with the target melody using detailed audio analysis.

## Required Files

- `notes.py`: Extracts note segments from an audio file.
- `modify.py`: Provides a GUI to visualize and modify extracted note segments.
- `main_script.py`: Performs melody matching and AI optimization.
- `sound1.wav`: Example audio file containing a melody for note extraction.
- `target_melody.wav`: Audio file containing the target melody for matching.
- `notes/`: Directory to store extracted note segments and metadata.

## System Requirements

- Python 3.6 or higher
- Operating System: Windows, macOS, or Linux
- Audio Playback Capability

## Installation

1. Clone or download the repository:
   ```bash
   git clone <repository-url>
   cd <project-directory>

   ## Expected Output

When you run the `main_script.py`, the following outputs are generated:

- **Generated Melodies**: 
  - Audio files saved in the `melody_matching_results` directory as `generated_melody_round_X.wav`.

- **Visualizations**: 
  - Plots comparing generated and target melodies saved as `melody_match_round_X.png` in the `melody_matching_results` directory.

- **Results Summary**: 
  - A CSV file `training_results.csv` summarizing each round's results, including similarity scores and matrix configurations.

## Understanding the Output

- **Generated Melodies**: These are audio files created by the AI, representing the best attempts to match the target melody. Listen to these files to evaluate the quality and similarity to the target.

- **Visualizations**: The plots provide a visual comparison between the generated and target melodies, showing waveform and chromagram similarities.

- **Results Summary**: The CSV file contains detailed information about each optimization round, including the similarity percentage achieved. Use this data to analyze the effectiveness of different configurations and improvements over time.

LICENCE 
Canonical URL  https://creativecommons.org/licenses/by-nc-nd/4.0/