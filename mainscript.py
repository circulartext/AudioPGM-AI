import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from scipy.io import wavfile
import os
import json
import pandas as pd
import random
from collections import deque

# Machine Learning Imports
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

# Constants for sound generation
SAMPLE_RATE = 48000
COLUMN_DURATION = 1  # Duration of each column in seconds
MAX_AUDIO_DURATION = 15  # Maximum allowed audio duration in seconds
OUTPUT_FOLDER = "melody_matching_results"

def generate_tone(frequency, duration):
    """Generates a tone for a specified frequency and duration."""
    t = np.linspace(0, duration, int(SAMPLE_RATE * duration), endpoint=False)
    
    # Generate tone with harmonics
    tone = 0.5 * np.sin(2 * np.pi * frequency * t)
    tone += 0.25 * np.sin(2 * np.pi * (frequency * 2) * t)  # 2nd harmonic
    tone += 0.125 * np.sin(2 * np.pi * (frequency * 3) * t)  # 3rd harmonic
    
    # Apply soft envelope
    envelope = np.ones_like(tone)
    attack = int(0.1 * len(tone))
    release = int(0.1 * len(tone))
    
    envelope[:attack] = np.linspace(0, 1, attack)
    envelope[-release:] = np.linspace(1, 0, release)
    
    return tone * envelope * 0.5  # Reduce amplitude

def note_to_frequency(note):
    """Convert a note name to its frequency with more robust handling."""
    note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    
    # Normalize note representation
    note = note.replace('♯', '#').upper()
    
    # Default to A4 if note not recognized
    if note not in note_names:
        print(f"Warning: Note {note} not found. Defaulting to A4.")
        return 440.0
    
    # Default to 4th octave
    octave = 4
    note_idx = note_names.index(note)
    midi_number = note_idx + (octave + 1) * 12
    
    return 440 * 2**((midi_number - 69) / 12)

def pad_audio(audio, target_length):
    """Pad or truncate audio to target length."""
    if len(audio) < target_length:
        return np.concatenate([audio, np.zeros(target_length - len(audio))])
    else:
        return audio[:target_length]

class AIMatrixOptimizer:
    def __init__(self, max_memory=100):
        self.scaler = StandardScaler()
        self.training_data = deque(maxlen=max_memory)
        self.training_labels = deque(maxlen=max_memory)
        self.is_trained = False
        self.model = self.build_model()

    def build_model(self):
        model = Sequential()
        model.add(Dense(128, input_dim=6, activation='relu'))  # Adjust input_dim based on number of features
        model.add(Dropout(0.2))  # Prevent overfitting
        model.add(Dense(64, activation='relu'))
        model.add(Dropout(0.1))
        model.add(Dense(1, activation='linear'))  # Output layer for regression
        model.compile(loss='mean_squared_error', optimizer=tf.keras.optimizers.Adam(learning_rate=0.001))
        return model

    def matrix_to_features(self, matrix):
        # Using the enhanced feature extraction method
        matrix = np.array(matrix)
        features = []
        # Number of non-zero entries
        features.append(np.count_nonzero(matrix))
        # Sum of non-zero values
        non_zero_values = matrix[matrix > 0]
        features.append(np.sum(non_zero_values))
        # Mean of non-zero values
        features.append(np.mean(non_zero_values) if non_zero_values.size > 0 else 0)
        # Standard deviation of non-zero values
        features.append(np.std(non_zero_values) if non_zero_values.size > 0 else 0)
        # Mean row index of non-zero entries
        indices = np.argwhere(matrix > 0)
        features.append(np.mean(indices[:, 0]) if indices.size > 0 else 0)
        # Mean column index of non-zero entries
        features.append(np.mean(indices[:, 1]) if indices.size > 0 else 0)
        return np.array(features)

    def train(self):
        if len(self.training_data) < 50:
            return False
        X = np.array(list(self.training_data))
        y = np.array(list(self.training_labels))
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y, epochs=2, batch_size=32, verbose=0)
        self.is_trained = True
        return True

    def predict(self, matrix):
        if not self.is_trained:
            return 0.0
        features = self.matrix_to_features(matrix)
        features_scaled = self.scaler.transform(features.reshape(1, -1))
        return self.model.predict(features_scaled)[0][0]

    def add_training_example(self, matrix, similarity):
        features = self.matrix_to_features(matrix)
        self.training_data.append(features)
        self.training_labels.append(similarity)

class NumberSpreadSimulator:
    def __init__(self, initial_matrix, note_segments, max_steps=1000):
        self.grid = np.array(initial_matrix, dtype=int)
        self.note_segments = note_segments
        self.audio_frames = []
        self.max_steps = max_steps  # Maximum number of simulation steps
        self.steps_taken = 0  # Counter for steps taken
        
        # Track total audio duration
        self.total_audio_duration = 0  # in seconds
        
        # Keep track of the original starting values of numbers
        self.original_values = {}
        for row in range(self.grid.shape[0]):
            for col in range(self.grid.shape[1]):
                value = self.grid[row, col]
                if value > 1 and value != 11:
                    # Store the original value at this position
                    self.original_values[(row, col)] = value
        
        self.initialize_audio()

    def initialize_audio(self):
        """Generate audio for initial state and store it."""
        audio = self.get_matrix_audio()
        if len(audio) > 0:
            self.audio_frames.append(audio)
            self.total_audio_duration += len(audio) / SAMPLE_RATE

    def get_matrix_audio(self):
        """Generate audio sequence using note segments."""
        audio_sequence = []
        num_rows = self.grid.shape[0]

        for col in range(self.grid.shape[1]):
            col_audio = np.zeros(0)
            for row in range(num_rows):
                num = self.grid[row, col]
                if num > 1 and num != 11:
                    # Select a note segment based on the column index
                    segment_index = min(col, len(self.note_segments) - 1)
                    segment = self.note_segments[segment_index]

                    # Ensure consistent length
                    target_length = int(SAMPLE_RATE * COLUMN_DURATION / num_rows)
                    if len(segment) > target_length:
                        segment = segment[:target_length]
                    elif len(segment) < target_length:
                        segment = np.pad(segment, (0, target_length - len(segment)), 'constant')

                    col_audio = np.concatenate((col_audio, segment))

                elif num == 11:
                    # For 11 (formerly -1), add silence
                    silence = np.zeros(int(SAMPLE_RATE * COLUMN_DURATION / num_rows))
                    col_audio = np.concatenate((col_audio, silence))

            if len(col_audio) > 0:
                audio_sequence.append(col_audio)

        return np.concatenate(audio_sequence) if audio_sequence else np.zeros(0)

    def step(self):
        """Simulates a step in the matrix spread."""
        if self.steps_taken >= self.max_steps:
            return False, self.grid  # Terminate simulation after max_steps
        
        if self.total_audio_duration >= MAX_AUDIO_DURATION:
            return False, self.grid  # Terminate simulation if audio duration limit is reached
        
        new_grid = np.copy(self.grid)
        has_changes = False
        active_numbers_exist = False  # Flag to check if any numbers can still spread

        # Keep track of positions where numbers have interacted with 11 (formerly -1)
        positions_already_updated = set()

        # Process each position in the grid
        for row in range(self.grid.shape[0]):
            for col in range(self.grid.shape[1]):
                current = self.grid[row, col]

                if current > 1 and current != 11:
                    active_numbers_exist = True  # At least one number can still spread
                    half = current // 2

                    # Define potential target positions
                    targets = []
                    if current % 2 == 0:  # Even
                        if row > 0: targets.append((row - 1, col, half))
                        if col > 0: targets.append((row, col - 1, half))
                    else:  # Odd
                        if col > 0: targets.append((row, col - 1, half))
                        if row > 0: targets.append((row - 1, col, half))
                        if row > 0 and col > 0: targets.append((row - 1, col - 1, 1))

                    # Process each target position
                    for target_row, target_col, value in targets:
                        # Check bounds
                        if target_row < 0 or target_col < 0:
                            continue
                        if target_row >= self.grid.shape[0] or target_col >= self.grid.shape[1]:
                            continue

                        target_value = self.grid[target_row, target_col]

                        if target_value == 11:
                            # Interaction with 11
                            if (target_row, target_col) not in positions_already_updated:
                                # First time interaction at this position
                                original_value = self.original_values.get((row, col), current)
                                new_grid[target_row, target_col] = original_value
                                positions_already_updated.add((target_row, target_col))
                                # Reduce the value from current position
                                new_grid[row, col] -= value
                                has_changes = True
                            else:
                                # The 11 has already been replaced; treat as normal position
                                new_grid[target_row, target_col] += value
                                new_grid[row, col] -= value
                                has_changes = True
                        else:
                            # Normal spread behavior
                            new_grid[target_row, target_col] += value
                            new_grid[row, col] -= value
                            has_changes = True

                elif current == 11:
                    # Positions with 11 remain static unless a number interacts with them
                    continue

        self.grid = new_grid
        self.steps_taken += 1  # Increment step counter

        if has_changes:
            audio = self.get_matrix_audio()
            if len(audio) > 0:
                self.audio_frames.append(audio)
                self.total_audio_duration += len(audio) / SAMPLE_RATE
                
                if self.total_audio_duration >= MAX_AUDIO_DURATION:
                    # Stop the simulation if audio duration limit is reached
                    return False, self.grid

        # Continue simulation only if changes occurred and active numbers remain
        return has_changes and active_numbers_exist, self.grid

class MelodyMatcher:
    def __init__(self, target_wav, notes_dir='notes', threshold=0.8):
        print(f"Loading target melody: {target_wav}")
        self.target_wav = target_wav
        self.target_audio, self.target_sr = librosa.load(target_wav, sr=SAMPLE_RATE)

        if len(self.target_audio.shape) > 1:
            self.target_audio = self.target_audio.mean(axis=1)

        print(f"Target audio length: {len(self.target_audio)} samples.")

        # Load note segments
        self.note_segments = []
        metadata_file = os.path.join(notes_dir, 'note_segments_metadata.json')

        with open(metadata_file, 'r') as f:
            segment_metadata = json.load(f)

        # Load actual audio segments
        for segment_info in segment_metadata:
            segment_path = os.path.join(notes_dir, segment_info['filename'])
            try:
                segment_audio, _ = librosa.load(segment_path, sr=SAMPLE_RATE)
                self.note_segments.append(segment_audio)
                print(f"Loaded segment {segment_info['filename']} with {len(segment_audio)} samples.")
            except Exception as e:
                print(f"Error loading segment {segment_info['filename']}: {e}")

        print(f"Total note segments loaded: {len(self.note_segments)}")

        if len(self.note_segments) == 0:
            raise ValueError("No note segments loaded. Please check the notes directory and metadata.")

        self.threshold = threshold

    def generate_matrix(self):
        matrix = np.zeros((5, 11), dtype=int)
        num_numbers = random.randint(1, 5)

        # Randomly place numbers from 2 to 10
        for _ in range(num_numbers):
            row = random.randint(0, 4)
            col = random.randint(0, 10)
            value = random.randint(2, 10)
            matrix[row][col] = value

        # Randomly place 11s (formerly -1s)
        num_special = random.randint(0, 2)  # Adjust the number of 11s as desired
        for _ in range(num_special):
            row = random.randint(0, 4)
            col = random.randint(0, 10)
            if matrix[row][col] == 0:  # Ensure we don't overwrite existing numbers
                matrix[row][col] = 11

        return matrix.tolist()

    def compare_audio(self, generated_audio):
        if len(generated_audio) == 0:
            return 0.0

        # Normalize
        generated_audio = generated_audio.astype(float) / np.max(np.abs(generated_audio))
        target_audio = self.target_audio.astype(float) / np.max(np.abs(self.target_audio))

        # Match lengths
        generated_audio = pad_audio(generated_audio, len(self.target_audio))

        # Frequency content comparison
        try:
            target_chroma = librosa.feature.chroma_stft(y=target_audio, sr=self.target_sr)
            gen_chroma = librosa.feature.chroma_stft(y=generated_audio, sr=SAMPLE_RATE)
            chroma_correlation = np.corrcoef(target_chroma.flatten(), gen_chroma.flatten())[0, 1]
            return chroma_correlation * 100
        except Exception as e:
            print(f"Error during audio comparison: {str(e)}")
            return 0.0

    def plot_and_export_comparison(self, target, generated, round_num, similarity):
        os.makedirs(OUTPUT_FOLDER, exist_ok=True)
        
        gen_chroma = librosa.feature.chroma_stft(y=generated, sr=SAMPLE_RATE)
        ref_chroma = librosa.feature.chroma_stft(y=target, sr=SAMPLE_RATE)
        
        plt.figure(figsize=(15, 10))
        
        plt.subplot(4, 1, 1)
        plt.title(f"Generated Melody Waveform (Round {round_num}, Similarity {similarity:.2f}%)")
        plt.plot(np.linspace(0, len(generated) / SAMPLE_RATE, len(generated)), generated)
        
        plt.subplot(4, 1, 2)
        plt.title("Reference Melody Waveform")
        plt.plot(np.linspace(0, len(target) / SAMPLE_RATE, len(target)), target)
        
        plt.subplot(4, 1, 3)
        librosa.display.specshow(gen_chroma, y_axis='chroma', x_axis='time', sr=SAMPLE_RATE)
        plt.title("Generated Melody Chromagram")
        plt.colorbar()
        
        plt.subplot(4, 1, 4)
        librosa.display.specshow(ref_chroma, y_axis='chroma', x_axis='time', sr=SAMPLE_RATE)
        plt.title("Reference Melody Chromagram")
        plt.colorbar()
        
        plt.tight_layout()
        
        plot_filename = os.path.join(OUTPUT_FOLDER, f"melody_match_round_{round_num}.png")
        plt.savefig(plot_filename)
        plt.close()
        
        generated_audio_filename = os.path.join(OUTPUT_FOLDER, f"generated_melody_round_{round_num}.wav")
        wavfile.write(generated_audio_filename, SAMPLE_RATE, (generated * 32767).astype(np.int16))
        
        return plot_filename, generated_audio_filename

class EnhancedMelodyMatcher(MelodyMatcher):
    def __init__(self, target_wav, notes_dir='notes', threshold=0.8):
        super().__init__(target_wav, notes_dir, threshold)
        self.ai_optimizer = AIMatrixOptimizer()
        self.results_data = []

    def generate_random_matrix(self):
        while True:
            matrix = super().generate_matrix()
            # Check if the matrix will produce audio within the time limit
            if self.is_matrix_acceptable(matrix):
                return matrix

    def is_matrix_acceptable(self, matrix):
        sim = NumberSpreadSimulator(matrix, self.note_segments, max_steps=100)
        sim_steps = 0
        while sim_steps < sim.max_steps:
            has_changes, _ = sim.step()
            sim_seteps += 1
            if not has_changes or sim.total_audio_duration >= MAX_AUDIO_DURATION:
                break
        # Check if total audio duration is within limit
        return sim.total_audio_duration <= MAX_AUDIO_DURATION

    def perturb_matrix(self, base_matrix):
        matrix = np.array(base_matrix)
        # Randomly change some elements
        for _ in range(random.randint(1, 3)):
            row = random.randint(0, matrix.shape[0] - 1)
            col = random.randint(0, matrix.shape[1] - 1)
            current_value = matrix[row][col]
            if current_value == 0:
                # Possibly add a special number (11)
                if random.random() < 0.2:
                    matrix[row][col] = 11
                else:
                    matrix[row][col] = random.randint(2, 10)
            else:
                # Remove the existing number
                matrix[row][col] = 0
        return matrix.tolist()

    def optimize_matrix_with_ai(self):
        if not self.ai_optimizer.is_trained:
            return self.generate_random_matrix()

        best_matrix = None
        best_predicted_similarity = -float('inf')

        # Take the top N matrices with highest similarity
        top_matrices = [eval(entry['matrix']) for entry in sorted(self.results_data, key=lambda x: x['similarity'], reverse=True)[:10]]
        
        # Generate candidates by perturbing top matrices
        for base_matrix in top_matrices:
            for _ in range(5):  # Generate multiple variations
                candidate = self.perturb_matrix(base_matrix)
                if self.is_matrix_acceptable(candidate):
                    predicted_similarity = self.ai_optimizer.predict(candidate)
                    if predicted_similarity > best_predicted_similarity:
                        best_predicted_similarity = predicted_similarity
                        best_matrix = candidate
        if best_matrix is None:
            # If no acceptable matrix found, generate a random acceptable matrix
            best_matrix = self.generate_random_matrix()
        return best_matrix

    def find_matching_matrix_with_ai(self, num_rounds=100):
        best_overall_similarity = -float('inf')
        best_overall_matrix = None
        best_overall_audio = None
        best_plot = None
        best_audio_file = None

        print(f"Starting matrix optimization for {num_rounds} rounds...")
        for round_num in range(1, num_rounds + 1):
            try:
                if self.ai_optimizer.is_trained and random.random() < 0.9:
                    matrix = self.optimize_matrix_with_ai()
                    is_ai_generated = True
                else:
                    matrix = self.generate_random_matrix()
                    is_ai_generated = False

                sim = NumberSpreadSimulator(matrix, self.note_segments, max_steps=1000)  # Set max_steps to prevent infinite loops

                # Simulate steps until the termination condition is met
                while True:
                    has_changes, current_matrix = sim.step()
                    if not has_changes:
                        break

                # Check if total audio duration exceeds limit
                if sim.total_audio_duration > MAX_AUDIO_DURATION:
                    continue  # Discard this matrix and proceed to next iteration

                if len(sim.audio_frames) == 0:
                    continue

                final_audio = np.concatenate(sim.audio_frames)
                similarity = self.compare_audio(final_audio)

                # Add to AI training data
                self.ai_optimizer.add_training_example(matrix, similarity)

                # Predicted similarity
                predicted_similarity = self.ai_optimizer.predict(matrix) if self.ai_optimizer.is_trained else None

                # Store result data
                self.results_data.append({
                    'round': round_num,
                    'similarity': similarity,
                    'predicted_similarity': predicted_similarity,
                    'matrix': str(matrix),
                    'is_ai_generated': is_ai_generated
                })

                if similarity > best_overall_similarity:
                    best_overall_similarity = similarity
                    best_overall_matrix = matrix
                    best_overall_audio = final_audio
                    
                    best_plot, best_audio_file = self.plot_and_export_comparison(
                        self.target_audio, final_audio, round_num, similarity
                    )
                    
                    print(f"Round {round_num}: New best match! Similarity={similarity:.2f}%")

                if round_num % 50 == 0:
                    print(f"Round {round_num}: Best similarity so far: {best_overall_similarity:.2f}%")
                    self.export_results()

                # Train AI periodically
                if round_num % 10 == 0:
                    self.ai_optimizer.train()

            except Exception as e:
                import traceback
                traceback.print_exc()
                print(f"Error in round {round_num}: {str(e)}")
                continue

        self.export_results()
        return (best_overall_matrix, best_overall_similarity, best_overall_audio, 
                best_plot, best_audio_file)

    def export_results(self):
        df = pd.DataFrame(self.results_data)
        df.to_csv('training_results.csv', index=False)
        print("Results exported to training_results.csv")

def main():
    try:
        os.makedirs(OUTPUT_FOLDER, exist_ok=True)
        
        notes_dir = 'notes'
        
        matcher = EnhancedMelodyMatcher("sound1.wav", notes_dir, threshold=0.8)
        
        # Run the optimization for a specified number of rounds
        num_rounds = 1000  # You can adjust this number as needed
        result = matcher.find_matching_matrix_with_ai(num_rounds=num_rounds)
        best_matrix, similarity, audio, plot_file, audio_file = result
        
        if best_matrix is not None:
            print("\nOptimization complete!")
            print(f"Best similarity achieved: {similarity:.2f}%")
            print(f"Plot saved to: {plot_file}")
            print(f"Audio saved to: {audio_file}")
            print("\nBest Matrix:")
            for row in best_matrix:
                print(row)
        else:
            print("\nNo matching matrix found.")

    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()