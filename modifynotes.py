import os
import tkinter as tk
from tkinter import messagebox
import sounddevice as sd
import soundfile as sf
import numpy as np
import shutil
import re
import json

class NotesViewer:
    def __init__(self, notes_dir='notes'):
        self.notes_dir = notes_dir

        # Custom sorting function to handle numeric order
        def custom_sort(filename):
            match = re.search(r'note_segment_(\d+)\.wav', filename)
            return int(match.group(1)) if match else float('inf')

        # Sort files numerically
        self.original_note_files = sorted(
            [f for f in os.listdir(notes_dir) if f.startswith('note_segment_') and f.endswith('.wav')],
            key=custom_sort
        )
        self.current_note_files = self.original_note_files.copy()

        # Create main window
        self.root = tk.Tk()
        self.root.title("Note Segments Viewer")
        self.root.geometry("1600x1000")  # Wider window

        # Create canvas with scrollbar
        self.canvas = tk.Canvas(self.root)
        self.scrollbar = tk.Scrollbar(self.root, orient=tk.VERTICAL, command=self.canvas.yview)
        self.canvas.configure(yscrollcommand=self.scrollbar.set)

        self.scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Save button
        self.save_button = tk.Button(self.root, text="Save Order", command=self.save_order)
        self.save_button.pack(side=tk.BOTTOM)

        # Display waveforms
        self.display_waveforms()

        self.root.mainloop()

    def display_waveforms(self):
        # Clear previous drawings
        self.canvas.delete('all')

        # Drawing parameters
        rect_width = 120  # Slightly wider
        rect_height = 250  # Slightly taller
        padding_x = 70  # Reduced horizontal spacing
        padding_y = 70  # Reduced vertical spacing
        cols = 4

        for i, filename in enumerate(self.current_note_files):
            # Calculate row and column
            row = i // cols
            col = i % cols

            # Calculate rectangle position
            x1 = col * (rect_width + padding_x) + 50  # Added left margin
            y1 = row * (rect_height + padding_y) + 50  # Added top margin
            x2 = x1 + rect_width
            y2 = y1 + rect_height

            # Draw rectangle
            rect = self.canvas.create_rectangle(x1, y1, x2, y2, fill='white', outline='black')

            # Load audio file
            filepath = os.path.join(self.notes_dir, filename)
            audio, sr = sf.read(filepath)

            # Normalize audio
            audio = audio / np.max(np.abs(audio))

            # Create vertical waveform inside rectangle
            for j, sample in enumerate(audio[::max(1, len(audio)//rect_height)]):
                if j >= rect_height:
                    break

                # Map sample to rectangle width
                wave_x = x1 + rect_width/2 + sample * (rect_width/2)
                wave_y = y1 + rect_height - j  # Reversed to go upwards

                # Draw a point for the waveform
                self.canvas.create_oval(wave_x-2, wave_y-2, wave_x+2, wave_y+2, fill='blue')

            # Add filename label
            self.canvas.create_text(x1 + rect_width/2, y2 + 20, text=filename, font=('Arial', 10))

            # Add play button
            play_button = tk.Button(self.root, text="Play", 
                                    command=lambda f=filepath: self.play_audio(f),
                                    width=6)
            play_button_window = self.canvas.create_window(x1 + rect_width/4, y2 + 50, 
                                                           window=play_button)

            # Add delete button
            delete_button = tk.Button(self.root, text="Delete", 
                                      command=lambda f=filename: self.delete_note(f),
                                      width=6)
            delete_button_window = self.canvas.create_window(x1 + 3*rect_width/4, y2 + 50, 
                                                             window=delete_button)

            # Add left arrow button
            if i > 0:  # Only add if not the first segment
                left_arrow = tk.Button(self.root, text="\u2190",  # Unicode for left arrow
                                       command=lambda idx=i: self.move_left(idx),
                                       width=2)
                left_arrow_window = self.canvas.create_window(x1 - 30, y1 + rect_height/2, 
                                                              window=left_arrow)

            # Add right arrow button
            if i < len(self.current_note_files) - 1:  # Only add if not the last segment
                right_arrow = tk.Button(self.root, text="\u2192",  # Unicode for right arrow
                                        command=lambda idx=i: self.move_right(idx),
                                        width=2)
                right_arrow_window = self.canvas.create_window(x2 + 30, y1 + rect_height/2, 
                                                               window=right_arrow)

        # Update scroll region
        self.canvas.configure(scrollregion=self.canvas.bbox('all'))

    def play_audio(self, filepath):
        # Play audio file
        audio, sr = sf.read(filepath)
        sd.play(audio, sr)
        sd.wait()

    def delete_note(self, filename):
        # Remove from current list
        self.current_note_files.remove(filename)

        # Refresh display
        self.display_waveforms()

    def move_left(self, index):
        if index > 0:
            # Swap elements
            self.current_note_files[index], self.current_note_files[index-1] = \
            self.current_note_files[index-1], self.current_note_files[index]

            # Refresh display
            self.display_waveforms()

    def move_right(self, index):
        if index < len(self.current_note_files) - 1:
            # Swap elements
            self.current_note_files[index], self.current_note_files[index+1] = \
            self.current_note_files[index+1], self.current_note_files[index]

            # Refresh display
            self.display_waveforms()

    def save_order(self):
        # Confirm save
        if messagebox.askyesno("Confirm", "Save the new order of note segments?"):
            # Create a temporary directory
            temp_dir = os.path.join(self.notes_dir, 'temp')
            os.makedirs(temp_dir, exist_ok=True)

            # Load existing metadata
            metadata_path = os.path.join(self.notes_dir, 'note_segments_metadata.json')
            try:
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
            except FileNotFoundError:
                metadata = []

            # Handle different metadata formats
            if isinstance(metadata, list):
                # Convert list to dictionary if needed
                metadata_dict = {}
                for entry in metadata:
                    metadata_dict[entry['filename']] = entry
                metadata = metadata_dict

            # Create a new metadata list to store reordered entries
            new_metadata = []

            # Copy files to temp directory with new names and update metadata
            for i, filename in enumerate(self.current_note_files):
                new_filename = f'note_segment_{i}.wav'
                
                # Copy audio file
                src_path = os.path.join(self.notes_dir, filename)
                dst_path = os.path.join(temp_dir, new_filename)
                shutil.copy2(src_path, dst_path)
                
                # Update metadata
                if filename in metadata:
                    # Create a new entry with updated filename
                    entry = metadata[filename].copy()
                    entry['filename'] = new_filename
                    new_metadata.append(entry)

            # Remove old files
            for filename in os.listdir(self.notes_dir):
                if filename.startswith('note_segment_') and filename.endswith('.wav'):
                    os.remove(os.path.join(self.notes_dir, filename))

            # Move files from temp to notes directory
            for filename in os.listdir(temp_dir):
                src_path = os.path.join(temp_dir, filename)
                dst_path = os.path.join(self.notes_dir, filename)
                shutil.move(src_path, dst_path)

            # Remove temporary directory
            os.rmdir(temp_dir)

            # Save updated metadata
            with open(metadata_path, 'w') as f:
                json.dump(new_metadata, f, indent=4)

            # Reset and refresh
            self.original_note_files = sorted([f for f in os.listdir(self.notes_dir) if f.startswith('note_segment_') and f.endswith('.wav')])
            self.current_note_files = self.original_note_files.copy()
            self.display_waveforms()

            messagebox.showinfo("Success", "Note segments saved in new order!")

# Run the viewer
if __name__ == '__main__':
    NotesViewer()