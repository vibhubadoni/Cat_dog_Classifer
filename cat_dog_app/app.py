import os
import sys
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import cv2
import numpy as np
from PIL import Image, ImageTk
import tensorflow as tf
from ui.main_window import MainWindow

def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)

def load_model():
    """Load the pre-trained model"""
    try:
        model_path = resource_path("model/cat_dog_classifier.h5")
        model = tf.keras.models.load_model(model_path)
        return model
    except Exception as e:
        messagebox.showerror("Error", f"Failed to load model: {str(e)}")
        return None

def main():
    # Initialize the main application
    root = tk.Tk()
    root.title("Cat vs Dog Classifier")
    
    # Set window size and center it
    window_width = 800
    window_height = 600
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    center_x = int(screen_width/2 - window_width/2)
    center_y = int(screen_height/2 - window_height/2)
    root.geometry(f'{window_width}x{window_height}+{center_x}+{center_y}')
    
    # Set application icon (if available)
    try:
        icon_path = resource_path("assets/icon.ico")
        root.iconbitmap(icon_path)
    except:
        pass  # Icon is optional
    
    # Load the model
    model = load_model()
    if model is None:
        return  # Exit if model loading failed
    
    # Create and run the main application window
    app = MainWindow(root, model)
    root.mainloop()

if __name__ == "__main__":
    main()
