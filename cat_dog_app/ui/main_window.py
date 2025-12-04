import tkinter as tk
from tkinter import ttk, filedialog
from PIL import Image, ImageTk
import cv2
import numpy as np

class MainWindow:
    def __init__(self, root, model):
        self.root = root
        self.model = model
        self.current_image = None
        self.setup_ui()
    
    def setup_ui(self):
        """Set up the main user interface"""
        # Configure styles
        style = ttk.Style()
        style.configure('TButton', font=('Arial', 12), padding=10)
        style.configure('TLabel', font=('Arial', 12))
        
        # Main container
        main_frame = ttk.Frame(self.root, padding="20")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Title
        title_label = ttk.Label(
            main_frame, 
            text="Cat vs Dog Classifier", 
            font=('Arial', 20, 'bold')
        )
        title_label.pack(pady=(0, 20))
        
        # Button frame
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(pady=10)
        
        # Buttons
        self.capture_btn = ttk.Button(
            button_frame, 
            text="📸 Click Image", 
            command=self.capture_image,
            style='TButton'
        )
        self.capture_btn.pack(side=tk.LEFT, padx=10)
        
        self.upload_btn = ttk.Button(
            button_frame, 
            text="📁 Upload Image", 
            command=self.upload_image,
            style='TButton'
        )
        self.upload_btn.pack(side=tk.LEFT, padx=10)
        
        # Image display
        self.image_label = ttk.Label(main_frame)
        self.image_label.pack(pady=20, expand=True)
        
        # Result display
        self.result_label = ttk.Label(
            main_frame, 
            text="Prediction: None\nConfidence: 0%",
            font=('Arial', 14),
            justify='center'
        )
        self.result_label.pack(pady=(20, 0))
    
    def capture_image(self):
        """Open a preview window to capture an image from the webcam"""
        # Create a new top-level window for the camera preview
        preview_window = tk.Toplevel(self.root)
        preview_window.title("Camera Preview")
        preview_window.resizable(False, False)
        
        # Add a label to display the camera feed
        preview_label = ttk.Label(preview_window)
        preview_label.pack()
        
        # Add a capture button
        capture_btn = ttk.Button(
            preview_window,
            text="Capture",
            command=lambda: self.on_capture_clicked(cap, preview_window)
        )
        capture_btn.pack(pady=10)
        
        # Start the video capture
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            preview_window.destroy()
            tk.messagebox.showerror("Error", "Could not access the camera")
            return
        
        # Function to update the preview
        def update_preview():
            ret, frame = cap.read()
            if ret:
                # Flip the frame horizontally for mirror effect
                frame = cv2.flip(frame, 1)
                # Convert the image from BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # Convert to PIL Image
                img = Image.fromarray(frame_rgb)
                # Resize for preview (optional)
                img.thumbnail((640, 480))
                # Convert to PhotoImage
                imgtk = ImageTk.PhotoImage(image=img)
                # Update the label
                preview_label.imgtk = imgtk
                preview_label.configure(image=imgtk)
            # Schedule the next update
            preview_label.after(10, update_preview)
        
        # Start the preview
        update_preview()
        
        # Handle window close
        preview_window.protocol("WM_DELETE_WINDOW", 
                             lambda: self.on_preview_close(cap, preview_window))
    
    def on_capture_clicked(self, cap, preview_window):
        """Handle capture button click"""
        ret, frame = cap.read()
        if ret:
            # Release the camera
            cap.release()
            # Close the preview window
            preview_window.destroy()
            # Process the captured image
            self.process_image(frame)
    
    def on_preview_close(self, cap, preview_window):
        """Handle preview window close event"""
        # Release the camera
        cap.release()
        # Close the window
        preview_window.destroy()
    
    def upload_image(self):
        """Open a file dialog to upload an image"""
        file_types = [
            ('Image files', '*.jpg *.jpeg *.png'),
            ('All files', '*.*')
        ]
        file_path = filedialog.askopenfilename(filetypes=file_types)
        
        if file_path:
            try:
                # Read the image using OpenCV
                image = cv2.imread(file_path)
                if image is None:
                    raise ValueError("Could not read the image file")
                self.process_image(image)
            except Exception as e:
                tk.messagebox.showerror("Error", f"Failed to load image: {str(e)}")
    
    def process_image(self, image):
        """Process the captured/uploaded image and make a prediction"""
        # Store the original image for display
        self.current_image = image.copy()
        
        # Display the image
        self.display_image(image)
        
        # Preprocess and predict
        processed_image = self.preprocess_image(image)
        prediction = self.model.predict(processed_image)
        
        # Get the prediction result
        class_name = "Dog" if prediction[0][0] > 0.5 else "Cat"
        confidence = max(prediction[0][0], 1 - prediction[0][0]) * 100
        
        # Update the result label
        self.result_label.config(
            text=f"Prediction: {class_name}\n"
                 f"Confidence: {confidence:.1f}%"
        )
    
    def preprocess_image(self, image):
        """Preprocess the image for the model"""
        # Resize to 150x150 (model's expected input size)
        resized = cv2.resize(image, (150, 150))
        # Convert BGR to RGB
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        # Normalize pixel values
        normalized = rgb / 255.0
        # Add batch dimension
        return np.expand_dims(normalized, axis=0)
    
    def display_image(self, image):
        """Display the image in the UI"""
        # Convert from BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Convert to PIL Image
        pil_image = Image.fromarray(image_rgb)
        
        # Calculate aspect ratio and resize for display
        max_width = 400
        max_height = 300
        
        # Calculate new size maintaining aspect ratio
        width, height = pil_image.size
        ratio = min(max_width/width, max_height/height)
        new_size = (int(width * ratio), int(height * ratio))
        
        # Resize the image
        pil_image = pil_image.resize(new_size, Image.Resampling.LANCZOS)
        
        # Convert to PhotoImage
        photo = ImageTk.PhotoImage(image=pil_image)
        
        # Update the image label
        self.image_label.config(image=photo)
        self.image_label.image = photo  # Keep a reference!
