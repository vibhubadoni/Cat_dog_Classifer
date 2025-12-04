# Cat vs Dog Classifier

A desktop application that classifies images as either "Cat" or "Dog" using a pre-trained TensorFlow model.

## Features

- Capture images using your webcam
- Upload images from your computer
- Real-time classification with confidence scores
- Clean and user-friendly interface

## Requirements

- Python 3.8 or higher
- Webcam (for capturing images)
- Required Python packages (see requirements.txt)

## Installation

1. Clone this repository or download the source code
2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```
3. Make sure you have the model file `cat_dog_classifier.h5` in the `model` directory

## Running the Application

To run the application, navigate to the project directory and run:

```bash
python app.py
```

## Building the Application (Optional)

To create a standalone executable, you can use PyInstaller:

1. Install PyInstaller if you haven't already:
   ```bash
   pip install pyinstaller
   ```

2. Build the executable:
   ```bash
   pyinstaller --onefile --windowed --add-data "model;model" --add-data "assets;assets" app.py
   ```

3. The executable will be created in the `dist` directory

## Usage

1. Click "Click Image" to take a photo using your webcam
   OR
   Click "Upload Image" to select an image file from your computer

2. The application will process the image and display the prediction along with a confidence score

## Troubleshooting

- If you encounter issues with the webcam, make sure no other application is using it
- Ensure all required packages are installed
- Make sure the model file is in the correct location (`model/cat_dog_classifier.h5`)

## License

This project is licensed under the MIT License - see the LICENSE file for details.
