# AI Image Classifier

## Overview
This project is an AI-powered web application built with Streamlit that uses a pre-trained MobileNetV2 model to classify objects in uploaded images (JPG or PNG formats). The application leverages deep learning to identify objects from the ImageNet dataset, providing users with the top three predictions along with confidence scores. The project showcases image processing with OpenCV, model inference with TensorFlow, and a user-friendly interface with Streamlit, making it an excellent example of combining deep learning with web development for practical AI applications.

## Project Steps
Here’s a step-by-step guide to setting up and running the project:

1. **Initialize a New UV Project**  
   Inside your project directory, initialize a new UV project to set up a virtual environment and project structure:  
   ```bash
   uv init .
   ```

2. **Install Dependencies**  
   Install the required Python packages using UV, a faster alternative to pip for package installation and dependency resolution:  
   ```bash
   uv add streamlit tensorflow opencv-python
   ```

3. **Write the Code**  
   Create a `main.py` file and implement the code for the AI Image Classifier. The code:  
   - Uses Streamlit to create a web interface for uploading and displaying images.  
   - Loads a pre-trained MobileNetV2 model from TensorFlow with ImageNet weights.  
   - Processes uploaded images using OpenCV and Pillow for compatibility with the model.  
   - Performs image classification and displays the top three predictions with confidence scores.  
   See the `main.py` file in this repository for the complete implementation.

4. **Run the Application**  
   Activate the UV virtual environment and run the Streamlit app:  
   ```bash
   uv run streamlit run main.py
   ```  
   Open the provided local URL (typically `http://localhost:8501`) in a browser to interact with the app. Upload a JPG or PNG image, click "Classify Image," and view the AI’s predictions.

## Python Packages Used
The following Python packages are required for this project:
- **streamlit**: Creates an interactive web interface for uploading images and displaying results.
- **tensorflow**: Provides the MobileNetV2 model and tools for deep learning inference.
- **opencv-python**: Handles image processing tasks like resizing for model compatibility.
- **pillow**: Manages image file operations for loading and displaying uploaded images.
- **numpy**: Supports numerical operations on image arrays for preprocessing.
- **uv**: A high-performance Python package installer and resolver used instead of pip for dependency management.

## Additional Information
- **How to Use the Application**:  
   After running the app, access it via a web browser. Upload a JPG or PNG image, click "Classify Image," and the app will display the top three predicted objects along with their confidence scores. The application uses the pre-trained MobileNetV2 model, which recognizes a wide variety of objects from the ImageNet dataset.

- **Why UV?**  
   UV is a modern, high-performance alternative to pip, offering faster dependency resolution and installation. It simplifies project setup and ensures a clean virtual environment. Learn more at [UV’s official documentation](https://docs.astral.sh/uv/).
