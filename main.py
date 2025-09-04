import cv2      # OpenCV for image processing
import numpy as np      # NumPy for numerical operations on arrays
import streamlit as st      # Streamlit for creating the web application UI
from tensorflow.keras.applications.mobilenet_v2 import(
    MobileNetV2,
    preprocess_input,
    decode_predictions
)
from PIL import Image       # Pillow for handling image files


# This function loads the MobileNetV2 deep learning model.
# It uses pre-trained weights from the 'imagenet' dataset, which allows it to recognize a wide variety of objects without requiring new training.
def load_model():
    model = MobileNetV2(weights="imagenet")
    return model

# This function preprocesses an image to make it compatible with the MobileNetV2 model.
def preprocess_image(image):

    # Convert the Pillow image object to a NumPy array.
    img = np.array(image)       

    # Resize the image to 224x224 pixels, as required by the MobileNetV2 input layer.
    img = cv2.resize(img, (224, 224))   

    # Preprocess the image according to the model's specific requirements    
    img = preprocess_input(img)    

    # Add a new dimension to the array to represent the batch size. The model expects a batch of images, so we create a batch of one image.     
    img = np.expand_dims(img, axis=0)   
    return img

# This function takes a pre-trained model and an image, then returns the top predictions.
def classify_image(model, image):
    try:
        # Prepare the image for the model.
        processed_image = preprocess_image(image) 

        # Make a prediction using the model. The output is a set of probabilities for each class.
        predictions = model.predict(processed_image)  

        # Decode the raw predictions into human-readable labels, scores, and class IDs.
        # We get the top 3 predictions by using top=3.
        decoded_predictions = decode_predictions(predictions, top=3)[0] 

        return decoded_predictions
    
    except Exception as e:

        # Display an error message if classification fails.
        st.error(f"Error classifying image: {str(e)}")
        return None
    
# The main function that defines the Streamlit web application layout and logic.    
def main():
    # Set the configuration for the web page
    st.set_page_config(page_title = "AI Image Classifier", page_icon="🖼️", layout="centered")

    # Display the title and a brief description for the user.
    st.title("AI Image Classifier")
    st.write("Upload an image and let AI tell you what is in it!")

    # Use Streamlit's cache_resource decorator to load the model only once.
    # This prevents the model from being reloaded every time the user interacts with the app,
    # significantly improving performance.
    @st.cache_resource
    def load_cached_model():
        return load_model()
    
    # Load the model using the cached function.
    model = load_cached_model() 

    # Create a file uploader widget. The app accepts jpg and png file types.
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png"])

    # This block of code runs only if a file has been uploaded.
    if uploaded_file is not None:
        # Display the uploaded image with a caption.
        image = st.image(
            uploaded_file, caption = "Uploaded Image", use_container_width=True
        )
        # Create a button to trigger the classification.
        btn = st.button("Classify Image")

        # This block runs when the "Classify Image" button is clicked.
        if btn: 
            # Show a spinner while the image is being processed.
            with st.spinner("Analyzing image..."):
                # Open the uploaded file as a PIL Image object.
                image = Image.open(uploaded_file)

                # Call the classification function to get predictions.
                predictions = classify_image(model, image)

                # Display the predictions if they are successfully returned.
                if predictions:
                    st.subheader("Predictions")
                    # Loop through the top 3 predictions and display the label and confidence score.
                    for _, label, score in predictions:
                        st.write(f"**{label}**: {score:.2%}")

# The entry point of the script.
if __name__ == "__main__":
    main()