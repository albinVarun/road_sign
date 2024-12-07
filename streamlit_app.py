import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
import pandas as pd
from PIL import Image
import io

# Constants
IMG_SIZE = 32

class RoadSignPredictor:
    def __init__(self):
        # Load the model
        self.model = tf.keras.models.load_model('/home/stilskin/Python/project/final_upload/road_sign_detection_testcas4.h5')
        
        # Load class labels
        self.labels_df = pd.read_csv('/home/stilskin/Python/project/road_sign/Indian-Traffic Sign-Dataset/traffic_sign.csv')
    
    def preprocess_image(self, image):
        # Convert PIL Image to numpy array
        image = np.array(image)
        # Convert RGB to BGR (if needed)
        if len(image.shape) == 3 and image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        # Resize image
        image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
        # Add batch dimension
        image = np.expand_dims(image, axis=0)
        return image
    
    def get_prediction(self, image):
        # Get model prediction
        predictions = self.model.predict(image)
        predicted_class = np.argmax(predictions[0])
        confidence = predictions[0][predicted_class]
        
        # Get class name from CSV
        class_name = self.labels_df[self.labels_df['ClassId'] == predicted_class]['Name'].values
        if len(class_name) > 0:
            class_name = class_name[0]
        else:
            class_name = "Unknown"
            
        return class_name, confidence

def main():
    st.set_page_config(
        page_title="Road Sign Detector",
        page_icon="🚸",
        layout="centered"
    )
    
    st.title("Road Sign Detection")
    st.write("Upload an image of a road sign to identify it!")
    
    # Initialize predictor
    predictor = RoadSignPredictor()
    
    # File uploader
    uploaded_file = st.file_uploader("Choose an image...", type=['jpg', 'jpeg', 'png'])
    
    if uploaded_file is not None:
        # Display the uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)
        
        # Add a prediction button
        if st.button('Predict Sign'):
            with st.spinner('Analyzing image...'):
                # Preprocess image and get prediction
                processed_image = predictor.preprocess_image(image)
                class_name, confidence = predictor.get_prediction(processed_image)
                
                # Display results
                st.success("Prediction Complete!")
                st.write("---")
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Predicted Sign", class_name)
                with col2:
                    st.metric("Confidence", f"{confidence:.2%}")
                
                # Additional information
                st.write("---")
                st.write("### Prediction Details")
                st.write(f"- **Sign Type:** {class_name}")
                st.write(f"- **Confidence Score:** {confidence:.2%}")
                if confidence < 0.5:
                    st.warning("⚠️ Low confidence prediction. The image might be unclear or the sign might not be in our dataset.")
                elif confidence > 0.8:
                    st.success("✅ High confidence prediction!")

if __name__ == "__main__":
    main()
