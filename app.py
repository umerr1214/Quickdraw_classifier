import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
from streamlit_drawable_canvas import st_canvas
import io

# Load your trained model
model = tf.keras.models.load_model('sketch_classifier.h5')
#model = tf.keras.models.load_model("quickdraw_model.h5")

# Class labels (without .npz extension)
CLASS_LABELS = ['The Eiffel Tower', 'The Great Wall of China', 'The Mona Lisa',
                'aircraft carrier', 'airplane', 'alarm clock', 'ambulance', 
                'angel', 'animal migration', 'ant']

def preprocess_drawing(image_data):
    # Convert to grayscale and resize to 28x28
    image = Image.fromarray(image_data).convert('L')
    image = image.resize((28, 28))
    
    # Normalize and reshape
    image = np.array(image).astype('float32') / 255.0
    image = image.reshape(1, 28, 28, 1)
    
    return image

def main():
    st.title("Sketch Recognition App")
    st.write("Draw any of the following objects:")
    st.write(", ".join(CLASS_LABELS))

    # Create canvas for drawing
    canvas_result = st_canvas(
        stroke_width=8,
        stroke_color='#000000',
        background_color='#FFFFFF',
        height=280,
        width=280,
        drawing_mode="freedraw",
        key="canvas",
    )

    if st.button('Predict'):
        if canvas_result.image_data is not None:
            # Preprocess the drawing
            image = preprocess_drawing(canvas_result.image_data)
            
            # Make prediction
            prediction = model.predict(image)
            predicted_class = CLASS_LABELS[np.argmax(prediction)]
            confidence = np.max(prediction) * 100
            
            # Show results
            st.write(f"Prediction: {predicted_class}")
            st.write(f"Confidence: {confidence:.2f}%")
            
            # Show top 3 predictions
            top3_idx = np.argsort(prediction[0])[-3:][::-1]
            st.write("Top 3 Predictions:")
            for idx in top3_idx:
                st.write(f"{CLASS_LABELS[idx]}: {prediction[0][idx]*100:.2f}%")

if __name__ == "__main__":
    main()