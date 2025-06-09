import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model
import cv2
from sklearn.model_selection import train_test_split
import kagglehub
import glob
import matplotlib.pyplot as plt
from tqdm import tqdm
import gc
import streamlit as st
from PIL import Image
import io
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Input, Conv2D, UpSampling2D, Concatenate

# Constants
BATCH_SIZE = 32  # Increased batch size for faster processing
IMAGE_SIZE = (160, 160)
LEARNING_RATE = 0.001
EPOCHS = 20
CHUNK_SIZE = 25

# Configure TensorFlow to use memory growth
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

def create_density_map(label, image_size=IMAGE_SIZE):
    """
    Create density map from label (single number representing count)
    """
    density_map = np.zeros(image_size, dtype=np.float32)
    count = float(label)
    
    if count > 0:
        # Create a more concentrated density map
        center = (image_size[0] // 2, image_size[1] // 2)
        sigma = min(image_size) / 8  # Adjust sigma based on image size
        
        # Create a 2D Gaussian kernel
        x = np.arange(0, image_size[0], 1)
        y = np.arange(0, image_size[1], 1)
        xx, yy = np.meshgrid(x, y)
        
        # Calculate Gaussian
        gaussian = np.exp(-((xx - center[0])**2 + (yy - center[1])**2) / (2 * sigma**2))
        
        # Normalize and scale by count
        gaussian = gaussian / np.sum(gaussian)
        density_map = gaussian * count
    
    return density_map

class MemoryEfficientDataGenerator(tf.keras.utils.Sequence):
    def __init__(self, image_path, labels_path, batch_size=BATCH_SIZE, image_size=IMAGE_SIZE):
        self.image_path = image_path
        self.labels_path = labels_path
        self.batch_size = batch_size
        self.image_size = image_size
        
        self.images_data = np.load(image_path, mmap_mode='r')
        self.labels_data = np.load(labels_path, mmap_mode='r')
        self.total_images = self.images_data.shape[0]
        
        # Calculate number of batches needed to process all images
        self.num_batches = (self.total_images + self.batch_size - 1) // self.batch_size
    
    def __len__(self):
        return self.num_batches
    
    def __getitem__(self, idx):
        batch_start = idx * self.batch_size
        batch_end = min(batch_start + self.batch_size, self.total_images)
        
        batch_images = np.zeros((batch_end - batch_start, *self.image_size, 3), dtype=np.float32)
        batch_density = np.zeros((batch_end - batch_start, *self.image_size, 1), dtype=np.float32)
        
        for i in range(batch_start, batch_end):
            img = self.images_data[i]
            img_resized = cv2.resize(img, self.image_size).astype(np.float32) / 255.0
            batch_images[i - batch_start] = img_resized
            
            density_map = create_density_map(self.labels_data[i], self.image_size)
            batch_density[i - batch_start] = np.expand_dims(density_map, axis=-1)
        
        return batch_images, batch_density

def create_crowd_counting_model(input_shape=(160, 160, 3)):
    """
    Create the crowd counting model with proper upsampling
    """
    # Use MobileNetV2 as base model
    base_model = tf.keras.applications.MobileNetV2(
        include_top=False,
        weights='imagenet',
        input_shape=input_shape
    )
    
    base_model.trainable = False
    
    inputs = layers.Input(shape=input_shape)
    x = base_model(inputs)
    
    # Add upsampling layers to restore spatial dimensions
    # MobileNetV2 reduces spatial dimensions by factor of 32
    # We'll use 5 upsampling layers (2^5 = 32)
    x = layers.Conv2DTranspose(256, (3, 3), strides=2, padding='same', activation='relu')(x)  # 5x5
    x = layers.Conv2DTranspose(128, (3, 3), strides=2, padding='same', activation='relu')(x)  # 10x10
    x = layers.Conv2DTranspose(64, (3, 3), strides=2, padding='same', activation='relu')(x)   # 20x20
    x = layers.Conv2DTranspose(32, (3, 3), strides=2, padding='same', activation='relu')(x)   # 40x40
    x = layers.Conv2DTranspose(16, (3, 3), strides=2, padding='same', activation='relu')(x)   # 80x80
    
    # Final convolution to get density map
    outputs = layers.Conv2D(1, (1, 1), padding='same', activation='relu')(x)
    
    # Add a resize layer to ensure correct output size
    outputs = tf.image.resize(outputs, input_shape[:2])
    
    model = Model(inputs=inputs, outputs=outputs)
    return model

def train_model():
    """
    Train the crowd counting model
    """
    base_path = 'C:/Users/Dell/.cache/kagglehub/datasets/fmena14/crowd-counting/versions/3'
    image_path = os.path.join(base_path, 'images.npy')
    labels_path = os.path.join(base_path, 'labels.npy')
    
    train_generator = MemoryEfficientDataGenerator(image_path, labels_path)
    
    print(f"Total images: {train_generator.total_images}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Number of batches per epoch: {len(train_generator)}")
    
    model = create_crowd_counting_model()
    model.summary()
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss='mse',
        metrics=['mae']
    )
    
    # Add memory cleanup callback
    class MemoryCleanupCallback(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            gc.collect()
            tf.keras.backend.clear_session()
    
    # Add model checkpoint callback
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        'best_model.h5',
        monitor='loss',
        save_best_only=True,
        mode='min',
        verbose=1
    )
    
    history = model.fit(
        train_generator,
        epochs=EPOCHS,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                monitor='loss',
                patience=5,
                restore_best_weights=True,
                min_delta=1e-6
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='loss',
                factor=0.5,
                patience=3,
                min_delta=1e-6
            ),
            MemoryCleanupCallback(),
            checkpoint_callback
        ]
    )
    
    return model, history

def predict_density_map(model, image):
    """
    Predict density map for a single image
    """
    # Preprocess image
    img_resized = cv2.resize(image, IMAGE_SIZE).astype(np.float32) / 255.0
    img_batch = np.expand_dims(img_resized, axis=0)
    
    # Predict
    density_map = model.predict(img_batch)[0]
    
    # Normalize density map for visualization
    density_map = (density_map - density_map.min()) / (density_map.max() - density_map.min() + 1e-8)
    
    return density_map

def calculate_crowd_count(density_map):
    """
    Calculate total count from density map
    """
    return np.sum(density_map)

def create_streamlit_app():
    """
    Create Streamlit interface for crowd counting
    """
    st.title("Crowd Counting Application")
    
    # Load model
    try:
        model = tf.keras.models.load_model('best_model.h5')
        st.success("Model loaded successfully!")
    except:
        st.error("No trained model found. Please train the model first.")
        return
    
    # File uploader
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # Read and display original image
        image = Image.open(uploaded_file)
        image = np.array(image)
        
        st.subheader("Original Image")
        st.image(image, use_column_width=True)
        
        # Predict density map
        density_map = predict_density_map(model, image)
        
        # Calculate count
        count = calculate_crowd_count(density_map)
        
        # Display results
        st.subheader(f"Estimated Count: {count:.0f} people")
        
        # Display density map
        st.subheader("Density Map")
        fig, ax = plt.subplots(figsize=(10, 10))
        im = ax.imshow(density_map, cmap='jet', vmin=0, vmax=1)
        plt.colorbar(im, ax=ax)
        st.pyplot(fig)
        plt.close(fig)

if __name__ == "__main__":
    # Check if model exists
    if not os.path.exists('best_model.h5'):
        print("Training model...")
        model, history = train_model()
        print("Training completed!")
    else:
        print("Model already exists. Skipping training.")
    
    # Run Streamlit app
    create_streamlit_app()