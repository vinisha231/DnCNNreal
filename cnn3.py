import tensorflow as tf
from tensorflow.keras import layers, models, Input
from tensorflow.keras.layers import Subtract
import numpy as np
import matplotlib.pyplot as plt
import os
from glob import glob
import struct

# Function to define the DnCNN model
def DnCNN(depth=17, filters=64, image_channels=1, use_bnorm=True):
    input_layer = Input(shape=(None, None, image_channels), name='input')
    x = layers.Conv2D(filters=filters, kernel_size=3, padding='same', activation='relu')(input_layer)

    for i in range(depth - 2):
        x = layers.Conv2D(filters=filters, kernel_size=3, padding='same')(x)
        if use_bnorm:
            x = layers.BatchNormalization(axis=3)(x)
        x = layers.Activation('relu')(x)

    output_layer = layers.Conv2D(filters=image_channels, kernel_size=3, padding='same')(x)
    output_layer = Subtract()([input_layer, output_layer])
    model = models.Model(inputs=input_layer, outputs=output_layer, name='DnCNN')
    return model

# Function to load .flt files
def load_flt_file(file_path, shape):
    with open(file_path, 'rb') as f:
        data = f.read()
    image = np.array(struct.unpack('f' * (len(data) // 4), data)).reshape(shape)
    return image

# Function to load and resize real image data
def load_real_data(image_dir, shape):
    image_paths = glob(os.path.join(image_dir, '*'))
    images = []
    for path in image_paths:
        if path.endswith('.flt'):
            image = load_flt_file(path, shape)
       # elif path.endswith('.png'):
        #    image = tf.keras.preprocessing.image.load_img(path, color_mode='grayscale')
         #   image = tf.keras.preprocessing.image.img_to_array(image).astype('float32') / 255.0
          #  image = np.squeeze(image)  # Remove single channel dimension if exists
        else:
            continue
        image = np.expand_dims(image, axis=-1)  # Add channel dimension
        image = tf.image.resize(image, shape)
        images.append(image)
    images = np.array(images)
    noise = np.random.normal(loc=0.0, scale=0.1, size=images.shape).astype('float32')
    noisy_images = np.clip(images + noise, 0., 1.)
    return noisy_images, images

# Function to add noise to an image
def add_noise(image, noise_factor=0.1):
    noisy_image = image + noise_factor * np.random.normal(loc=0.0, scale=0.1, size=image.shape)
    noisy_image = np.clip(noisy_image, 0., 1.)
    return noisy_image

# Load and preprocess a grayscale image from .flt or .png file
def preprocess_image(image_path, shape):
    if image_path.endswith('.flt'):
        image = load_flt_file(image_path, shape)
   # elif image_path.endswith('.png'):
   #     image = tf.keras.preprocessing.image.load_img(image_path, color_mode='grayscale')
    #    image = tf.keras.preprocessing.image.img_to_array(image).astype('float32') / 255.0
     #   image = np.squeeze(image)  # Remove single channel dimension if exists
    else:
	raise ValueError(f"Unsupported file format: {image_path}")
    image = np.expand_dims(image, axis=-1)  # Add channel dimension
    image = tf.image.resize(image, shape)
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

# Function to save .flt files
def save_flt_file(file_path, data):
    with open(file_path, 'wb') as f:
        f.write(struct.pack('f' * data.size, *data.flatten()))

# Function to save .png files
def save_png_file(file_path, data):
    tf.keras.preprocessing.image.save_img(file_path, data, scale=False)

# Function to display images
def display_images(original, noisy, denoised):
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.title('Original Image')
    plt.imshow(original[0, :, :, 0], cmap='gray')  # Ensure to plot back in the [0, 1] range for grayscale images
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.title('Noisy Image')
    plt.imshow(noisy[0, :, :, 0], cmap='gray')  # Ensure to plot back in the [0, 1] range for grayscale images
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.title('Denoised Image')
    plt.imshow(denoised[0, :, :, 0], cmap='gray')  # Ensure to plot back in the [0, 1] range for grayscale images
    plt.axis('off')

    plt.show()

# Define your image directory and size
image_dir = '/mmfs1/gscratch/uwb/CT_images/RECONS2024/60views'
output_dir = '/mmfs1/gscratch/uwb/vdhaya/output'
os.makedirs(output_dir, exist_ok=True)
image_shape = (512, 512)  # Define the shape of the .flt images

# Load real data
X_train, y_train = load_real_data(image_dir, image_shape)

# Check the shapes of the loaded data
print(f'X_train shape: {X_train.shape}')
print(f'y_train shape: {y_train.shape}')

# Create the model for grayscale images
model = DnCNN(depth=17, filters=64, image_channels=1, use_bnorm=True)
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=250, batch_size=32, verbose=1)

# Save the model architecture and weights
model.save('dncnn_model.h5')
model.save_weights('dncnn_model.weights.h5')

# Print the model summary
model.summary()

# Get list of image paths
image_paths = glob(os.path.join(image_dir, '*'))

# Process each image in the directory
for i, image_path in enumerate(image_paths):
    if not (image_path.endswith('.flt') or image_path.endswith('.png')):
        continue

    print(f'Processing image: {image_path}')

    # Preprocess and add noise to the test image
    image = preprocess_image(image_path, image_shape)
    noisy_image = add_noise(image)

    # Denoise the grayscale image using the DnCNN model
    predicted_noise = model.predict(noisy_image)
    denoised_image = Subtract()([noisy_image, predicted_noise])
    # Save the denoised image to the specified folder
    if image_path.endswith('.flt'):
        output_image_path = os.path.join(output_dir, f'denoised_image_{i + 1}.flt')
        save_flt_file(output_image_path, denoised_image[0, :, :, 0])
    elif image_path.endswith('.png'):
        output_image_path = os.path.join(output_dir, f'denoised_image_{i + 1}.png')
        save_png_file(output_image_path, denoised_image[0])

    print(f"Denoised image saved to: {output_image_path}")

    # Optionally display the images
    display_images(image, noisy_image, denoised_image)
