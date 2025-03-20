# This script includes the function calls for the report notebook.
# imports

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image


def create_occlusion(image, occlusion_percentage):
    """
    Create a square black occlusion based on image size and occlusion percentage
    """
    occluded_image = image.copy()
    height, width = image.shape[:2]

    area = height * width
    occlusion_area = (occlusion_percentage / 100) * area
    occlusion_size = int(np.ceil(np.sqrt(occlusion_area)))
    # The above line of code was created with ChatGPT at 2/12/25 10:01pm

    x = width // 2 - occlusion_size // 2
    y = height // 2 - occlusion_size // 2

    # Make empty mask of image size
    mask = np.zeros((height, width), dtype=np.float32)
    # Downsize mask for occlusion size
    mask[y : y + occlusion_size, x : x + occlusion_size] = 1
    # The above line of code was created with ChatGPT at 2/12/25 10:03pm

    for c in range(3):
        occluded_image[..., c] = occluded_image[..., c] * (1 - mask) + 0 * mask
    # The above line of code was created with ChatGPT at 2/12/25 10:07pm

    return occluded_image


def generate_saliency_map(model, img):
    """
    This function creates the saliency map.
    This was taken from Dr.Bent's Saliency Map Notebook example. I changed the image processing from ResNet50 to Inception_V3.
    Citation: (https://github.com/AIPI-590-XAI/Duke-AI-XAI/blob/main/explainable-ml-example-notebooks/saliency_maps.ipynb)
    """

    # Convert the input image to a TensorFlow variable
    x = tf.Variable(img)

    # Add an extra dimension to the image tensor to match the model's input shape
    x = tf.expand_dims(x, axis=0)

    # Preprocess the image according to Inception V3 requirements
    x = tf.keras.applications.inception_v3.preprocess_input(x)

    # Create a gradient tape context to record operations for automatic differentiation
    with tf.GradientTape() as tape:
        # Watch the input tensor to calculate gradients
        tape.watch(x)

        # Forward pass: get model predictions for the input image
        preds = model(x)

        # Find the index of the highest predicted class probability
        top_pred_index = tf.argmax(preds[0])

    # Calculate the gradients of the top prediction with respect to the input image
    grads = tape.gradient(preds, x)

    # Compute the saliency map by taking the maximum absolute gradient across color channels
    saliency = tf.reduce_max(tf.abs(grads), axis=-1)[0]

    # Return the saliency map and the index of the top predicted class as a numpy array
    return saliency, top_pred_index.numpy()


def visualize_occlusion_effect(
    img_path,
    model,
    occlusion_percentages,
    create_occlusion,
    generate_saliency_map,
    class_idx,
):
    """
    This function creates the 2x3 occlusion and saliencny map images side-by-side.
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    for i, occlusion in enumerate(occlusion_percentages):

        img = Image.open(img_path)
        img = img.resize((299, 299))  # resize for inception v3
        img = np.array(img).astype(np.float32)

        occluded_img = create_occlusion(img, occlusion_percentage=occlusion)

        # Generate saliency map
        # These line of code were taken from Dr.Bent's saliency map notebook
        saliency_map, top_pred_index = generate_saliency_map(model, occluded_img)
        predicted_class = class_idx[str(top_pred_index)][1]

        # plot on subplot
        axes[0, i].imshow(occluded_img.astype(np.uint8))
        axes[0, i].set_title(f"Occlusion: {occlusion}%")
        axes[0, i].axis("off")

        axes[1, i].imshow(saliency_map, cmap="viridis")
        axes[1, i].set_title(f"Prediction: {predicted_class}")
        axes[1, i].axis("off")

    plt.tight_layout()
    plt.show()
