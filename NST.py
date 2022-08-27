import math
import tensorflow as tf
from PIL import Image
import numpy as np
from   tqdm  import tqdm
import matplotlib.pyplot as plt
from tensorflow import keras
from keras import Model
import keras
from tensorflow.keras.applications import vgg19
from tensorflow.keras.optimizers import SGD
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.python.keras.models import Model
from tensorflow.keras.utils import plot_model
from tensorflow.keras.models import load_model
from datetime import datetime
from tensorflow.keras import optimizers
from tensorflow.keras.optimizers import schedules
import cv2

Image.MAX_IMAGE_PIXELS = None

style_image_path = 'images/internal_colored.jpg'
base_image_path = 'images/internal_white.jpg'

#defining the model
model = vgg19.VGG19(
    include_top=False,
    weights='imagenet',
)
# set training to False
model.trainable = False

model.compile(optimizer='adam', metrics=['accuracy'])

#Print details of any sigle layer
model.summary()

#Function useful for our AI

#1) display image if we want to show the result
def display_image(image):
    # remove one dimension if image has 4 dimension
    if len(image.shape) == 4:
        img = np.squeeze(image, axis=0)
 
    img = deprocess_image(img)
 
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(img)
    return

#2) Function to open, resize and format pictures into appropriate tensors
def preprocess_image(image_path):
    img = tf.keras.utils.load_img(
        image_path, target_size=(img_nrows, img_ncols)
    )
    img = tf.keras.utils.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = vgg19.preprocess_input(img)
    return tf.convert_to_tensor(img)

#3) Conversion into an array
def deprocess_image(x):

    x = x.reshape((img_nrows, img_ncols, 3))
   
    x[:, :, 0] += 103.939
    x[:, :, 1] += 116.779
    x[:, :, 2] += 123.68

    # Conversion from BGR to RGB.
    x = x[:, :, ::-1]

    x = np.clip(x, 0, 255).astype("uint8")

    return x

#dimensions of the generated picture
width, height = keras.utils.load_img(base_image_path).size
img_nrows = 256
img_ncols = int(width * img_nrows / height)

content_layer = "block5_conv2"
content_model = Model(
    inputs=model.input,
    outputs=model.get_layer(content_layer).output
)
content_model.summary()

style_layers = [
    "block1_conv1",
    "block3_conv1",
    "block5_conv1",
]
style_models = [Model(inputs=model.input,
                      outputs=model.get_layer(layer).output) for layer in style_layers]


#4) Loss of content
def content_loss(content, generated):
    return tf.reduce_sum(tf.square(generated - content))

#5)
def gram_matrix(A):
    channels = int(A.shape[-1])
    a = tf.reshape(A, [-1, channels])
    n = tf.shape(a)[0]
    gram = tf.matmul(a, a, transpose_a=True)
    return gram / tf.cast(n, tf.float32)


weight_of_layer = 1. / len(style_models)
model.save_weights('weight_of_layer.h5')

#6)
def style_cost(style, generated):
    J_style = 0
 
    for style_model in style_models:
        a_S = style_model(style)
        a_G = style_model(generated)
        GS = gram_matrix(a_S)
        GG = gram_matrix(a_G)
        current_cost = tf.reduce_mean(tf.square(GS - GG))
        J_style += current_cost * weight_of_layer
 
    return J_style

generated_images = []

#Parameters and variables 
epochs = 150
num_epochs=10
epochs_loss_avg = tf.keras.metrics.Mean()
epochs_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()
best_image = None
losses= np.zeros(epochs+1)
accuracy= np.zeros(epochs)
epoch = list(range(len(losses)))

content = preprocess_image(base_image_path) 
style = preprocess_image(style_image_path)
generated = tf.Variable(content, dtype=tf.float32)

#8) Saving model and images
def result_saver(iteration):
  # Create name
  now = datetime.now()
  now = now.strftime("%Y%m%d_%H%M%S")
  model_name = 'best_model'
  image_name = 'images/'+ str(iteration)+'generated' +'.jpg'

  # Save image
  img = deprocess_image(generated.numpy())
  tf.keras.utils.save_img(image_name, img)
  #model.save_weights('./best_model.t7')
  model.save('Training/best_model.h5')

#9) Define the training of our model
def training_loop(base_image_path, style_image_path, a=10, b=1000):
    for n in range(epochs+1):
        model.load_weights('weight_of_layer.h5')
    # load content and style images from their respective path

    #optimization
        opt = tf.keras.optimizers.Adam(learning_rate=2)

        best_cost = math.inf
        best_image = None
        for i in tqdm(range(num_epochs), desc="Epoch {:03d}".format(n)):
            with tf.GradientTape() as tape:

                J_content = content_loss(content, generated)
                J_style = style_cost(style, generated)
                J_total = a * J_content + b * J_style

            grads = tape.gradient(J_total, generated)
            opt.apply_gradients([(grads, generated)])

            if J_total < best_cost:
                best_cost = J_total
                best_image = generated.numpy()

            generated_images.append(generated.numpy())
            epochs_loss_avg.update_state(J_total)
        losses[n]= epochs_loss_avg.result().numpy()
        
        if n == 30:
            result_saver(n)
        if n == 80:
            result_saver(n)
        if n == epochs:
            result_saver(n)
            
        
        model.save_weights('weight_of_layer.h5')


    
    return best_image


# Train the model and get best image
final_img = training_loop(base_image_path, style_image_path)

#Plotting result
plt.plot(epoch,losses)
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.savefig('Total loss.png')
plt.show()
plt.plot(epoch,accuracy, 'm')
plt.show()
newmodel= load_model('Training/best_model.h5')
