import tkinter.filedialog
import tkinter as tk
import tensorflow as tf
import edsr
import numpy as np
import matplotlib.pyplot as plt

def load_generator():
    return edsr.create_edsr()

def load_image():
    file = tk.filedialog.askopenfilename()

    if file != None:
        image = tf.io.read_file(file)
        image = tf.io.decode_image(image)
        return image

def process_demo(image):
    image = image.numpy()
    image = (np.expand_dims(image, 0))

    image = tf.cast(image, tf.float32)
    image = image / 255
    pred = generator.predict(image)
    #pred = tf.nn.relu(pred)
    #pred = pred * 255
    #pred = tf.cast(pred, tf.uint16)
    #pred = tf.reshape(pred, (1, 960, 1280, 3))

    pred = tf.cast(pred, tf.int32)
    plt.figure(figsize = (48, 48))
    plt.subplot(1, 2, 1)
    plt.imshow(pred[0], vmin = 0, vmax = 255)
    plt.savefig("Demo.png")

    return pred

def upload_sample():
    image = load_image()
    sample = process_demo(image)
    #file = tf.image.encode_png(sample[0])
    #tf.io.write_file("demo.png", file)

# Create window
main_window = tk.Tk()
main_window.title("CI301 - Presentation")
generator = load_generator()

print(generator)

button = tk.Button(main_window, text='Open File', width=48, command=upload_sample) 
button.pack() 

# Tkinter loop
main_window.mainloop()