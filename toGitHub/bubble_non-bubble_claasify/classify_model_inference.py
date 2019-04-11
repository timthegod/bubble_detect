from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
import tensorflow as tf
import cv2
import os

epo = 33

model = tf.keras.models.load_model("bubble_3_128_1_0{}.model".format(epo))
CATEGORIES = ["bubble", "non_bubble"]
IMG_SIZE = 50  # 50 in txt-based

def prepare(filepath):
    img_array_new = cv2.imread(filepath, -1)  # read in the image, convert to grayscale
    new_array_new = cv2.resize(img_array_new, (IMG_SIZE, IMG_SIZE))  # resize image to match model's expected sizing
    new_array_new = new_array_new/255.0
    return new_array_new.reshape(-1, IMG_SIZE, IMG_SIZE, 1)  # return the image with shaping that TF wants.

def store_FP_image(filepath, catagory):
	img_array_new = cv2.imread(filepath, -1)
	new_array_new = cv2.resize(img_array_new, (IMG_SIZE, IMG_SIZE))
	cv2.imwrite(catagory, new_array_new)

bubble_eval_path = 'bubble_inference'
non_bubble_eval_path = 'non_bubble_inference'

bubble_eval_FP_path = 'bubble_inference/bubble_FP_{}'.format(epo)
non_bubble_eval_FP_path = 'non_bubble_inference/non_bubble_FP_{}'.format(epo)

if not os.path.isdir(bubble_eval_FP_path):
	os.mkdir(bubble_eval_FP_path)
if not os.path.isdir(non_bubble_eval_FP_path):
	os.mkdir(non_bubble_eval_FP_path)

print('BUBBLE')
bubble_total = 0
bubble_FP = 0
for file in os.listdir(bubble_eval_path):
	if ".jpg" in file:
		prediction = model.predict([prepare(os.path.join(bubble_eval_path, file))])  # REMEMBER YOU'RE PASSING A LIST OF THINGS YOU WISH TO PREDICT
		if prediction[0][0] > 0.5:
			bubble_FP += 1
			store_FP_image(os.path.join(bubble_eval_path, file), os.path.join(bubble_eval_FP_path, file))
		bubble_total += 1
	else:
		pass

print('NON')
non_bubble_total = 0
non_bubble_FP = 0
for file in os.listdir(non_bubble_eval_path):
	if ".jpg" in file:
		prediction = model.predict([prepare(os.path.join(non_bubble_eval_path, file))])  # REMEMBER YOU'RE PASSING A LIST OF THINGS YOU WISH TO PREDICT
		if prediction[0][0] <= 0.5:
			non_bubble_FP += 1
			store_FP_image(os.path.join(non_bubble_eval_path, file), os.path.join(non_bubble_eval_FP_path, file))
		non_bubble_total += 1
	else:
		pass

print('Bubble Accuracy: ', (bubble_total - bubble_FP)/bubble_total)
print('Non Bubble Accuracy: ', (non_bubble_total - non_bubble_FP)/non_bubble_total)

