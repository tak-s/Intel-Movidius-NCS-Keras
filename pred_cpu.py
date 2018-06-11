import argparse
import sys
import cv2
import numpy
import time
import glob, os


def pred(file_path, size):
	# Read an image from file
	img = cv2.imread(file_path)
	img = cv2.resize(img, (size, size))
	arr = numpy.array(img).reshape((size,size,3))
	arr = numpy.expand_dims(arr, axis=0)

	preds = model.predict(arr)
	return preds[0]


# parse argument
parser = argparse.ArgumentParser()
parser.add_argument("model_path", help="model file path", type=str)
parser.add_argument("weights_path", help="model weights file path", type=str)
parser.add_argument("img_dir", help="test image dir", type=str)
parser.add_argument("size", help="model input image size", type=int)
args = parser.parse_args()

# load model and weights
from keras.models import model_from_json
model = model_from_json(open(args.model_path).read())
model.load_weights(args.weights_path)

print("========= predict on CPU =====")
# setup image files
types = ('*.png', '*.jpg') # the tuple of file types
files_grabbed = []
for files in types:
     files_grabbed.extend(glob.glob(os.path.join(args.img_dir, files)))
files_grabbed = sorted(files_grabbed)

# start prediction
start = time.time()
for filename in files_grabbed:
	print("{0}: {1}".format(filename, pred(filename, args.size)))

elapsed_time = time.time() - start

print("elapsed_time:{0}".format(elapsed_time) + "[sec]")
print()

