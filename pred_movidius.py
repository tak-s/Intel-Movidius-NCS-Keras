"""NCAPI v2"""
import sys
import argparse
from mvnc import mvncapi
import cv2
import numpy
import time, glob, os

# parse argument
parser = argparse.ArgumentParser()
parser.add_argument("graph_path", help="Movidius NCS graph file path", type=str)
parser.add_argument("img_dir", help="test image dir", type=str)
parser.add_argument("size", help="input image size", type=int)
args = parser.parse_args()

# Initialize and open a device
device_list = mvncapi.enumerate_devices()
device = mvncapi.Device(device_list[0])
device.open()

# Initialize a graph from file at some GRAPH_FILEPATH
GRAPH_FILEPATH = args.graph_path
with open(GRAPH_FILEPATH, mode='rb') as f:
	graph_buffer = f.read()
graph = mvncapi.Graph('graph1')

# CONVENIENCE FUNCTION: 
# Allocate the graph to the device and create input/output Fifos with default options in one call
input_fifo, output_fifo = graph.allocate_with_fifos(device, graph_buffer)


def pred(file_path):
	# Read an image from file
	tensor = cv2.imread(file_path)
	tensor = cv2.resize(tensor, (args.size, args.size))
	# Convert an input tensor to 32FP data type
	input_tensor = tensor.astype(numpy.float32)


	# CONVENIENCE FUNCTION: 
	# Write the image to the input queue and queue the inference in one call
	graph.queue_inference_with_fifo_elem(input_fifo, output_fifo, input_tensor, None)

	# Get the results from the output queue
	output, user_obj = output_fifo.read_elem()
	return output


print("========= predict on Movidius =====")
# setup image files
types = ('*.png', '*.jpg') # the tuple of file types
files_grabbed = []
for files in types:
     files_grabbed.extend(glob.glob(files))
files_grabbed = sorted(files_grabbed)

# start prediction
start = time.time()
for filename in files_grabbed:
	print("{0}: {1}".format(filename, pred(filename)))
elapsed_time = time.time() - start
print ("elapsed_time:{0}".format(elapsed_time) + "[sec]")


# Clean up
input_fifo.destroy()
output_fifo.destroy()
graph.destroy()
device.close()
device.destroy()



