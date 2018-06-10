# Arda Mavi

import argparse
import os
import sys
import tensorflow as tf
from keras import backend as K
from keras.models import model_from_json

def get_keras_model(model_path, weights_path, verbose=False):
    # Reading model file:
    with open(model_path, 'r') as model_file:
        model = model_file.read()

    # Readed model file to model:
    model = model_from_json(model)

    # Loading weights:
    model.load_weights(weights_path)

    if(verbose==True):
        print('Model Summary:')
        print(model.summary())

    return model

def keras_to_tf(tf_model_path):
    saver = tf.train.Saver()
    with K.get_session() as sess:
        saver.save(sess, tf_model_path)
    return True

def tf_to_graph(tf_model_path, model_in, model_out, graph_path, do_profile=False, shaves=12, verbose=False):
    if(do_profile==True):
        os.system('mvNCProfile {0}.meta -in {1} -on {2} -s {3}'.format(tf_model_path, model_in, model_out, shaves))

    if(verbose==True):
        for i in tf.get_default_graph().get_operations():
            print(i.name)

        print('mvNCCompile {0}.meta -in {1} -on {2} -o {3} -s {4}'.format(tf_model_path, model_in, model_out, graph_path, shaves))

    os.system('mvNCCompile {0}.meta -in {1} -on {2} -o {3} -s {4}'.format(tf_model_path, model_in, model_out, graph_path, shaves))
    return True

def keras_to_graph(model_path, model_in, model_out, weights_path, graph_path, take_tf_files, do_profile, shaves, verbose):
    K.set_learning_phase(0)
    # Getting Keras Model:
    keras_model = get_keras_model(model_path, weights_path, verbose)

    # Saving TensorFlow Model from Keras Model:
    tf_model_path = './TF_Model/tf_model'
    keras_to_tf(tf_model_path)

    tf_to_graph(tf_model_path, model_in, model_out, graph_path, do_profile=do_profile, shaves=shaves, verbose=verbose)

    if take_tf_files == False:
        os.system('rm -rf ./TF_Model')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("model_path", help="model file path", type=str)
    parser.add_argument("model_in", help="model input name", type=str)
    parser.add_argument("model_out", help="model out name", type=str)
    parser.add_argument("weights_path", help="model weights file path", type=str)
    parser.add_argument("--output_graph", "-o", help="output graph file path", type=str, default="./graph")
    parser.add_argument("--take_tf_files", help="keep tf files(TF_Model/*)", action="store_true")
    parser.add_argument("--shaves", "-s", help="max_number_of_shaves", type=int, default="12")
    parser.add_argument("--do_profile", help="exec mvNCProfile", action="store_true")
    parser.add_argument("--verbose", "-v", help="output detail", action="store_true")
    args = parser.parse_args()

    keras_to_graph(args.model_path, args.model_in, args.model_out, args.weights_path, args.output_graph, args.take_tf_files, args.do_profile, args.shaves, args.verbose)

