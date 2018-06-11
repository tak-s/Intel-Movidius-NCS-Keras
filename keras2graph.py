# Arda Mavi

import argparse
import os
import sys
import tensorflow as tf
from keras import backend as K
from keras.models import model_from_json

'''
Referred to:
  https://gist.github.com/morgangiraud/249505f540a5e53a48b0c1a869d370bf#file-medium-tffreeze-1-py
'''
def freeze_graph(model_dir, output_node_names, tf_frozen_model_path):
    # We retrieve our checkpoint fullpath
    checkpoint = tf.train.get_checkpoint_state(model_dir)
    input_checkpoint = checkpoint.model_checkpoint_path
    
    # We precise the file fullname of our freezed graph
    output_graph = tf_frozen_model_path

    # We clear devices to allow TensorFlow to control on which device it will load operations
    clear_devices = True

    # We start a session using a temporary fresh Graph
    with tf.Session(graph=tf.Graph()) as sess:
        # We import the meta graph in the current default Graph
        saver = tf.train.import_meta_graph(input_checkpoint + '.meta', clear_devices=clear_devices)

        # We restore the weights
        saver.restore(sess, input_checkpoint)

        # We use a built-in TF helper to export variables to constants
        output_graph_def = tf.graph_util.convert_variables_to_constants(
            sess, # The session is used to retrieve the weights
            tf.get_default_graph().as_graph_def(), # The graph_def is used to retrieve the nodes 
            output_node_names.split(",") # The output node names are used to select the usefull nodes
        ) 

        # Finally we serialize and dump the output graph to the filesystem
        with tf.gfile.GFile(output_graph, "wb") as f:
            f.write(output_graph_def.SerializeToString())
        print("%d ops in the final graph." % len(output_graph_def.node))

    return output_graph_def


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

def keras_to_tf(tf_model_dir, tf_model_path):
    saver = tf.train.Saver()
    with K.get_session() as sess:
        saver.save(sess, tf_model_path)
        #tf.train.write_graph( sess.graph_def, tf_model_dir, "tf.pb", as_text=False )
    return True

def tf_to_graph(tf_frozen_model_path, model_in, model_out, graph_path, do_profile=False, shaves=12, verbose=False):
    if(do_profile==True):
        os.system('mvNCProfile {0} -in {1} -on {2} -s {3}'.format(
            tf_frozen_model_path, model_in, model_out, shaves))

    if(verbose==True):
        for i in tf.get_default_graph().get_operations():
            print(i.name)

        print('mvNCCompile {0} -in {1} -on {2} -o {3} -s {4}'.format(
            tf_frozen_model_path, model_in, model_out, graph_path, shaves))

    os.system('mvNCCompile {0} -in {1} -on {2} -o {3} -s {4}'.format(
         tf_frozen_model_path, model_in, model_out, graph_path, shaves))
    return True

def keras_to_graph(model_path, model_in, model_out, weights_path, graph_path, take_tf_files, do_profile, shaves, verbose):
    K.set_learning_phase(0)
    # Getting Keras Model:
    keras_model = get_keras_model(model_path, weights_path, verbose)

    # Saving TensorFlow Model from Keras Model:
    tf_model_dir = './TF_Model'
    tf_model_path = './TF_Model/tf_model'
    tf_frozen_model_path = './TF_Model/tf_model_frozen.pb'
    keras_to_tf(tf_model_dir, tf_model_path)

    freeze_graph(tf_model_dir, model_out, tf_frozen_model_path)

    tf_to_graph(tf_frozen_model_path, model_in, model_out, 
        graph_path, do_profile=do_profile, shaves=shaves, verbose=verbose)

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

    keras_to_graph(args.model_path, args.model_in, args.model_out, args.weights_path, 
        args.output_graph, args.take_tf_files, args.do_profile, args.shaves, args.verbose)

