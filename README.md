# keras2graph
Convert Keras model to Intel Movidius NCS graph file, using NCSDK 2.x.

### Intel Movidius Neural Compute Stick
#### About Intel Movidius Neural Compute Stick:
Official Web Page: [developer.movidius.com](https://developer.movidius.com)  
Intel® Movidius™ Neural Compute SDK(NCSDK 2.x): [github](https://github.com/movidius/ncsdk/tree/ncsdk2)

## Keras Model to NCS Graph:
`keras2graph` process turn used TensorFlow backend Keras models to NCS graph.
Converts Keras model and weights to Intel Movidius technology internal compiled format.

Keras Model -> TensorFlow Session -> TensorFlow Meta File -> NCS Graph

Important: You need to install NCSDK 2.x.

## Command usage:
    keras2graph.py [-h] [--output_graph OUTPUT_GRAPH] [--take_tf_files]  
                   [--shaves SHAVES]  
                   [--do_profile] [--verbose]  
                   model_path model_in model_out weights_path  

This command creates and saves graph file as `<OUTPUT_GRAPH>`, default is ``'./graph'``.

### Arguments:
- `model_path`: Model location. Json file. Data Type: String
- `model_in`: Name of model's input layer. Data Type: String
- `model_out`: Name of model's output layer. Data Type: String
- `weights_path`: Model weights files location. Data Type: String
- `output_graph`: Location of output Intel Movidius technology internal compiled format graph. Data Type: String
- `take_tf_files`: If you don't want to delete created TensorFlow model files(meta file inside), select `False`. Data Type: Bool
- `shaves`: The number of available SHAVEs depends on your neural compute device. Data Type: Int
- `do_profile`: Run mvNCProfile before mvNCCompile. Data Type: Bool
- `verbose`: Show model details. Data Type: Bool

### Example Run Command:
    python3 keras2graph.py keras_model/model_CNN_1.json conv2d_1_input activation_5/Softmax keras_model/weights-best_CNN_1.h5 -o keras_model/CNN.graph --shaves=12 --do_profile

## Check result:
Use `pred_cpu.py` for prediction on CPU, and use `pred_movidius.py` for prediction on Intel Movidius.

for Intel Movidius:

    python3 pred_movidius.py keras_model/CNN.graph ./img 64


for CPU:

    python3 pred_cpu.py keras_model/model_CNN_1.json keras_model/weights-best_CNN_1.h5 ./img 64


## Important Notes:
- Install necessary modules with `sudo pip3 install -r requirements.txt` command.
