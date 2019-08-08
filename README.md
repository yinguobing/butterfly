# butterfly
A lightweight python module to load TensorFlow frozen model (a single pb file).

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites

TensorFlow

```
# For CPU
python3 -m pip install tensorflow

# or, if you have a CUDA compatible GPU
python3 -m pip install tensorflow-gpu
```

OpenCV, only if you want to run the demo with video/camera.

```
python3 -m pip install opencv-python
```

### Installing

From your favorite development directory:

```
git clone https://github.com/yinguobing/butterfly.git
```

## Running the tests

A demonstration of how to use this module through the `demo.py` script. Before we start make sure you have your frozen model named `frozen_graph.pb` ready.

### List all ops in graph

Most of the time you should know the input and output nodes' name. In case you don't, we can output all the node names in the graph.

```
python3 demo.py --model frozen_model.pb --list_ops true
```

and the output looks like:
```
['butterfly/image_tensor', 'butterfly/map/Shape', 'butterfly/map/strided_slice/stack', ... , 'butterfly/model/final_dense', 'butterfly/ArgMax/dimension', 'butterfly/ArgMax', 'butterfly/softmax_tensor']
```

Here the input node is `butterfly/image_tensor` and the output node is `butterfly/ArgMax`. We will use these later.

### Inference with image.

Use the `--image` argument.

```
python3 demo.py \
    --model frozen_model.pb \
    --input_node butterfly/image_tensor \
    --output_node butterfly/ArgMax \
    --image image.jpg
```

### Inference with video/camera.

For video file, use the `--video` argument.
```
python3 demo.py \
    --model frozen_model.pb \
    --input_node butterfly/image_tensor \
    --output_node butterfly/ArgMax \
    --video video.mp4
```

For webcam, use the `--cam` argument.
```
python3 demo.py \
    --model frozen_model.pb \
    --input_node butterfly/image_tensor \
    --output_node butterfly/ArgMax \
    --cam 0
```

## Deployment

`butterfly.py` is sufficient for your development.


## Authors

* yinguobing (尹国冰) - [yinguobing](https:yinguobing.com)


## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details