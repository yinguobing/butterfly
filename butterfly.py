"""A lightwight module to load TensorFlow frozen model"""
import tensorflow as tf


class Butterfly(object):

    def __init__(self, model_filename, input_node_name, output_node_names):
        """
        Args:
            model_filename: the frozen model's filename.
            input_node_name: the name of the input node.
            output_node_names: the name of the output nodes as a list.
        """

        # load the frozen model file and parse it to retrieve the GraphDef.
        with tf.gfile.GFile(model_filename, "rb") as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())

        # Import the graph_def into a new Graph and create a session.
        self.graph = tf.Graph()
        with self.graph.as_default():
            tf.import_graph_def(graph_def, name='butterfly')
            self.sess = tf.Session(graph=self.graph)

        self.input_node_name = input_node_name
        self.output_node_names = output_node_names.split(',')

    def list_ops(self):
        """Output the op names in the graph."""
        names = [op.name for op in self.graph.get_operations()]
        return names

    def run(self, inputs):
        """Run inference and return the results.
        Args:
            inputs: inputs as a list.

        Returns:
            inference results.
        """
        input_node = self.graph.get_tensor_by_name(self.input_node_name + ":0")
        output_nodes = [self.graph.get_tensor_by_name(
            name + ":0") for name in self.output_node_names]
        return self.sess.run(output_nodes, feed_dict={input_node: inputs})
