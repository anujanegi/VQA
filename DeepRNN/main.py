#!/usr/bin/python
import tensorflow as tf
from config import Config
from model import CaptionGenerator
from dataset import prepare_test_data

flags = tf.app.flags.FLAGS

tf.flags.DEFINE_string('test_image', 'image.jpg', 'Test image name')

def main(argv):
    config = Config()
    config.test_file_name = flags.test_image
    config.phase = 'test'
    config.beam_size = 3

    with tf.Session() as sess:
        data, vocabulary = prepare_test_data(config)
        model = CaptionGenerator(config)
        model.load(sess, './data/289999.npy')
        tf.get_default_graph().finalize()
        model.test(sess, data, vocabulary)

if __name__ == '__main__':
    tf.app.run()
