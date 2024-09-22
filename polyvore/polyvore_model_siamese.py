"""Siamese Network for compatibility modeling."""

from __future__ import absolute_import, division, print_function

import numpy as np
import tensorflow as tf
import scipy.io as sio
from scipy.linalg import block_diag

from ops import image_embedding
from ops import image_processing
from ops import inputs as input_ops


class PolyvoreModel(object):
    """Model for fashion set on Polyvore dataset."""

    def __init__(self, config, mode, train_inception=False):
        """Basic setup.

        Args:
            config: Object containing configuration parameters.
            mode: "train", "eval" or "inference".
            train_inception: Whether the inception submodel variables are trainable.
        """
        assert mode in ["train", "eval", "inference"]
        self.config = config
        self.mode = mode
        self.train_inception = train_inception

        # Reader for the input data.
        self.reader = tf.TFRecordReader()

        # Initialize all variables with a random uniform initializer.
        self.initializer = tf.random_uniform_initializer(
            minval=-self.config.initializer_scale,
            maxval=self.config.initializer_scale
        )

        # Model attributes
        self.images = None
        self.input_mask = None
        self.image_embeddings = None
        self.total_loss = None
        self.inception_variables = []
        self.init_fn = None
        self.global_step = None

    def is_training(self):
        """Returns true if the model is built for training mode."""
        return self.mode == "train"

    def process_image(self, encoded_image, thread_id=0, image_idx=0):
        """Decodes and processes an image string.

        Args:
            encoded_image: A scalar string Tensor; the encoded image.
            thread_id: Preprocessing thread id used to select the ordering of color distortions.
            image_idx: Index of the image in an outfit. Only used for summaries.
        Returns:
            A float32 Tensor of shape [height, width, 3]; the processed image.
        """
        return image_processing.process_image(
            encoded_image,
            is_training=self.is_training(),
            height=self.config.image_height,
            width=self.config.image_width,
            image_format=self.config.image_format,
            image_idx=image_idx
        )

    def build_inputs(self):
        """Input prefetching, preprocessing and batching."""
        if self.mode == "inference":
            # In inference mode, images and inputs are fed via placeholders.
            image_feed = tf.placeholder(dtype=tf.string, shape=[], name="image_feed")
            image_feed = self.process_image(image_feed)
            image_seqs = tf.expand_dims(image_feed, 0)
            input_mask = tf.placeholder(dtype=tf.int64, shape=[1, 8], name="input_mask")
        else:
            # Prefetch serialized SequenceExample protos.
            input_queue = input_ops.prefetch_input_data(
                self.reader,
                self.config.input_file_pattern,
                is_training=self.is_training(),
                batch_size=self.config.batch_size,
                values_per_shard=self.config.values_per_input_shard,
                input_queue_capacity_factor=self.config.input_queue_capacity_factor,
                num_reader_threads=self.config.num_input_reader_threads
            )

            images_and_captions = []
            for thread_id in range(self.config.num_preprocess_threads):
                serialized_sequence_example = input_queue.dequeue()
                set_id, encoded_images, image_ids, captions, likes = (
                    input_ops.parse_sequence_example(
                        serialized_sequence_example,
                        set_id=self.config.set_id_name,
                        image_feature=self.config.image_feature_name,
                        image_index=self.config.image_index_name,
                        caption_feature=self.config.caption_feature_name,
                        number_set_images=self.config.number_set_images
                    )
                )

                images = [self.process_image(encoded_images[i], image_idx=i) for i in range(self.config.number_set_images)]
                images_and_captions.append([set_id, images, image_ids, captions, likes])

            queue_capacity = 5 * self.config.num_preprocess_threads * self.config.batch_size
            set_ids, image_seqs, image_ids, input_mask, loss_mask, cap_seqs, cap_mask, likes = (
                input_ops.batch_with_dynamic_pad(
                    images_and_captions,
                    batch_size=self.config.batch_size,
                    queue_capacity=queue_capacity
                )
            )

        self.images = image_seqs
        self.input_mask = input_mask

    def build_image_embeddings(self):
        """Builds the image model subgraph and generates image embeddings."""
        images = tf.reshape(self.images, [-1, self.config.image_height, self.config.image_height, 3])
        inception_output = image_embedding.inception_v3(
            images,
            trainable=self.train_inception,
            is_training=self.is_training()
        )
        self.inception_variables = tf.get_collection(tf.GraphKeys.VARIABLES, scope="InceptionV3")

        with tf.variable_scope("image_embedding") as scope:
            image_embeddings = tf.contrib.layers.fully_connected(
                inputs=inception_output,
                num_outputs=self.config.embedding_size,
                activation_fn=None,
                weights_initializer=self.initializer,
                biases_initializer=None,
                scope=scope
            )

        tf.constant(self.config.embedding_size, name="embedding_size")
        self.image_embeddings = tf.reshape(image_embeddings, [tf.shape(self.images)[0], -1, self.config.embedding_size])

    def build_model(self):
        """Builds the model."""
        norm_image_embeddings = tf.nn.l2_normalize(self.image_embeddings, 2, name="norm_image_embeddings")

        if self.mode != "inference":
            emb_loss_mask = np.ones((self.config.number_set_images, self.config.number_set_images))
            emb_loss_mask = block_diag(*[emb_loss_mask] * 10)

            norm_image_embeddings = tf.reshape(norm_image_embeddings, [self.config.number_set_images * self.config.batch_size, self.config.embedding_size])
            scores = tf.matmul(norm_image_embeddings, norm_image_embeddings, transpose_a=False, transpose_b=True, name="scores")

            posi_scores = tf.reduce_sum(tf.multiply(scores, emb_loss_mask)) / np.sum(emb_loss_mask)

            emb_loss_mask = 1.0 - emb_loss_mask
            m = 0.8  # margin in Siamese network
            nega_scores = tf.maximum(tf.multiply(scores, emb_loss_mask) - m, 0.0)
            nega_scores = tf.reduce_sum(nega_scores) / np.sum(emb_loss_mask)

            emb_batch_loss = tf.subtract(nega_scores, posi_scores, name="emb_batch_loss")
            tf.contrib.losses.add_loss(emb_batch_loss)

            total_loss = tf.contrib.losses.get_total_loss()

            tf.scalar_summary("emb_batch_loss", emb_batch_loss)
            tf.scalar_summary("total_loss", total_loss)
            for var in tf.trainable_variables():
                tf.histogram_summary(var.op.name, var)

            self.total_loss = total_loss

    def setup_inception_initializer(self):
        """Sets up the function to restore inception variables from checkpoint."""
        if self.mode != "inference":
            saver = tf.train.Saver(self.inception_variables)

            def restore_fn(sess):
                tf.logging.info("Restoring Inception variables from checkpoint file %s", self.config.inception_checkpoint_file)
                saver.restore(sess, self.config.inception_checkpoint_file)

            self.init_fn = restore_fn

    def setup_global_step(self):
        """Sets up the global step Tensor."""
        global_step = tf.Variable(
            initial_value=0,
            name="global_step",
            trainable=False,
            collections=[tf.GraphKeys.GLOBAL_STEP, tf.GraphKeys.VARIABLES]
        )

        self.global_step = global_step

    def build(self):
        """Creates all ops for training and evaluation."""
        self.build_inputs()
        self.build_image_embeddings()
        self.build_model()
        self.setup_inception_initializer()
        self.setup_global_step()
