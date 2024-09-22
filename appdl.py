

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import json
import os
import random
import sys
import threading

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import Binarizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import tensorflow as tf
from absl import app, flags
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, mean_absolute_error, mean_squared_error

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, LSTM, Embedding, Concatenate
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.image import load_img

from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, mean_absolute_error, mean_squared_error
import logging
import requests
from io import BytesIO
from PIL import Image

logging.basicConfig(level=logging.INFO)

# Disable eager execution
tf.compat.v1.disable_eager_execution()

FLAGS = flags.FLAGS


flags.DEFINE_string('train_label', 'data/label/train_no_dup.json', 'Training label file')
flags.DEFINE_string('test_label', 'data/label/test_no_dup.json', 'Testing label file')
flags.DEFINE_string('valid_label','data/label/valid_no_dup.json', 'Validation label file')
flags.DEFINE_string('output_directory', 'data/tf_records/', 'Output data directory')
flags.DEFINE_string('image_dir', 'data/images/', 'Directory of image patches')
flags.DEFINE_string('word_dict_file', 'data/final_word_dict.txt', 'File containing the word dictionary.')

flags.DEFINE_integer('train_shards', 128, 'Number of shards in training TFRecord files.')
flags.DEFINE_integer('test_shards', 16, 'Number of shards in test TFRecord files.')
flags.DEFINE_integer('valid_shards', 8, 'Number of shards in validation TFRecord files.')
flags.DEFINE_integer('num_threads', 8, 'Number of threads to preprocess the images.')

class Vocabulary(object):
    """Simple vocabulary wrapper."""

    def __init__(self, vocab, unk_id):
        """Initializes the vocabulary.
        Args:
          vocab: A dictionary of word to word_id.
          unk_id: Id of the special 'unknown' word.
        """
        self._vocab = vocab
        self._unk_id = unk_id

    def word_to_id(self, word):
        """Returns the integer id of a word string."""
        if word in self._vocab:
            return self._vocab[word]
        else:
            print('unknown: ' + word)
            return self._unk_id

def _is_png(filename):
    """Determine if a file contains a PNG format image.
    Args:
      filename: string, path of the image file.
    Returns:
      boolean indicating if the image is a PNG.
    """
    return '.png' in filename

def _int64_feature(value):
    """Wrapper for inserting int64 features into Example proto."""
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def _float_feature(value):
    """Wrapper for inserting float features into Example proto."""
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def _bytes_feature(value):
    """Wrapper for inserting bytes features into Example proto."""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[tf.compat.v1.compat.as_bytes(value)]))

def _int64_feature_list(values):
    """Wrapper for inserting an int64 FeatureList into a SequenceExample proto."""
    return tf.train.FeatureList(feature=[_int64_feature(v) for v in values])

def _int64_list_feature_list(values):
    """Wrapper for inserting an int64 list FeatureList into a SequenceExample proto."""
    return tf.train.FeatureList(feature=[_int64_feature(v) for v in values])

def _bytes_feature_list(values):
    """Wrapper for inserting a bytes FeatureList into a SequenceExample proto."""
    return tf.train.FeatureList(feature=[_bytes_feature(v) for v in values])

def _float_feature_list(values):
    """Wrapper for inserting a float FeatureList into a SequenceExample proto."""
    return tf.train.FeatureList(feature=[_float_feature(v) for v in values])

def _to_sequence_example(set_info, decoder, vocab):
    """Builds a SequenceExample proto for an outfit."""
    set_id = set_info['set_id']
    image_data = []
    image_ids = []
    caption_data = []
    caption_ids = []
    # Prepare the data
    images = []
    texts = []
    likes = []

    for image_info in set_info['items']:
        filename = os.path.join(FLAGS.image_dir, set_id, str(image_info['index']) + '.jpg')
        if not os.path.exists(filename):
            logging.warning("File not found: %s", filename)
            continue

        with open(filename, "rb") as f:
            encoded_image = f.read()

        try:
            decoded_image = decoder.decode_jpeg(encoded_image)
        except (tf.errors.InvalidArgumentError, AssertionError):
            logging.warning("Skipping file with invalid JPEG data: %s", filename)
            continue
        
        # Prepare the data
        image = preprocess_image(encoded_image, target_size=(IMG_HEIGHT, IMG_WIDTH))
        images.append(image)
        texts.append(set_info['desc'])
        likes.append(set_info['likes'])
        image_data.append(encoded_image)
        image_ids.append(image_info['index'])

        images = np.array(images)
        likes = np.array(likes)

        # Tokenize the text data
        tokenizer = Tokenizer(num_words=vocab_size)
        tokenizer.fit_on_texts(texts)
        text_sequences = tokenizer.texts_to_sequences(texts)
        text_sequences = pad_sequences(text_sequences, maxlen=SEQ_LENGTH)

        # Ensure caption is decoded to a string
        caption = image_info['name']
        if isinstance(caption, bytes):
            caption = caption.decode('utf-8')
        caption_data.append(caption.encode('utf-8'))
        caption_id = [vocab.word_to_id(word) + 1 for word in caption.split()]
        caption_ids.append(caption_id)

    if not image_data:
        logging.warning("No valid images found for set: %s", set_id)
        return None

    feature = {}
    # Only keep 8 images, if outfit has less than 8 items, repeat the last one.
    for index in range(8):
        if index >= len(image_data):
            feature['images/' + str(index)] = _bytes_feature(image_data[-1])
        else:
            feature['images/' + str(index)] = _bytes_feature(image_data[index])

    feature["set_id"] = _bytes_feature(set_id.encode('utf-8'))
    feature["set_url"] = _bytes_feature(set_info['set_url'].encode('utf-8'))
    # Likes and Views are not used in our model, but we put it into TFRecords.
    feature["likes"] = _int64_feature(set_info['likes'])
    feature["views"] = _int64_feature(set_info['views'])

    context = tf.train.Features(feature=feature)

    feature_lists = tf.train.FeatureLists(feature_list={
        "caption": _bytes_feature_list(caption_data),
        "caption_ids": _int64_feature_list(caption_ids),
        "image_index": _int64_feature_list(image_ids)
    })

    sequence_example = tf.train.SequenceExample(
        context=context, feature_lists=feature_lists)

    return sequence_example

class ImageCoder(object):
    """Helper class that provides TensorFlow image coding utilities."""

    def __init__(self):
        # Create a single Session to run all image coding calls.
        self._sess = tf.compat.v1.Session()

        # Initializes function that converts PNG to JPEG data.
        self._png_data = tf.compat.v1.placeholder(dtype=tf.string)
        image = tf.image.decode_png(self._png_data, channels=3)
        self._png_to_jpeg = tf.image.encode_jpeg(image, format='rgb', quality=100)

        # Initializes function that decodes RGB JPEG data.
        self._decode_jpeg_data = tf.compat.v1.placeholder(dtype=tf.string)
        self._decode_jpeg = tf.image.decode_jpeg(self._decode_jpeg_data, channels=3)

    def png_to_jpeg(self, image_data):
        return self._sess.run(self._png_to_jpeg, feed_dict={self._png_data: image_data})

    def decode_jpeg(self, image_data):
        image = self._sess.run(self._decode_jpeg, feed_dict={self._decode_jpeg_data: image_data})
        assert len(image.shape) == 3
        assert image.shape[2] == 3
        return image

def _process_image_files_batch(coder, thread_index, ranges, name, all_sets, vocab, num_shards):
    """Processes and saves list of images as TFRecord in 1 thread."""
    num_threads = len(ranges)
    assert not num_shards % num_threads
    num_shards_per_batch = int(num_shards / num_threads)

    shard_ranges = np.linspace(ranges[thread_index][0], ranges[thread_index][1], num_shards_per_batch + 1).astype(int)
    num_files_in_thread = ranges[thread_index][1] - ranges[thread_index][0]

    counter = 0
    for s in range(num_shards_per_batch):
        shard = thread_index * num_shards_per_batch + s
        output_filename = '%s-%.5d-of-%.5d' % (name, shard, num_shards)
        output_file = os.path.join(FLAGS.output_directory, output_filename)
        writer = tf.io.TFRecordWriter(output_file)

        shard_counter = 0
        files_in_shard = np.arange(shard_ranges[s], shard_ranges[s + 1], dtype=int)
        for i in files_in_shard:
            sequence_example = _to_sequence_example(all_sets[i], coder, vocab)
            if not sequence_example:
                logging.warning('Failed for set: %s', all_sets[i]['set_id'])
                continue
            writer.write(sequence_example.SerializeToString())
            shard_counter += 1
            counter += 1

            if not counter % 100:
                logging.info('%s [thread %d]: Processed %d of %d images in thread batch.',
                             datetime.now(), thread_index, counter, num_files_in_thread)

        writer.close()
        logging.info('%s [thread %d]: Wrote %d images to %s',
                     datetime.now(), thread_index, shard_counter, output_file)
        shard_counter = 0
    logging.info('%s [thread %d]: Wrote %d images to %d shards.',
                 datetime.now(), thread_index, counter, num_files_in_thread)


def _process_image_files(name, all_sets, vocab, num_shards):
    """
    Process and save list of images as TFRecord of Example protos.
    
    Args:
        name (str): Unique identifier specifying the data set.
        all_sets (list): List of all image sets.
        vocab (Vocabulary): Vocabulary object for processing captions.
        num_shards (int): Number of shards for this data set.
    """
    # Break all images into batches with a [ranges[i][0], ranges[i+1]].
    spacing = np.linspace(0, len(all_sets), FLAGS.num_threads + 1).astype(int)
    ranges = []
    for i in range(len(spacing) - 1):
        ranges.append([spacing[i], spacing[i + 1]])

    # Launch a thread for each batch.
    logging.info('Launching %d threads for spacings: %s', FLAGS.num_threads, ranges)

    # Create a mechanism for monitoring when all threads are finished.
    coord = tf.train.Coordinator()

    # Create a generic TensorFlow-based utility for converting all image codings.
    coder = ImageCoder()

    threads = []
    for thread_index in range(len(ranges)):
        args = (coder, thread_index, ranges, name, all_sets, vocab, num_shards)
        t = threading.Thread(target=_process_image_files_batch, args=args)
        t.start()
        threads.append(t)

    # Wait for all the threads to terminate.
    coord.join(threads)
    logging.info('%s: Finished writing all %d fashion sets in data set.', datetime.now(), len(all_sets))

def _create_vocab(filename):
    """Creates the vocabulary of word to word_id.
    """
    # Create the vocabulary dictionary.
    word_counts = open(filename).read().splitlines()
    #print('word_counts %s' % word_counts)
    reverse_vocab = [x.split()[0] for x in word_counts]
    unk_id = len(reverse_vocab)
    vocab_dict = dict([(x, y) for (y, x) in enumerate(reverse_vocab)])
    vocab = Vocabulary(vocab_dict, unk_id)

    return vocab

def _find_image_files(labels_file, name):
    """Build a list of all images files and labels in the data set.
    """
    # Read image ids
    all_sets = json.load(open(labels_file))

    # Shuffle the ordering of all image files in order to guarantee
    # random ordering of the images with respect to label in the
    # saved TFRecord files. Make the randomization repeatable.

    shuffled_index = list(range(len(all_sets)))
    random.seed(12345)
    random.shuffle(shuffled_index)

    all_sets = [all_sets[i] for i in shuffled_index]  
    print('Found %d fashion sets.' % (len(all_sets)))
    return all_sets

def _process_dataset(name, label_file, vocab, num_shards):
    """Process a complete data set and save it as a TFRecord.
    Args:
      name: string, unique identifier specifying the data set.
      directory: string, root path to the data set.
      num_shards: integer number of shards for this data set.
      labels_file: string, path to the labels file.
    """
    print(label_file)
    all_sets  = _find_image_files(label_file, name)
    _process_image_files(name, all_sets, vocab, num_shards)

def _read_dataset(name, label_file, vocab):
    """Process a json file and return a list of values.
    Args:
      name: string, json file name.
      label_file: string, path to the labels file.
      vocab: vocabulary for processing.
    """
    print(label_file)
    sets = []
    items = []
    # Initialize df here
    df = None
    try:
        with open(label_file, 'r') as file:
            data = json.load(file)

        if isinstance(data, list):
            print(f"JSON data is a list with {len(data)} elements.")
            for set_data in data:
                set_id = set_data['set_id']
                set_name = set_data['name']
                set_likes = set_data['likes']
    
                for item in set_data['items']:
                    item_data = {
                        'set_id': set_id,
                        'set_name': set_name,
                        'set_likes': set_likes,
                        'item_index': item['index'],
                        'item_name': item['name'],
                        'price': item['price'],
                        'likes': item['likes'],
                        'categoryid': item['categoryid'],
                        'item_image': item['image']
                    }
                    items.append(item_data)

            df = pd.DataFrame(items)
            print(df.head())
        else:
            print(json.dumps(data, indent=4))

    except FileNotFoundError:
        print(f"Error: The file {label_file} was not found.")
    except json.JSONDecodeError:
        print(f"Error: The file {label_file} is not a valid JSON.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    
    #plot_graphs(df)
    _split_train_test_dataset(df)

def plot_graphs(df):
    plt.figure(figsize=(10, 5))
    sns.histplot(df['price'], bins=30, kde=True)
    plt.title('Distribution of Item Prices')
    plt.xlabel('Price')
    plt.ylabel('Frequency')
    plt.show()

    plt.figure(figsize=(10, 5))
    sns.histplot(df['likes'], bins=30, kde=True)
    plt.title('Distribution of Item Likes')
    plt.xlabel('Number of Likes')
    plt.ylabel('Frequency')
    plt.show()

    plt.figure(figsize=(10, 5))
    sns.scatterplot(data=df, x='price', y='likes', hue='categoryid', palette='tab10', alpha=0.7)
    plt.title('Price vs. Likes by Category')
    plt.xlabel('Price')
    plt.ylabel('Likes')
    plt.legend(title='Category ID', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.show()

def calculate_classification_metrics(true_matrix, predicted_matrix, threshold=0.5):
    # Binarize the predicted matrix
    binarizer = Binarizer(threshold=threshold)
    predicted_binary = binarizer.fit_transform(predicted_matrix)
    
    # Binarize the true matrix (assuming likes > 0 means liked)
    true_binary = binarizer.fit_transform(true_matrix)
    
    # Flatten the matrices
    true_flat = true_binary.flatten()
    predicted_flat = predicted_binary.flatten()
    
    # Calculate metrics
    precision = precision_score(true_flat, predicted_flat, average='weighted')
    recall = recall_score(true_flat, predicted_flat, average='weighted')
    f1 = f1_score(true_flat, predicted_flat, average='weighted')
    
    return precision, recall, f1

def _split_train_test_dataset(df_data):
    train_data, test_data = train_test_split(df_data, test_size=0.2, random_state=42)
    print(train_data)
    print(test_data)
    user_item_matrix = train_data.pivot_table(index='set_id', columns='item_name', values='likes').fillna(0)
    test_user_item_matrix = test_data.pivot_table(index='set_id', columns='item_name', values='likes').fillna(0)
    
    # Call Hybrid Model
    hybrid_model(train_data, test_data, user_item_matrix, test_user_item_matrix)





def calculate_metrics(true_matrix, predicted_matrix):
    mae = mean_absolute_error(true_matrix, predicted_matrix)
    rmse = np.sqrt(mean_squared_error(true_matrix, predicted_matrix))
    return mae, rmse


def collaborative_filtering(train_data, test_data, user_item_matrix, test_user_item_matrix):
    svd = TruncatedSVD(n_components=50, random_state=42)
    user_matrix = svd.fit_transform(user_item_matrix)
    item_matrix = svd.components_
    predicted_ratings = np.dot(user_matrix, item_matrix)
    
    # Ensure the predicted ratings have the same shape as the test matrix
    predicted_ratings_test = predicted_ratings[:test_user_item_matrix.shape[0], :test_user_item_matrix.shape[1]]
    
    mae_cf, rmse_cf = calculate_metrics(test_user_item_matrix.values, predicted_ratings_test)
    precision_cf, recall_cf, f1_cf = calculate_classification_metrics(test_user_item_matrix.values, predicted_ratings_test)
    
    print(f"Collaborative Filtering - Precision: {precision_cf:.2f}, Recall: {recall_cf:.2f}, F1-Score: {f1_cf:.2f}, MAE: {mae_cf:.2f}, RMSE: {rmse_cf:.2f}")
    return predicted_ratings_test


def content_based_filtering(train_data, test_data, user_item_matrix_train, user_item_matrix_test):
    all_items = pd.concat([train_data[['set_id', 'item_name']], test_data[['set_id', 'item_name']]])
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(all_items['item_name'])
    item_id_to_index = {item_id: idx for idx, item_id in enumerate(all_items['set_id'].unique())}
    train_indices = train_data['set_id'].map(item_id_to_index).values
    test_indices = test_data['set_id'].map(item_id_to_index).values
    tfidf_matrix_train = tfidf_matrix[train_indices]
    tfidf_matrix_test = tfidf_matrix[test_indices]
    print("tfidf_matrix_train shape:", tfidf_matrix_train.shape)
    print("tfidf_matrix_test shape:", tfidf_matrix_test.shape)
    user_item_matrix_test = user_item_matrix_test.reindex(columns=user_item_matrix_train.columns, fill_value=0)
    user_item_matrix_train = user_item_matrix_train.reindex(index=train_data['set_id'].map(item_id_to_index).values, fill_value=0)
    user_item_matrix_test = user_item_matrix_test.reindex(index=test_data['set_id'].map(item_id_to_index).values, fill_value=0)
    cosine_sim = cosine_similarity(tfidf_matrix_test, tfidf_matrix_train)
    predicted_likes_test = np.dot(cosine_sim, user_item_matrix_train)
    
    # Ensure the predicted likes have the same shape as the test matrix
    predicted_likes_test = predicted_likes_test[:user_item_matrix_test.shape[0], :user_item_matrix_test.shape[1]]
    
    mae_cb, rmse_cb = calculate_metrics(user_item_matrix_test.values, predicted_likes_test)
    precision_cb, recall_cb, f1_cb = calculate_classification_metrics(user_item_matrix_test.values, predicted_likes_test)
    print(f"Content-Based Filtering - Precision: {precision_cb:.2f}, Recall: {recall_cb:.2f}, F1-Score: {f1_cb:.2f}, MAE: {mae_cb:.2f}, RMSE: {rmse_cb:.2f}")
    return predicted_likes_test


def hybrid_model(train_data, test_data, user_item_matrix, test_user_item_matrix):
    # Get predictions from collaborative filtering and content-based filtering models
    cf_predictions = collaborative_filtering(train_data, test_data, user_item_matrix, test_user_item_matrix)
    cb_predictions = content_based_filtering(train_data, test_data, user_item_matrix, test_user_item_matrix)

    # Ensure that the predictions have the same shape
    cf_predictions, cb_predictions = reindex_data(cf_predictions, cb_predictions)

    # Combine predictions (e.g., via weighted sum or average)
    hybrid_predictions = (cf_predictions + cb_predictions) / 2.0

    mae_hybrid, rmse_hybrid = calculate_metrics(test_user_item_matrix.values, hybrid_predictions)
    precision_hybrid, recall_hybrid, f1_hybrid = calculate_classification_metrics(test_user_item_matrix.values, hybrid_predictions)
    
    print(f"Hybrid Model - Precision: {precision_hybrid:.2f}, Recall: {recall_hybrid:.2f}, F1-Score: {f1_hybrid:.2f}, MAE: {mae_hybrid:.2f}, RMSE: {rmse_hybrid:.2f}")
    return hybrid_predictions
    return hybrid_predictions



def reindex_data(cf_data, cb_data):
    # Ensure both predictions have the same shape
    if cf_data.shape != cb_data.shape:
        # Find the common shape to align both arrays
        min_shape = (min(cf_data.shape[0], cb_data.shape[0]), min(cf_data.shape[1], cb_data.shape[1]))
        
        # Slice both arrays to the common shape
        cf_data = cf_data[:min_shape[0], :min_shape[1]]
        cb_data = cb_data[:min_shape[0], :min_shape[1]]
    
    return cf_data, cb_data


## adding DL
# Define image dimensions and text sequence length
IMG_HEIGHT, IMG_WIDTH = 128, 128
SEQ_LENGTH = 100
vocab_size = 10000  # Adjust based on your vocabulary size

# Load and preprocess the image
def preprocess_image(image_url, target_size):
    response = requests.get(image_url)
    image = Image.open(BytesIO(response.content))
    image = image.resize(target_size)
    image = img_to_array(image)
    image = image / 255.0  # Normalize to [0, 1]
    return image

# Preprocess the text
def preprocess_text(text, tokenizer, max_length):
    sequences = tokenizer.texts_to_sequences([text])
    padded_sequences = pad_sequences(sequences, maxlen=max_length)
    return padded_sequences



def main(_argv):
    print("Running")
    os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
    assert not FLAGS.train_shards % FLAGS.num_threads, (
        'Please make the FLAGS.num_threads commensurate with FLAGS.train_shards')
    assert not FLAGS.test_shards % FLAGS.num_threads, (
        'Please make the FLAGS.num_threads commensurate with FLAGS.test_shards')
    assert not FLAGS.valid_shards % FLAGS.num_threads, (
        'Please make the FLAGS.num_threads commensurate with FLAGS.valid_shards')
    print('Saving results to %s' % FLAGS.output_directory)

    print('Vocab file %s' % FLAGS.word_dict_file)

    vocab = _create_vocab(FLAGS.word_dict_file)

    print('Vocab file %s' % vocab.word_to_id('famous'))
    # Run it!
    _process_dataset('valid-no-dup', FLAGS.valid_label, vocab, FLAGS.valid_shards)
    #_process_dataset('test-no-dup', FLAGS.test_label, vocab, FLAGS.test_shards)
    #_process_dataset('train-no-dup', FLAGS.train_label, vocab, FLAGS.train_shards)
    #_read_dataset('valid-no-dup', FLAGS.valid_label, vocab)

if __name__ == '__main__':
    app.run(main)
