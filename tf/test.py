import os
import tensorflow as tf

tpu_grpc_url = tf.contrib.cluster_resolver.TPUClusterResolver(
            tpu=[os.environ['TPU_NAME']]).get_master()
# Read and print data:
sess = tf.InteractiveSession(tpu_grpc_url)

def do_things(total_batch_size=1):
  filenames = ["gs://jeremylorino-staging-bq-data/chess/lczero_data/test.tfrecords"]
  dataset = tf.data.TFRecordDataset(filenames)

  def map_fn(example_proto):
      """
      planes <tf.Tensor 'Reshape:0' shape=(2048, 112, 64) dtype=float32>
      probs <tf.Tensor 'Reshape_1:0' shape=(2048, 1858) dtype=float32>
      winner <tf.Tensor 'Reshape_2:0' shape=(2048, 1) dtype=float32>
      """

      tfrecord_features = {
          # "image": tf.FixedLenFeature((), tf.string, default_value=""),
          # "label": tf.FixedLenFeature((), tf.int64, default_value=0)
          
          # 'planes': tf.FixedLenFeature((112, 64), tf.float32),
          'probs': tf.FixedLenFeature((1858), tf.float32),
          # 'winner': tf.FixedLenFeature((1), tf.float32)
          # 'planes': tf.FixedLenFeature([], tf.float32),
          # 'probs': tf.FixedLenFeature([], tf.float32)
          # 'winner': tf.FixedLenFeature([], tf.float32)
      }
      parsed_features = tf.parse_single_example(example_proto, tfrecord_features)
      # planes, probs, winner = parsed_features["planes"], parsed_features["probs"], parsed_features["winner"]
      probs = parsed_features["probs"]
      # winner = parsed_features["winner"]

      # return planes, probs, tf.cast(winner, tf.float32)
      # return winner
      return probs
  
  # dataset = dataset.map(ChunkParser.parse_function, num_parallel_calls=10)
  dataset = dataset.map(map_fn, num_parallel_calls=10).batch(total_batch_size)
  dataset = dataset.prefetch(4)
  train_iterator = dataset.make_one_shot_iterator()

  next_element = train_iterator.get_next()
  print(sess.run(next_element))

do_things(10)



# filenames = ["gs://jeremylorino-staging-bq-data/chess/lczero_data/train.tfrecords"]

# # Define features
# read_features = {
#   'planes': tf.FixedLenFeature((112, 64), tf.float32),
#   # 'probs': tf.FixedLenFeature((1858), tf.float32),
#   # 'winner': tf.FixedLenFeature((1), tf.float32)
#   # 'planes': tf.FixedLenFeature([], tf.float32),
#   # 'probs': tf.FixedLenFeature([], tf.float32),
#   'winner': tf.FixedLenFeature([], tf.float32)}

# # Read TFRecord file
# reader = tf.TFRecordReader()
# # reader = tf.data.TFRecordDataset(['/tmp/data/train.tfrecords'])
# filename_queue = tf.train.string_input_producer(filenames)
# ds1 = tf.data.Dataset.from_tensor_slices(read_features)
# print(ds1)
# # .shuffle(tf.shape(input_tensor, out_type=tf.int64)[0])
# # .repeat(num_epochs)

# _, serialized_example = reader.read(filename_queue)

# # Extract features from serialized data
# read_data = tf.parse_single_example(serialized=serialized_example,
#                                     features=read_features)

# # Many tf.train functions use tf.train.QueueRunner,
# # so we need to start it before we read
# tf.train.start_queue_runners(sess)
# # tf.data.start_queue_runners(sess)

# # Print features
# for name, tensor in read_data.items():
#   value = tensor.eval()
#   print('{}: {} ({})'.format(name, value, type(value)))