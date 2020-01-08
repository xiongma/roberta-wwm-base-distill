from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from bert import modeling
import collections
import tensorflow as tf

flags = tf.flags

FLAGS = flags.FLAGS

## Required parameters
flags.DEFINE_string(
    "bert_config_file", None,
    "The config json file corresponding to the pre-trained BERT model. "
    "This specifies the model architecture.")

flags.DEFINE_string(
    "input_file", None,
    "Input TF example files (can be a glob or comma separated).")

flags.DEFINE_string(
    "output_dir", None,
    "The output directory where the model checkpoints will be written.")

flags.DEFINE_integer("truncation_factor", 128,
                     "Number of top probable words to save from teacher network output")

## Other parameters
flags.DEFINE_string(
    "init_checkpoint", r'D:\Awake\Code\DistillBERT\bert-base-chinese\bert_model.ckpt',
    "Initial checkpoint (usually from a pre-trained BERT model).")

flags.DEFINE_integer(
    "max_seq_length", 512,
    "The maximum total input sequence length after WordPiece tokenization. "
    "Sequences longer than this will be truncated, and sequences shorter "
    "than this will be padded. Must match data generation.")

flags.DEFINE_integer(
    "max_predictions_per_seq", 20,
    "Maximum number of masked LM predictions per sequence. "
    "Must match data generation.")

flags.DEFINE_integer("predict_batch_size", 4, "Total batch size.")

flags.DEFINE_bool("tpu", False, "Whether to use TPU or GPU/CPU.")

tf.flags.DEFINE_string("master", None, "[Optional] TensorFlow master URL.")

tf.flags.DEFINE_string(
    "tpu_name", None,
    "The Cloud TPU to use for training. This should be either the name "
    "used when creating the Cloud TPU, or a grpc://ip.address.of.tpu:8470 "
    "url.")

tf.flags.DEFINE_string(
    "tpu_zone", None,
    "[Optional] GCE zone where the Cloud TPU is located in. If not "
    "specified, we will attempt to automatically detect the GCE project from "
    "metadata.")

tf.flags.DEFINE_string(
    "gcp_project", None,
    "[Optional] Project name for the Cloud TPU-enabled project. If not "
    "specified, we will attempt to automatically detect the GCE project from "
    "metadata.")

flags.DEFINE_integer(
    "num_tpu_cores", 8,
    "Only used if `use_tpu` is True. Total number of TPU cores to use.")

def input_fn_builder(input_files):
    """Decodes a record to a TensorFlow example."""
    name_to_features = {
        "input_ids":
            tf.FixedLenFeature([FLAGS.max_seq_length], tf.int64),
        "input_mask":
            tf.FixedLenFeature([FLAGS.max_seq_length], tf.int64),
        "segment_ids":
            tf.FixedLenFeature([FLAGS.max_seq_length], tf.int64),
        "masked_lm_positions":
            tf.FixedLenFeature([FLAGS.max_predictions_per_seq], tf.int64),
        "masked_lm_ids":
            tf.FixedLenFeature([FLAGS.max_predictions_per_seq], tf.int64),
        "masked_lm_weights":
            tf.FixedLenFeature([FLAGS.max_predictions_per_seq], tf.float32),
        "next_sentence_labels":
            tf.FixedLenFeature([1], tf.int64),
    }

    def _decode_record(record, name_to_features):
        example = tf.parse_single_example(record, name_to_features)

        for name in list(example.keys()):
            t = example[name]
            if t.dtype == tf.int64:
                t = tf.to_int32(t)
            example[name] = t

        return example

    def input_fn(params):
        batch_size = params["batch_size"]
        d = tf.data.TFRecordDataset(input_files)
        # d = tf.data.Dataset.from_tensor_slices(tf.constant(input_files))
        d = d.repeat(1)

        d = d.apply(
            tf.contrib.data.map_and_batch(
                lambda record: _decode_record(record, name_to_features),
                batch_size=batch_size,
                drop_remainder=False))

        return d

    return input_fn

def get_masked_lm_output(bert_config, input_tensor, output_weights, positions, truncation_factor):
  """Get loss and log probs for the masked LM."""
  input_tensor = gather_indexes(input_tensor, positions)

  with tf.variable_scope("cls/predictions"):
    # We apply one more non-linear transformation before the output layer.
    # This matrix is not used after pre-training.
    with tf.variable_scope("transform"):
      input_tensor = tf.layers.dense(
          input_tensor,
          units=bert_config.hidden_size,
          activation=modeling.get_activation(bert_config.hidden_act),
          kernel_initializer=modeling.create_initializer(
              bert_config.initializer_range))
      input_tensor = modeling.layer_norm(input_tensor)

    # The output weights are the same as the input embeddings, but there is
    # an output-only bias for each token.
    output_bias = tf.get_variable(
        "output_bias",
        shape=[bert_config.vocab_size],
        initializer=tf.zeros_initializer())
    logits = tf.matmul(input_tensor, output_weights, transpose_b=True)
    logits = tf.nn.bias_add(logits, output_bias)
    masked_lm_probs = tf.nn.softmax(logits, axis=-1)
    trunc_masked_lm_probs, top_indices = tf.math.top_k(masked_lm_probs, k=truncation_factor, sorted=False)

    max_predictions_per_seq = positions.get_shape().as_list()[1]
    truncation_factor_ = top_indices.get_shape().as_list()[1]

    trunc_masked_lm_probs = tf.reshape(trunc_masked_lm_probs, [-1, max_predictions_per_seq, truncation_factor_])
    top_indices = tf.reshape(top_indices, [-1, max_predictions_per_seq, truncation_factor_])
  return trunc_masked_lm_probs, top_indices

def get_next_sentence_output(bert_config, input_tensor):
  """Get loss and log probs for the next sentence prediction."""

  # Simple binary classification. Note that 0 is "next sentence" and 1 is
  # "random sentence". This weight matrix is not used after pre-training.
  with tf.variable_scope("cls/seq_relationship"):
    output_weights = tf.get_variable(
        "output_weights",
        shape=[2, bert_config.hidden_size],
        initializer=modeling.create_initializer(bert_config.initializer_range))
    output_bias = tf.get_variable(
        "output_bias", shape=[2], initializer=tf.zeros_initializer())

    logits = tf.matmul(input_tensor, output_weights, transpose_b=True)
    logits = tf.nn.bias_add(logits, output_bias)
    next_sentence_probs = tf.nn.softmax(logits, axis=-1)

    return next_sentence_probs

def gather_indexes(sequence_tensor, positions):
  """Gathers the vectors at the specific positions over a minibatch."""
  sequence_shape = modeling.get_shape_list(sequence_tensor, expected_rank=3)
  batch_size = sequence_shape[0]
  seq_length = sequence_shape[1]
  width = sequence_shape[2]

  flat_offsets = tf.reshape(
      tf.range(0, batch_size, dtype=tf.int32) * seq_length, [-1, 1])
  flat_positions = tf.reshape(positions + flat_offsets, [-1])
  flat_sequence_tensor = tf.reshape(sequence_tensor,
                                    [batch_size * seq_length, width])
  output_tensor = tf.gather(flat_sequence_tensor, flat_positions)

  return output_tensor


def model_fn_builder(bert_config, init_checkpoint, truncation_factor):

    """The `model_fn` for TPUEstimator."""
    def model_fn(features, labels, mode, params):
        tf.logging.info("*** Features ***")
        for name in sorted(features.keys()):
          tf.logging.info("  name = %s, shape = %s" % (name, features[name].shape))

        input_ids = features["input_ids"]
        input_mask = features["input_mask"]
        segment_ids = features["segment_ids"]
        masked_lm_positions = features["masked_lm_positions"]

        model = modeling.BertModel(
            config=bert_config,
            is_training=False,
            input_ids=input_ids,
            input_mask=input_mask,
            token_type_ids=segment_ids,
            use_one_hot_embeddings=False)

        trunc_masked_lm_probs, top_indices = get_masked_lm_output(bert_config,
                                                   model.get_sequence_output(),
                                                   model.get_embedding_table(),
                                                   masked_lm_positions,
                                                   truncation_factor)

        tvars = tf.trainable_variables()

        initialized_variable_names = {}
        scaffold_fn = None

        if init_checkpoint:
            (assignment_map, initialized_variable_names) = modeling.get_assignment_map_from_checkpoint(tvars, init_checkpoint)
            tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

            if FLAGS.tpu:
                def tpu_scaffold():
                    tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
                    return tf.train.Scaffold()

                scaffold_fn = tpu_scaffold

            else:
                tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

        tf.logging.info("**** Trainable Variables ****")
        for var in tvars:
            init_string = ""
            if var.name in initialized_variable_names:
                init_string = ", *INIT_FROM_CKPT*"
            tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape, init_string)

        output_spec = None
        if mode == tf.estimator.ModeKeys.PREDICT:
            features['masked_lm_probs'] = trunc_masked_lm_probs
            features['top_indices'] = top_indices

            output_spec = tf.contrib.tpu.TPUEstimatorSpec(mode=mode, predictions=features, scaffold_fn=scaffold_fn)

        else:
            raise ValueError('this is only for predict, please change the estimator to predict')

        return output_spec

    return model_fn

def create_int_feature(values):
  feature = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
  return feature


def create_float_feature(values):
  feature = tf.train.Feature(float_list=tf.train.FloatList(value=list(values)))
  return feature


def main(_):
    tf.logging.set_verbosity(tf.logging.INFO)

    input_files = []

    tf.gfile.MakeDirs(FLAGS.output_dir)

    for input_pattern in FLAGS.input_file.split(","):
        input_files.extend(tf.gfile.Glob(input_pattern))

    tf.logging.info("*** Input Files ***")
    for input_file in input_files:
        tf.logging.info("  %s" % input_file)

    tf.contrib.tpu.RunConfig()

    tpu_cluster_resolver = None
    if FLAGS.tpu and FLAGS.tpu_name:
        tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(
            FLAGS.tpu_name, zone=FLAGS.tpu_zone, project=FLAGS.gcp_project)

    is_per_host = tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2

    run_config = tf.contrib.tpu.RunConfig(
        cluster=tpu_cluster_resolver,
        master=FLAGS.master,
        tpu_config=tf.contrib.tpu.TPUConfig(
            num_shards=FLAGS.num_tpu_cores,
            per_host_input_for_training=is_per_host))

    bert_config = modeling.BertConfig.from_json_file(FLAGS.bert_config_file)
    model_fn = model_fn_builder(bert_config, FLAGS.init_checkpoint, FLAGS.truncation_factor)

    estimator = tf.contrib.tpu.TPUEstimator(
            use_tpu=FLAGS.tpu,
            model_fn=model_fn,
            config=run_config,
            train_batch_size=256,
            predict_batch_size=FLAGS.predict_batch_size)

    input_fn = input_fn_builder(input_files)

    writer = None
    count = 0
    tf.logging.info('start create teacher output!')
    for features in estimator.predict(input_fn, yield_single_examples=False):
        masked_lm_probs = features['masked_lm_probs']
        top_indices = features['top_indices']

        input_ids = features["input_ids"]
        input_mask = features["input_mask"]
        segment_ids = features["segment_ids"]
        masked_lm_positions = features["masked_lm_positions"]
        masked_lm_ids = features["masked_lm_ids"]
        masked_lm_weights = features["masked_lm_weights"]
        next_sentence_labels = features["next_sentence_labels"]

        cur_batch_size, seq_len = input_ids.shape
        _, pred_per_seq = masked_lm_positions.shape
        feature = collections.OrderedDict()
        for i in range(cur_batch_size):
            if count % 50000 == 0:
                writer = tf.python_io.TFRecordWriter(FLAGS.output_dir + 'teacher_%d.tfrecord' % (count / 50000))

            feature['input_ids'] = create_int_feature(input_ids[i])
            feature["input_mask"] = create_int_feature(input_mask[i])
            feature["segment_ids"] = create_int_feature(segment_ids[i])
            feature["masked_lm_positions"] = create_int_feature(masked_lm_positions[i])
            feature["masked_lm_ids"] = create_int_feature(masked_lm_ids[i])
            feature["masked_lm_weights"] = create_float_feature(masked_lm_weights[i])
            feature["next_sentence_labels"] = create_int_feature([next_sentence_labels[i][0]])

            # retrieve predictions for (pred_per_seq) number of words, and flatten
            distribution = masked_lm_probs[i].reshape(-1)
            feature["truncated_masked_lm_probs"] = create_float_feature(distribution)

            top_k_dex = top_indices[i].reshape(-1)
            feature["top_k_indices"] = create_int_feature(top_k_dex)

            tf_example = tf.train.Example(features=tf.train.Features(feature=feature))
            writer.write(tf_example.SerializeToString())
            count += 1

    writer.close()
    tf.logging.info('create teacher output over!')

if __name__ =="__main__":
    tf.app.run()