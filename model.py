import tensorflow as tf
from tensorflow.keras.layers import Input, LSTM, Embedding, TimeDistributed, Bidirectional, Dense, Layer, InputSpec
from tensorflow.keras.models import Model
from tensorflow_addons.text import crf_decode, crf_log_likelihood
from tensorflow.keras import backend as K

class CRF(Layer):
    def __init__(self,
                 output_dim,
                 sparse_target=True,
                 transitions=None,
                 **kwargs):
        """
        Args:
            output_dim (int): the number of labels to tag each temporal input.
            sparse_target (bool): whether the the ground-truth label represented in one-hot.
        Input shape:
            (batch_size, sentence length, output_dim)
        Output shape:
            (batch_size, sentence length, output_dim)
        """
        super(CRF, self).__init__(**kwargs)
        self.output_dim = int(output_dim)
        self.sparse_target = sparse_target
        self.input_spec = InputSpec(min_ndim=3)
        self.supports_masking = False
        self.sequence_lengths = None
        self.transitions = transitions

    def build(self, input_shape):
        assert len(input_shape) == 3
        f_shape = tf.TensorShape(input_shape)
        input_spec = InputSpec(min_ndim=3, axes={-1: f_shape[-1]})

        if f_shape[-1] is None:
            raise ValueError('The last dimension of the inputs to `CRF` '
                             'should be defined. Found `None`.')
        if f_shape[-1] != self.output_dim:
            raise ValueError('The last dimension of the input shape must be equal to output'
                             ' shape. Use a linear layer if needed.')
        self.input_spec = input_spec
        self.transitions = self.add_weight(name='transitions',
                                           shape=[self.output_dim, self.output_dim],
                                           initializer='glorot_uniform',
                                           trainable=True)
        self.built = True

    def compute_mask(self, inputs, mask=None):
        # Just pass the received mask from previous layer, to the next layer or
        # manipulate it if this layer changes the shape of the input
        return mask

    def call(self, inputs, sequence_lengths=None, training=None, **kwargs):
        sequences = tf.convert_to_tensor(inputs, dtype=self.dtype)
        if sequence_lengths is not None:
            assert len(sequence_lengths.shape) == 2
            assert tf.convert_to_tensor(sequence_lengths).dtype == 'int32'
            seq_len_shape = tf.convert_to_tensor(sequence_lengths).get_shape().as_list()
            assert seq_len_shape[1] == 1
            self.sequence_lengths = K.flatten(sequence_lengths)
        else:
            self.sequence_lengths = tf.ones(tf.shape(inputs)[0], dtype=tf.int32) * (
                tf.shape(inputs)[1]
            )

        viterbi_sequence, _ = crf_decode(sequences,
                                         self.transitions,
                                         self.sequence_lengths)
        output = K.one_hot(viterbi_sequence, self.output_dim)
        return K.in_train_phase(sequences, output)

    @property
    def loss(self):
        def crf_loss(y_true, y_pred):
            y_pred = tf.convert_to_tensor(y_pred, dtype=self.dtype)
            log_likelihood, self.transitions = crf_log_likelihood(
                y_pred,
                tf.cast(K.argmax(y_true), dtype=tf.int32) if self.sparse_target else y_true,
                self.sequence_lengths,
                transition_params=self.transitions,
            )
            return tf.reduce_mean(-log_likelihood)
        return crf_loss

    @property
    def accuracy(self):
        def viterbi_accuracy(y_true, y_pred):
            # -1e10 to avoid zero at sum(mask)
            mask = K.cast(
                K.all(K.greater(y_pred, -1e10), axis=2), K.floatx())
            shape = tf.shape(y_pred)
            sequence_lengths = tf.ones(shape[0], dtype=tf.int32) * (shape[1])
            y_pred, _ = crf_decode(y_pred, self.transitions, sequence_lengths)
            if self.sparse_target:
                y_true = K.argmax(y_true, 2)
            y_pred = K.cast(y_pred, 'int32')
            y_true = K.cast(y_true, 'int32')
            corrects = K.cast(K.equal(y_true, y_pred), K.floatx())
            return K.sum(corrects * mask) / K.sum(mask)
        return viterbi_accuracy

    def compute_output_shape(self, input_shape):
        tf.TensorShape(input_shape).assert_has_rank(3)
        return input_shape[:2] + (self.output_dim,)


    def get_config(self):
        config = super(CRF, self).get_config()
        config.update({
            'output_dim': self.output_dim,
            'sparse_target': self.sparse_target,
            'transitions': self.transitions.numpy()  # Convert the transitions to a NumPy array
        })
        return config
    
    @classmethod
    def from_config(cls, config):
        # Since 'transitions' is a NumPy array, we need to convert it back to a tensor
        transitions = tf.convert_to_tensor(config['transitions'])
        # Create a new instance of CRF with the saved configuration
        return cls(output_dim=config['output_dim'], sparse_target=config['sparse_target'], transitions=transitions)
    
def embedding_layer(input_dim, output_dim, input_length, mask_zero):
    return Embedding(input_dim = input_dim, output_dim = output_dim, input_length = input_length, mask_zero = mask_zero)

def bilstm_crf(maxlen, n_tags, lstm_units, embedding_dim, n_words, mask_zero, training = True):
    """
    bilstm_crf - module to build BiLSTM-CRF model
    Inputs:
        - input_shape : tuple
            Tensor shape of inputs, excluding batch size
    Outputs:
        - output : tensorflow.keras.outputs.output
            BiLSTM-CRF output
    """
    input = Input(shape = (maxlen,))
    # Embedding layer
    embeddings = embedding_layer(input_dim = n_words, output_dim = embedding_dim, input_length = maxlen, mask_zero = mask_zero)
    output = embeddings(input)

    # BiLSTM layer
    output = Bidirectional(LSTM(units = lstm_units, return_sequences = True, recurrent_dropout = 0.1))(output)

    # Dense layer
    output = TimeDistributed(Dense(n_tags, activation = 'relu'))(output)

    output = CRF(n_tags, name = 'crf_layer')(output)
    return Model(input, output)


viterbi_accuracy = CRF.accuracy.fget
crf_loss = CRF.loss.fget

def load_model(model_path):
    loaded_model = tf.keras.models.load_model(model_path, custom_objects={'CRF': CRF, 'viterbi_accuracy': viterbi_accuracy, 'crf_loss': crf_loss})
    return loaded_model

def save_model(model, model_path):
    model.save(model_path)