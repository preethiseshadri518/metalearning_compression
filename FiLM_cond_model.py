import tensorflow as tf
import tensorflow_compression as tfc

SCALE_MIN = 0.11
SCALE_MAX = 255.
NUM_SCALES = 64


def mean(layer):
    return tf.keras.backend.mean(layer, axis=0)


def ContextModel(context_length=64, num_filters=128, kernel_size=[5,5,5,5]):
    # Using sequential model here because functional model doesn't allow for
    # input shape to contain None if there are dense layers
    model = tf.keras.Sequential()

    model.add(tf.keras.layers.Conv2D(filters=num_filters,
        kernel_size=kernel_size[0],
        strides=2,
        padding='same',
        activation='relu',
        name='context_0'))

    model.add(tf.keras.layers.Conv2D(filters=num_filters,
        kernel_size=kernel_size[1],
        strides=2,
        padding='same',
        activation='relu',
        name='context_1'))

    model.add(tf.keras.layers.Conv2D(filters=num_filters,
        kernel_size=kernel_size[2],
        strides=2,
        padding='same',
        activation='relu',
        name='context_2'))

    model.add(tf.keras.layers.Conv2D(filters=num_filters,
        kernel_size=kernel_size[3],
        strides=2,
        padding='same',
        activation='relu',
        name='context_3'))

    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(context_length, name='context_embedding'))
    # Pool c_1,...,c_m into a single context embedding
    model.add(tf.keras.layers.Lambda(mean))

    return model


def AnalysisTransform(num_filters=128, context_length=64, kernel_size=[5,5,5,5]):
    input_image = tf.keras.Input((None,None,3), name='encoder_input')
    context_embedding = tf.keras.Input(shape=(context_length))

    gamma_e0 = tf.keras.layers.Dense(num_filters, name='gamma_e0')(context_embedding)
    beta_e0 = tf.keras.layers.Dense(num_filters, name='beta_e0')(context_embedding)
    gamma_e1 = tf.keras.layers.Dense(num_filters, name='gamma_e1')(context_embedding)
    beta_e1 = tf.keras.layers.Dense(num_filters, name='beta_e1')(context_embedding)
    gamma_e2 = tf.keras.layers.Dense(num_filters, name='gamma_e2')(context_embedding)
    beta_e2 = tf.keras.layers.Dense(num_filters, name='beta_e2')(context_embedding)

    encoder_0 = tf.keras.layers.Conv2D(filters=num_filters,
        kernel_size=kernel_size[0],
        strides=2,
        padding='same',
        name='encoder_0')(input_image)
    encoder_0 = tf.keras.layers.Multiply()([encoder_0, gamma_e0]) # scale by gamma along channel dim
    encoder_0 = tf.keras.layers.Add()([encoder_0, beta_e0]) # shift with beta along channel dim
    encoder_0 = tf.keras.layers.ReLU()(encoder_0)

    encoder_1 = tf.keras.layers.Conv2D(filters=num_filters,
        kernel_size=kernel_size[1],
        strides=2,
        padding='same',
        name='encoder_1')(encoder_0)
    encoder_1 = tf.keras.layers.Multiply()([encoder_1, gamma_e1])
    encoder_1 = tf.keras.layers.Add()([encoder_1, beta_e1])
    encoder_1 = tf.keras.layers.ReLU()(encoder_1)

    encoder_2 = tf.keras.layers.Conv2D(filters=num_filters,
        kernel_size=kernel_size[2],
        strides=2,
        padding='same',
        name='encoder_2')(encoder_1)
    encoder_2 = tf.keras.layers.Multiply()([encoder_2, gamma_e2])
    encoder_2 = tf.keras.layers.Add()([encoder_2, beta_e2])
    encoder_2 = tf.keras.layers.ReLU()(encoder_2)

    encoder_output = tf.keras.layers.Conv2D(filters=num_filters,
        kernel_size=kernel_size[3],
        strides=2,
        padding='same',
        name='encoder_output')(encoder_2)

    return tf.keras.Model([input_image, context_embedding], encoder_output, name='encoder_network')


def SynthesisTransform(num_filters=128, context_length=64, kernel_size=[5,5,5,5]):
    decoder_input = tf.keras.Input((None,None,num_filters), name='decoder_input')
    context_embedding = tf.keras.Input(shape=(context_length))

    gamma_d0 = tf.keras.layers.Dense(num_filters, name='gamma_d0')(context_embedding)
    beta_d0 = tf.keras.layers.Dense(num_filters, name='beta_d0')(context_embedding)
    gamma_d1 = tf.keras.layers.Dense(num_filters, name='gamma_d1')(context_embedding)
    beta_d1 = tf.keras.layers.Dense(num_filters, name='beta_d1')(context_embedding)
    gamma_d2 = tf.keras.layers.Dense(num_filters, name='gamma_d2')(context_embedding)
    beta_d2 = tf.keras.layers.Dense(num_filters, name='beta_d2')(context_embedding)

    decoder_0 = tf.keras.layers.Conv2DTranspose(filters=num_filters,
        kernel_size=kernel_size[0],
        strides=2,
        padding='same',
        name='decoder_0')(decoder_input)
    decoder_0 = tf.keras.layers.Multiply()([decoder_0, gamma_d0]) # scale by gamma along channel dim
    decoder_0 = tf.keras.layers.Add()([decoder_0, beta_d0]) # shift with beta along channel dim
    decoder_0 = tf.keras.layers.ReLU()(decoder_0)

    decoder_1 = tf.keras.layers.Conv2DTranspose(filters=num_filters,
        kernel_size=kernel_size[1],
        strides=2,
        padding='same',
        name='decoder_1')(decoder_0)
    decoder_1 = tf.keras.layers.Multiply()([decoder_1, gamma_d1])
    decoder_1 = tf.keras.layers.Add()([decoder_1, beta_d1])
    decoder_1 = tf.keras.layers.ReLU()(decoder_1)

    decoder_2 = tf.keras.layers.Conv2DTranspose(filters=num_filters,
        kernel_size=kernel_size[2],
        strides=2,
        padding='same',
        name='decoder_2')(decoder_1)
    decoder_2 = tf.keras.layers.Multiply()([decoder_2, gamma_d2])
    decoder_2 = tf.keras.layers.Add()([decoder_2, beta_d2])
    decoder_2 = tf.keras.layers.ReLU()(decoder_2)

    decoder_output = tf.keras.layers.Conv2DTranspose(filters=3,
        kernel_size=kernel_size[3],
        strides=2,
        padding='same',
        name='decoder_output')(decoder_2)

    return tf.keras.Model([decoder_input, context_embedding], decoder_output, name='decoder_network')


def HyperAnalysisTransform(num_filters=128, context_length=64, kernel_size=[3,5,5]):
    hyper_encoder_input = tf.keras.Input((None,None,num_filters), name='hyper_encoder_input')
    context_embedding = tf.keras.Input(shape=(context_length))

    gamma_he0 = tf.keras.layers.Dense(num_filters, name='gamma_he0')(context_embedding)
    beta_he0 = tf.keras.layers.Dense(num_filters, name='beta_he0')(context_embedding)
    gamma_he1 = tf.keras.layers.Dense(num_filters, name='gamma_he1')(context_embedding)
    beta_he1 = tf.keras.layers.Dense(num_filters, name='beta_he1')(context_embedding)

    hyper_encoder_0 = tf.keras.layers.Conv2D(filters=num_filters,
        kernel_size=kernel_size[0],
        strides=1,
        padding='same',
        name='hyper_encoder_0')(hyper_encoder_input)
    hyper_encoder_0 = tf.keras.layers.Multiply()([hyper_encoder_0, gamma_he0])
    hyper_encoder_0 = tf.keras.layers.Add()([hyper_encoder_0, beta_he0])
    hyper_encoder_0 = tf.keras.layers.ReLU()(hyper_encoder_0)

    hyper_encoder_1 = tf.keras.layers.Conv2D(filters=num_filters,
        kernel_size=kernel_size[1],
        strides=2,
        padding='same',
        name='hyper_encoder_1')(hyper_encoder_0)
    hyper_encoder_1 = tf.keras.layers.Multiply()([hyper_encoder_1, gamma_he1])
    hyper_encoder_1 = tf.keras.layers.Add()([hyper_encoder_1, beta_he1])
    hyper_encoder_1 = tf.keras.layers.ReLU()(hyper_encoder_1)

    hyper_encoder_output = tf.keras.layers.Conv2D(filters=num_filters,
        kernel_size=kernel_size[2],
        strides=2,
        padding='same',
        name='hyper_encoder_output')(hyper_encoder_1)

    return tf.keras.Model([hyper_encoder_input, context_embedding], hyper_encoder_output, name='hyper_encoder_network')


def HyperSynthesisTransform(num_filters=128, context_length=64, kernel_size=[5,5,3]):
    hyper_decoder_input = tf.keras.Input((None,None,num_filters), name='hyper_decoder_input')
    context_embedding = tf.keras.Input(shape=(context_length))

    gamma_hd0 = tf.keras.layers.Dense(num_filters, name='gamma_hd0')(context_embedding)
    beta_hd0 = tf.keras.layers.Dense(num_filters, name='beta_hd0')(context_embedding)
    gamma_hd1 = tf.keras.layers.Dense(num_filters, name='gamma_hd1')(context_embedding)
    beta_hd1 = tf.keras.layers.Dense(num_filters, name='beta_hd1')(context_embedding)

    hyper_decoder_0 = tf.keras.layers.Conv2DTranspose(filters=num_filters,
        kernel_size=kernel_size[0],
        strides=2,
        padding='same',
        name='hyper_decoder_0')(hyper_decoder_input)
    hyper_decoder_0 = tf.keras.layers.Multiply()([hyper_decoder_0, gamma_hd0])
    hyper_decoder_0 = tf.keras.layers.Add()([hyper_decoder_0, beta_hd0])
    hyper_decoder_0 = tf.keras.layers.ReLU()(hyper_decoder_0)

    hyper_decoder_1 = tf.keras.layers.Conv2DTranspose(filters=num_filters,
        kernel_size=kernel_size[1],
        strides=2,
        padding='same',
        name='hyper_decoder_1')(hyper_decoder_0)
    hyper_decoder_1 = tf.keras.layers.Multiply()([hyper_decoder_1, gamma_hd1])
    hyper_decoder_1 = tf.keras.layers.Add()([hyper_decoder_1, beta_hd1])
    hyper_decoder_1 = tf.keras.layers.ReLU()(hyper_decoder_1)

    hyper_decoder_output = tf.keras.layers.Conv2DTranspose(filters=num_filters,
        kernel_size=kernel_size[2],
        strides=1,
        padding='same',
        name='hyper_decoder_output')(hyper_decoder_1)

    return tf.keras.Model([hyper_decoder_input, context_embedding], hyper_decoder_output, name='hyper_decoder_network')


class BMSHJ2018Model(tf.keras.Model):
    """Main model class."""
    def __init__(self, lmbda, context_length, num_filters, num_scales, scale_min, scale_max):
        super().__init__()
        self.lmbda = lmbda
        self.num_scales = num_scales
        offset = tf.math.log(scale_min)
        factor = (tf.math.log(scale_max) - tf.math.log(scale_min)) / (
            num_scales - 1.)
        self.context_model = ContextModel(context_length=context_length, num_filters=num_filters)
        self.scale_fn = lambda i: tf.math.exp(offset + factor * i)
        self.hyperprior = tfc.NoisyDeepFactorized(batch_shape=(num_filters,))
        self.analysis_transform = AnalysisTransform(context_length=context_length, num_filters=num_filters)
        self.hyper_analysis_transform = HyperAnalysisTransform(context_length=context_length, num_filters=num_filters)
        self.hyper_synthesis_transform = HyperSynthesisTransform(context_length=context_length, num_filters=num_filters)
        self.synthesis_transform = SynthesisTransform(context_length=context_length, num_filters=num_filters)
        self.entropy_model = tfc.LocationScaleIndexedEntropyModel(
            tfc.NoisyNormal, self.num_scales, self.scale_fn, coding_rank=3,
            compression=False)
        self.side_entropy_model = tfc.ContinuousBatchedEntropyModel(
            self.hyperprior, coding_rank=3, compression=False)

    def call(self, inputs, training):
        """Computes rate and distortion losses."""
        x, x_c = inputs
        c = self.context_model(x_c)
        batch_dim = tf.shape(x)[0]
        c_tiled = tf.tile(tf.expand_dims(c, axis=0), [batch_dim, 1])
        y = self.analysis_transform([x, c_tiled])
        z = self.hyper_analysis_transform([abs(y), c_tiled])
        z_hat, side_bits = self.side_entropy_model(z, training=training)
        indexes = self.hyper_synthesis_transform([z_hat, c_tiled])
        y_hat, bits = self.entropy_model(y, indexes, training=training)
        x_hat = self.synthesis_transform([y_hat, c_tiled])

        # Total number of bits divided by total number of pixels.
        num_pixels = tf.cast(tf.reduce_prod(tf.shape(x)[:-1]), bits.dtype)
        bpp = (tf.reduce_sum(bits) + tf.reduce_sum(side_bits)) / num_pixels
        # Mean squared error across pixels.
        mse = tf.reduce_mean(tf.math.squared_difference(x, x_hat))
        mse *= 255 ** 2
        # The rate-distortion Lagrangian.
        loss = bpp + self.lmbda * mse
        return loss, bpp, mse

    def compile(self, **kwargs):
        super().compile(
            loss=None,
            metrics=None,
            loss_weights=None,
            weighted_metrics=None,
            **kwargs,
        )
        self.loss = tf.keras.metrics.Mean(name="loss")
        self.bpp = tf.keras.metrics.Mean(name="bpp")
        self.mse = tf.keras.metrics.Mean(name="mse")

    def train_step(self, inputs, var_sub=None):
        with tf.GradientTape() as tape:
            loss, bpp, mse = self(inputs, training=True)
        if var_sub: # include a subset of trainable variables
            variables = [var for var in self.trainable_variables if any(i in var.name for i in var_sub)]
        else: # include all trainable variables
            variables = self.trainable_variables
        gradients = tape.gradient(loss, variables)
        self.optimizer.apply_gradients(zip(gradients, variables))
        self.loss.update_state(loss)
        self.bpp.update_state(bpp)
        self.mse.update_state(mse)
        return {m.name: m.result() for m in [self.loss, self.bpp, self.mse]}

    @tf.function(input_signature=[
      tf.TensorSpec(shape=(None, None, None, 3), dtype=tf.float32),
      tf.TensorSpec(shape=(None, None, None, 3), dtype=tf.float32)
    ])
    def evaluate_model(self, x_c, x):
        '''
        Should compute the same as call with training=False, but can take any input size unlike call
        '''
        c = self.context_model(x_c)
        batch_dim = tf.shape(x)[0]
        c_tiled = tf.tile(tf.expand_dims(c, axis=0), [batch_dim, 1])
        y = self.analysis_transform([x, c_tiled])
        z = self.hyper_analysis_transform([abs(y), c_tiled])
        z_hat, side_bits = self.side_entropy_model(z, training=False)
        indexes = self.hyper_synthesis_transform([z_hat, c_tiled])
        y_hat, bits = self.entropy_model(y, indexes, training=False)
        x_hat = self.synthesis_transform([y_hat, c_tiled])

        # Total number of bits divided by total number of pixels.
        num_pixels = tf.cast(tf.reduce_prod(tf.shape(x)[:-1]), bits.dtype)
        bpp = (tf.reduce_sum(bits) + tf.reduce_sum(side_bits)) / num_pixels
        # Mean squared error across pixels.
        mse = tf.reduce_mean(tf.math.squared_difference(x, x_hat))
        mse *= 255 ** 2
        # The rate-distortion Lagrangian.
        loss = bpp + self.lmbda * mse
        return x_hat, {'loss': loss, 'bpp': bpp, 'mse': mse}
