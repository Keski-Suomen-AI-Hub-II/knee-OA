import tensorflow as tf
from tensorflow.keras import layers, Model


class SingleNetwork:

    def __init__(self,
                 input_shape,
                 base_modelname,
                 classes=5,
                 weights='imagenet',
                 dropout=0):
        self.input_shape = input_shape
        self.base_modelname = base_modelname
        self.classes = classes
        self.weights = weights
        self.dropout = dropout

    def build(self):
        """Return the model."""
        input_ = layers.Input(self.input_shape)

        # Augmentations.
        x = self.augment(input_)

        # Convolutional layers.
        preprocess, conv = self.base_model()
        x = preprocess(x)
        x = conv(x, training=False)

        # Pool, classify, and return the model.
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dropout(self.dropout)(x)
        output_ = layers.Dense(self.classes, activation='softmax')(x)
        model = Model(inputs=input_, outputs=output_)
        return model

    def base_model(self):
        """Return the proprocessor and the base model without top layers."""
        name = self.base_modelname
        if name == 'vgg-16':
            preprocessor = tf.keras.applications.vgg16.preprocess_input
            base = tf.keras.applications.vgg16.VGG16
        elif name == 'vgg-19':
            preprocessor = tf.keras.applications.vgg19.preprocess_input
            base = tf.keras.applications.vgg19.VGG19
        elif name == 'inception_v3':
            preprocessor = tf.keras.applications.inception_v3.preprocess_input
            base = tf.keras.applications.inception_v3.InceptionV3
        elif name == 'xception':
            preprocessor = tf.keras.applications.xception.preprocess_input
            base = tf.keras.applications.xception.Xception
        elif name == 'densenet-121':
            preprocessor = tf.keras.applications.densenet.preprocess_input
            base = tf.keras.applications.densenet.DenseNet121
        elif name == 'densenet-169':
            preprocessor = tf.keras.applications.densenet.preprocess_input
            base = tf.keras.applications.densenet.DenseNet169
        elif name == 'mobilenet_v2':
            preprocessor = tf.keras.applications.mobilenet_v2.preprocess_input
            base = tf.keras.applications.mobilenet_v2.MobileNetV2
        elif name == 'resnet-50':
            preprocessor = tf.keras.applications.resnet.preprocess_input
            base = tf.keras.applications.resnet.ResNet50
        elif name == 'resnet-101':
            preprocessor = tf.keras.applications.resnet.preprocess_input
            base = tf.keras.applications.resnet.ResNet101
        else:
            return None
        base = base(include_top=False,
                    weights=self.weights,
                    input_shape=self.input_shape)
        return preprocessor, base

    def augment(self, input_):
        x = layers.RandomFlip(mode='horizontal')(input_)
        x = layers.RandomTranslation(.1, .1)(x)
        x = layers.RandomRotation(.1)(x)
        x = layers.RandomZoom(.2, .2)(x)
        x = layers.RandomContrast(.1, .1)(x)
        return x
