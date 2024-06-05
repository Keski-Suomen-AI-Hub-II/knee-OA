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
        # The names 'branch1', 'input1', and 'augment1' are as in
        # the parallel network, although this network only has one
        # branch.
        self.branch_name = 'branch1'
        self.input_name = 'input1'
        self.augm_name = 'augment1'

    def build(self):
        """Return the model."""
        # Augmentations.
        augm = self.augment()

        # Convolutional layers.
        conv = self.branch()

        # Input, augmentations, and convs.
        input_ = layers.Input(self.input_shape, name=self.input_name)
        augm_output = augm(input_)
        conv_output = conv(augm_output, training=False)

        # Pool, classify, and return the model.
        x = layers.GlobalAveragePooling2D()(conv_output)
        x = layers.Dropout(self.dropout)(x)
        output_ = layers.Dense(self.classes, activation='softmax')(x)
        model = Model(inputs=input_, outputs=output_)
        return model

    def augment(self):
        input_ = layers.Input(self.input_shape)
        x = layers.RandomTranslation(.05, .05)(input_)
        x = layers.RandomRotation(.01)(x)
        x = layers.RandomZoom(.05, .05)(x)
        model_augm = Model(inputs=input_, outputs=x, name=self.augm_name)
        return model_augm

    def branch(self):
        """Return the convolutional layers ('branch') as a model."""
        input_ = layers.Input(self.input_shape)
        preprocess, base = self.base_model()
        x = preprocess(input_)
        x = base(x)
        model_branch = Model(inputs=input_, outputs=x, name=self.branch_name)
        return model_branch

    def base_model(self):
        """Return the preprocessor and the base model without top layers."""
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
        else:
            return None
        base = base(include_top=False,
                    weights=self.weights,
                    input_shape=self.input_shape)
        return preprocessor, base
