import tensorflow as tf
from tensorflow.keras import layers, Model


class ParallelNetwork:

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
        self.branch_names = ('branch1', 'branch2')
        self.input_names = ('input1', 'input2')
        self.augm_names = ('augment1', 'augment2')

    def build(self):
        """Return the model."""
        # Augmentations.
        augm1 = self.augment(0)
        augm2 = self.augment(1)
        # Convolutional bases.
        conv1 = self.branch(0)
        conv2 = self.branch(1)
        # When conv1 and conv2 are called, the training attribute
        # (not the same as trainable!) is set to False just in case
        # the base model has BatchNormalization layer. For more
        # information about that, see
        # https://www.tensorflow.org/guide/keras/transfer_learning.

        # The first track:
        input1 = layers.Input(self.input_shape, name=self.input_names[0])
        augm1_output = augm1(input1)
        conv1_output = conv1(augm1_output, training=False)

        # The second track:
        input2 = layers.Input(self.input_shape, name=self.input_names[1])
        augm2_output = augm2(input2)
        conv2_output = conv2(augm2_output, training=False)

        # Concatenate and classify.
        combined = layers.Concatenate(axis=-1)([conv1_output, conv2_output])
        x = layers.GlobalAveragePooling2D()(combined)
        x = layers.Dropout(self.dropout)(x)
        output_layer = layers.Dense(self.classes, activation='softmax')(x)
        model = Model(inputs=(input1, input2), outputs=output_layer)
        return model

    def augment(self, branch_id):
        input_ = layers.Input(self.input_shape)
        x = layers.RandomFlip(mode='horizontal')(input_)
        x = layers.RandomTranslation(0, .05)(x)
        x = layers.RandomRotation(.01)(x)
        x = layers.RandomZoom(.05, .05)(x)
        model_augm = Model(inputs=input_,
                           outputs=x,
                           name=self.augm_names[branch_id])
        return model_augm

    def branch(self, branch_id):
        """Return the branch as a model."""
        input_ = layers.Input(self.input_shape)
        preprocess, base = self.base_model()
        x = preprocess(input_)
        x = base(x)
        model_branch = Model(inputs=input_,
                             outputs=x,
                             name=self.branch_names[branch_id])
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
