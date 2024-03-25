import tensorflow as tf
from tensorflow.keras import layers, Model


class ParallelNetwork:

    def __init__(self,
                 input_shape,
                 base_models,
                 classes=5,
                 weights=('imagenet', 'imagenet'),
                 dropout=0):
        self.input_shape = input_shape
        self.base_models = base_models
        self.classes = classes
        self.weights = weights
        self.dropout = dropout
        self.branch_names = ('branch1', 'branch2')
        self.input_names = ('input1', 'input2')

    def build(self):
        """Return the model."""
        # Branches are defined as model1 and model2.
        model1 = self.branch(0)
        model2 = self.branch(1)
        # Models model1 and model2 are then used as building blocks
        # for the final model. The training attribute (not the same as
        # trainable!) is set to False just in case any of the base models
        # has BatchNormalization layer. For more information about that, see
        # https://www.tensorflow.org/guide/keras/transfer_learning.
        input1 = layers.Input(self.input_shape, name=self.input_names[0])
        branch1_output = model1(input1, training=False)
        input2 = layers.Input(self.input_shape, name=self.input_names[1])
        branch2_output = model2(input2, training=False)
        combined = layers.Concatenate(axis=-1)(
            [branch1_output, branch2_output])
        x = layers.GlobalAveragePooling2D()(combined)
        x = layers.Dropout(self.dropout)(x)
        if self.classes == 2:
            output_layer = layers.Dense(1, activation='sigmoid')(x)
        else:
            output_layer = layers.Dense(self.classes, activation='softmax')(x)
        model = Model(inputs=(input1, input2), outputs=output_layer)
        return model

    def branch(self, branch_id):
        """Return the branch as a model."""
        input = layers.Input(self.input_shape)
        preprocess, base = self.base_model(branch_id)
        x = preprocess(input)
        x = base(x)
        model_branch = Model(inputs=input,
                             outputs=x,
                             name=self.branch_names[branch_id])
        return model_branch

    def base_model(self, branch_id):
        """Return the proprocessor and the base model without top layers."""
        name = self.base_models[branch_id]
        if name == 'vgg-19':
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
        elif name == 'densenet-201':
            preprocessor = tf.keras.applications.densenet.preprocess_input
            base = tf.keras.applications.densenet.DenseNet201
        elif name == 'resnet-50':
            preprocessor = tf.keras.applications.resnet50.preprocess_input
            base = tf.keras.applications.resnet50.ResNet50
        elif name == 'regnetx-002':
            preprocessor = tf.keras.applications.regnet.preprocess_input
            base = tf.keras.applications.regnet.RegNetX002
        else:
            return None
        base = base(include_top=False,
                    weights=self.weights[branch_id],
                    input_shape=self.input_shape)
        return preprocessor, base
