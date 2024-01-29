import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.regularizers import L2


class ParallelNetwork:

    def __init__(self,
                 input_shape,
                 base_models,
                 weights='imagenet',
                 input_names=('input1', 'input2'),
                 dropout=.2):
        self.input_shape = input_shape
        self.base_models = base_models
        self.weights = weights
        self.input_names = input_names
        self.dropout = dropout

    def build(self):
        """Return the model."""
        # Branches are defined as model1 and model2.
        model1 = self.branch('branch1', 0)
        model2 = self.branch('branch2', 1)
        # Models model1 and model2 are then used as building blocks
        # for the final model.
        input1 = layers.Input(self.input_shape, name=self.input_names[0])
        branch1_output = model1(input1)
        input2 = layers.Input(self.input_shape, name=self.input_names[1])
        branch2_output = model2(input2)
        combined = layers.Concatenate(axis=-1)(
            [branch1_output, branch2_output])
        #x = GlobalAveragePooling2D()(combined)
        x = layers.Flatten()(combined)
        #x = Dense(4096, activation='relu')(x)
        x = layers.Dense(1024, activation='tanh',
                         kernel_regularizer=L2(.04))(x)
        x = layers.Dropout(self.dropout)(x)
        output_layer = layers.Dense(5,
                                    activation='softmax',
                                    kernel_regularizer=L2(.04))(x)
        model = Model(inputs=(input1, input2), outputs=output_layer)
        return model

    def branch(self, branch_name, no_of_branch):
        """Return the branch as a model."""
        input = layers.Input(self.input_shape)
        preprocess, base = self.base_model(no_of_branch)
        x = preprocess(input)
        x = base(x)
        #x1 = GlobalAveragePooling2D()(x1)
        model_branch = Model(inputs=input, outputs=x, name=branch_name)
        return model_branch

    def base_model(self, no_of_branch):
        name = self.base_models[no_of_branch]
        """Return the proprocessor and the base model, without top layers."""
        if name == 'vgg-19':
            preprocessor = tf.keras.applications.vgg19.preprocess_input
            base = tf.keras.applications.vgg19.VGG19
        elif name == 'inception_v3':
            preprocessor = tf.keras.applications.inception_v3.preprocess_input
            base = tf.keras.applications.inception_v3.InceptionV3
        elif name == 'xception':
            preprocessor = tf.keras.applications.xception.preprocess_input
            base = tf.keras.applications.xception.Xception
        else:
            return None
        base = base(include_top=False,
                    weights=self.weights,
                    input_shape=self.input_shape)
        base.trainable = False
        return preprocessor, base
