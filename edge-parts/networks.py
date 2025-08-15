import torch.nn as nn
import torchvision.models as models


class ConvNet(nn.Module):
    def __init__(self, base_model, n_classes):
        super().__init__()
        self.base_model = base_model
        self.n_classes = n_classes
        self.select_model()

    def forward(self, x):
        return self.net(x).squeeze(-1)

    def select_model(self):
        """Return a model with custom classification head."""
        # Select the right model.
        if self.base_model == 'alexnet':
            self.weights = models.AlexNet_Weights.DEFAULT
            net = models.alexnet
        elif self.base_model == 'densenet-121':
            self.weights = models.DenseNet121_Weights.DEFAULT
            net = models.densenet121
        elif self.base_model == 'mobilenet_v3':
            self.weights = models.MobileNet_V3_Large_Weights.DEFAULT
            net = models.mobilenet_v3_large
        elif self.base_model == 'resnet-50':
            self.weights = models.ResNet50_Weights.DEFAULT
            net = models.resnet50
        elif self.base_model == 'vgg-16':
            self.weights = models.VGG16_Weights.DEFAULT
            net = models.vgg16
        elif self.base_model == 'inception_v3':
            self.weights = models.Inception_V3_Weights.DEFAULT
            net = models.inception_v3
        else:
            raise ValueError('The base_model value is not recognized')

        # Load the weights.
        self.net = net(weights=self.weights)

        # Define the classifier head according to the number of classes.
        if self.n_classes == 2:
            last_fc = nn.LazyLinear(1)
        else:
            last_fc = nn.LazyLinear(self.n_classes)

        # Ensure that every kind of network has the same kind of
        # end: average pooling, dropout, and fully connected layer.
        if self.base_model == 'alexnet':
            self.net.avgpool = nn.Sequential(
                nn.AdaptiveAvgPool2d((1, 1)), nn.Dropout())
            self.net.classifier = last_fc
        elif self.base_model == 'densenet-121':
            self.net.classifier = nn.Sequential(nn.Dropout(), last_fc)
        elif self.base_model == 'mobilenet_v3':
            self.net.avgpool.add_module('dropout', nn.Dropout())
            self.net.classifier = last_fc
        elif self.base_model == 'resnet-50':
            self.net.avgpool = nn.Sequential(
                nn.AdaptiveAvgPool2d((1, 1)), nn.Dropout())
            self.net.fc = last_fc
        elif self.base_model == 'vgg-16':
            self.net.avgpool = nn.Sequential(
                nn.AdaptiveAvgPool2d((1, 1)), nn.Dropout())
            self.net.classifier = last_fc
        else:  # 'inception_v3'
            self.net.fc = last_fc
            self.net.aux_logits = False

    def preprocessor(self):
        """Return the preprocessing transforms specific to the model."""
        return self.weights.transforms()

    def target_layer_for_gradcam(self):
        """Return the last convolutional layer."""
        if self.base_model == 'alexnet':
            return self.net.features[-3]
        elif self.base_model == 'densenet-121':
            return self.net.features.denseblock4.denselayer16.conv2
        elif self.base_model == 'mobilenet_v3':
            return self.net.features[-1][-3]
        elif self.base_model == 'resnet-50':
            return self.net.layer4[-1].conv3
        elif self.base_model == 'vgg-16':
            return self.net.features[-3]
        else:  # 'inception_v3':
            return self.net.Mixed_7c.branch_pool.conv

    def resizing_and_cropping_dims(self):
        """Return information about model-specific preprocessing."""
        if self.base_model == 'mobilenet_v3' or self.base_model == 'resnet-50':
            return 232, 224
        elif self.base_model == 'inception_v3':
            return 342, 299
        else:
            return 256, 224
