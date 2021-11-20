from torchvision import models
from torch import nn


def set_parameter_requires_grad(model, freeze):
    for param in model.parameters():
        # if freeze equals true, the below will be false
        param.requires_grad = not freeze


def get_model(config):
    # You can also refer: 
    # - https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html
    # - https://pytorch.org/tutorials/beginner/finetuning_torchvision_models_tutorial.html
    model = None
    input_size = 0

    if config.model_name == "resnet":
        """ Resnet34
        """
        # if true(config.use_pretrained), trained w loaded
        model = models.resnet34(pretrained=config.use_pretrained)
        set_parameter_requires_grad(model, config.freeze)

        
        n_features = model.fc.in_features
        # 2 classes
        model.fc = nn.Linear(n_features, config.n_classes)
        input_size = 224
    elif config.model_name == "alexnet":
        """ Alexnet
        """
        model = models.alexnet(pretrained=config.use_pretrained)
        set_parameter_requires_grad(model, config.freeze)

        n_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(n_features, config.n_classes)
        input_size = 224
    elif config.model_name == "vgg":
        """ VGG16_bn
        """
        model = models.vgg16_bn(pretrained=config.use_pretrained)
        set_parameter_requires_grad(model, config.freeze)

        n_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(n_features, config.n_classes)
        input_size = 224
    elif config.model_name == "squeezenet":
        """ Squeezenet
        """
        model = models.squeezenet1_0(pretrained=config.use_pretrained)
        set_parameter_requires_grad(model, config.freeze)

        model.classifier[1] = nn.Conv2d(
            512,
            config.n_classes,
            kernel_size=(1, 1),
            stride=(1, 1),
        )
        model.n_classes = config.n_classes
        input_size = 224
    elif config.model_name == "densenet":
        """ Densenet
        """
        model = models.densenet121(pretrained=config.use_pretrained)
        set_parameter_requires_grad(model, config.freeze)

        n_features = model.classifier.in_features
        model.classifier = nn.Linear(n_features, config.n_classes)
        input_size = 224
    else:
        raise NotImplementedError('You need to specify model name.')

    return model, input_size
