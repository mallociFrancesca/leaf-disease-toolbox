import keras
from keras.applications import VGG16, MobileNetV2, VGG19,InceptionV3, Xception
from keras.applications.resnet50 import ResNet50
from keras_efficientnets import EfficientNetB0


def cnn_model(network='resNet50', weights="imagenet", trainable=False, include_top=False):

    print("Getting {} as base network...\n".format(network))

    if network == 'vgg16':
        model = VGG16(weights=weights, include_top=include_top)
        if not trainable:
            model.trainable = False

    if network == 'vgg19':
        model = VGG19(weights=weights, include_top=include_top)
        if not trainable:
            model.trainable = False

    if network == 'resNet50':
        model = ResNet50(weights=weights, include_top=include_top)
        if not trainable:
            model.trainable = False

    if network == 'inceptionV3':
        model = InceptionV3(weights=weights, include_top=include_top)
        if not trainable:
            model.trainable = False

    if network == 'mobileNetV2':
        model = MobileNetV2(weights=weights, include_top=include_top)
        if not trainable:
            model.trainable = False

    if network == 'efficientNetB0':
        model = EfficientNetB0(weights=weights, include_top=include_top)
        if not trainable:
            model.trainable = False


    return model





def build_classifier(network ='resNet50', activation='sigmoid', loss='binary_crossentropy', num_classes=1, weights='imagenet', input_shape = (224, 224,3), lr=2e-5, momentum=0.9):

    conv_base = cnn_model(network=network, weights=weights)

    inputs = keras.layers.Input(shape=input_shape)
    x = conv_base(inputs)
    x = keras.layers.GlobalAveragePooling2D()(x)
    outputs = keras.layers.Dense(num_classes, activation=activation)(x)
    model = keras.models.Model(inputs, outputs)

    model.summary()

    model.compile(keras.optimizers.RMSprop(learning_rate=lr, momentum=momentum), loss=loss, metrics=['acc'])

    return model


