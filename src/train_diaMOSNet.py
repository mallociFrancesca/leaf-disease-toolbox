import os
import datetime
import pandas as pd
from keras.preprocessing.image import ImageDataGenerator
from helpers.dataset import *
from helpers.multi_output_generator import MultiOutputGenerator
from helpers.utils import *
from models.classifiers import *
from models.diaMOSNet import diaMOSNet


os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

#SETTINGS
BATCH_SIZE = 32
X_COL = "filename"
Y_COL = ['healthy', 'pear_slug', 'leaf_spot','severity_0','severity_1', 'severity_2', 'severity_3','severity_4']
IMG_W = 224
IMG_H = 224
DATA_DIR = "./data/diaMOSPlant/img/"
LABEL_DIR = "./data/diaMOSPlant/csv/"
OUT_PATH = './out/diaMOSPlant/'


networks = ['vgg16','vgg19','resNet50','inceptionV3', 'mobileNetV2', 'efficientNetB0']



if __name__ == "__main__":

    X_train, y_train, X_val, y_val, X_test, y_test = shuffle_split_data(os.path.join(LABEL_DIR,'diaMOSPlant.csv'),X_COL,Y_COL,sep=";",validation=True)

    train = pd.concat([X_train, y_train], axis=1, sort=False)
    val = pd.concat([X_val, y_val], axis=1, sort=False)
    test = pd.concat([X_test, y_test], axis=1, sort=False)

    data_save_path = OUT_PATH + '/csv/' + datetime.datetime.now().strftime('%d_%m%_Y')  + '/'
    if not(os.path.exists(data_save_path)):
        os.makedirs(data_save_path)

    df_train = pd.DataFrame(train)
    df_val = pd.DataFrame(val)
    df_test = pd.DataFrame(test)

    df_train.to_csv(os.path.join(data_save_path,'train.csv'), index=False)
    df_val.to_csv(os.path.join(data_save_path,'val.csv'), index=False)
    df_test.to_csv(os.path.join(data_save_path,'test.csv'), index=False)




    for net in networks:

        #base_path = os.path.join(BASE_PATH, net)

        train_datagen = ImageDataGenerator(
            horizontal_flip=True,
            vertical_flip=True,
            rescale=1/255

        )

        val_datagen = ImageDataGenerator(
            rescale=1./255,
        )

        test_datagen = ImageDataGenerator(
            rescale=1./255,
        )

        train_generator = MultiOutputGenerator(
                        generator = train_datagen,
                        dataframe = train,
                        directory=DATA_DIR,
                        batch_size = BATCH_SIZE,
                        x_col = X_COL,
                        y_col = Y_COL,
                        class_mode="raw",
                        target_size = (IMG_H, IMG_W),
                        shuffle=False
        )

        val_generator = MultiOutputGenerator(
                        generator = val_datagen,
                        dataframe = val,
                        directory=DATA_DIR,
                        batch_size = BATCH_SIZE,
                        x_col = X_COL,
                        y_col = Y_COL,
                        class_mode="raw",
                        target_size = (IMG_H, IMG_W),
                        shuffle=False
        )


        test_generator = MultiOutputGenerator(
                        generator = test_datagen,
                        dataframe = test,
                        directory=DATA_DIR,
                        batch_size = 1,
                        x_col = X_COL,
                        y_col = Y_COL,
                        class_mode="raw",
                        target_size = (IMG_H, IMG_W),
                        shuffle=False
        )


        model = diaMOSNet(conv_base=net, shape=(IMG_H,IMG_W,3))

        model.build()

        model.train(train_generator = train_generator,
                    steps_per_epoch = int(len(train)/32),
                    epochs = 100,
                    val_generator = val_generator,
                    validation_steps = int(len(val)/32),
                    verbose = 1)


        model.save_weights('_weights.h5')

        model.save('_arch.h5')
