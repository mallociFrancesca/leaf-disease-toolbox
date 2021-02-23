import os
import datetime
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
BASE_PATH = './out/diaMOSPlant/'
#networks = ['vgg16','vgg19','resNet50','inceptionV3', 'mobileNetV2', 'efficientNetB0']
#DATE = datetime.datetime.now().strftime('%d_%m%_Y')
DATE = "04022021_0731"
networks = ['vgg16']

if __name__ == "__main__":

    test_folder = BASE_PATH + '/csv/' + DATE + '/'
    test_path = os.path.join(test_folder,'test.csv')

    test = load_data(path_csv=test_path, sep=',')



    for net in networks:

        out_path = BASE_PATH + net + "/" + DATE + "/test/"

        test_datagen = ImageDataGenerator(
            rescale=1./255,
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

        arch_filename = net + '_arch.h5'
        model.load_model(path_arch=os.path.join(out_path,arch_filename))

        weights_filename = net + '_weights.h5'
        model.load_weights(path_weights=os.path.join(out_path,weights_filename), by_name = True, skip_mismatch=True)


        y_pred = model.test(test_generator= test_generator.getGenerator(),
                            batch_size = 1,
                            steps=int(len(test)/1),
                            verbose=1)


        print("\nCM - Biotic Stress:\n")
        compute_confusion_matrix(actual_labels=test_generator.labels[:,[0,1,2]].argmax(axis=1),
                                 predicted_labels=y_pred[0].argmax(axis=1),
                                 labels=[0,1,2],
                                 name_labels=['healthy', 'slug', 'spot'],
                                 save_to= out_path + + net + '_biotic_stress.png'
        )


        print("\nCM - Severity:\n")

        compute_confusion_matrix(actual_labels = test_generator.labels[:,[3,4,5,6,7]].argmax(axis=1),
                                 predicted_labels=y_pred[1].argmax(axis=1),
                                 labels=[0,1,2,3,4],
                                 name_labels= ['no risk', 'very low', 'low', 'medium','high'],
                                 save_to= out_path + net + '_severity.png'
        )


        print("Report - Biotic Stress:\n")
        compute_classification_report(actual_labels = test_generator.labels[:,[0,1,2]].argmax(axis=1),
                                      predicted_labels=y_pred[0].argmax(axis=1),
                                      save_to=out_path + net + '_report_biotic_stress.csv'
        )

        print("Report - Severity: \n")
        compute_classification_report(actual_labels = test_generator.labels[:,[3,4,5,6,7]].argmax(axis=1),
                                      predicted_labels=y_pred[1].argmax(axis=1),
                                      save_to= out_path + net + '_report_severity.csv'
        )






