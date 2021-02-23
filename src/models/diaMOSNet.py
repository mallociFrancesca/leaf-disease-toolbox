import os
import keras
import datetime
import pandas as pd
from helpers.metrics import TimeHistory
from models.classifiers import cnn_model

SAVE_PATH = './out/diaMOSPlant/'

class diaMOSNet(object):

    def __init__(self, name='diaMOSNet', conv_base='resNet50', shape=(224,224,3)):
        self.name = name
        self.conv_base = conv_base
        self.shape = shape
        self.model = None


    def build(self, loss='categorical_crossentropy', metrics='accuracy', lr=2e-5, momentum=0.9):

        print("\nBuilding diaMOSNet with {} as conv base.\n".format(self.conv_base))

        network = cnn_model(self.conv_base)
        leaf_img = keras.layers.Input(shape=self.shape, name=self.name)

        x = network(leaf_img)
        x = keras.layers.GlobalAveragePooling2D(name="avg_pool")(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Flatten()(x)

        x = keras.layers.Dense(1024, activation='relu')(x)
        x = keras.layers.Dropout(0.5)(x)
        leaf_biotic_stress = keras.layers.Dense(3, activation="softmax", name="disease")(x)

        x = keras.layers.Dense(1024, activation='relu')(x)
        x = keras.layers.Dropout(0.5)(x)
        leaf_severity = keras.layers.Dense(5, activation="softmax", name="severity")(x)

        self.model = keras.models.Model(leaf_img, [leaf_biotic_stress,leaf_severity])

        self.model.summary()

        self.model.compile(keras.optimizers.RMSprop(learning_rate=lr, momentum=momentum), loss={'disease': loss,'severity': loss}, metrics={'disease': metrics,'severity': metrics})



    def train(self, train_generator, steps_per_epoch, val_generator,validation_steps, epochs=100,verbose=1,save_hist=True):

        folder_name = self.conv_base + "/" + datetime.datetime.now().strftime('%d_%m_%Y') + '/train/cp/'
        save_path = os.path.join(SAVE_PATH, folder_name)

        if not(os.path.exists(save_path)):
            os.makedirs(save_path)


        time_history = TimeHistory()

        plateau = keras.callbacks.ReduceLROnPlateau(
             monitor='val_loss',
             factor=0.3,
             patience=3,
             min_lr=0.000001
        )

        early_stopping = keras.callbacks.EarlyStopping(

            monitor='severity_accuracy',
            patience=10

        )

        model_checkpoint = keras.callbacks.ModelCheckpoint(

            filepath=os.path.join(save_path,'weights-{epoch:02d}-{val_loss:.2f}.h5'),
            monitor='val_loss',
            save_best_only = True,
            save_weights_only=True
        )

        callbacks = [time_history,plateau,early_stopping,model_checkpoint]



        history = self.model.fit(train_generator.getGenerator(),
                       steps_per_epoch =steps_per_epoch,
                       epochs = epochs,
                       callbacks = callbacks,
                       validation_data = val_generator.getGenerator(),
                       validation_steps = validation_steps,
                       verbose=verbose)


        if save_hist:
            history_folder_path = SAVE_PATH + '/' + self.conv_base + "/" + datetime.datetime.now().strftime('%d_%m_%Y') + '/train/'
            if not(os.path.exists(history_folder_path)):
                os.makedirs(history_folder_path)

            history_filename = self.conv_base + '_history.csv'
            history_save_path = os.path.join(history_folder_path, history_filename)

            df = pd.DataFrame(history.history)
            df.to_csv(history_save_path, index=False)
            print("Saved history to {}".format(history_save_path))


        time_filename = self.conv_base + '_time_callback.csv'
        time_folder_path = SAVE_PATH + '/' + self.conv_base + "/" + datetime.datetime.now().strftime('%d_%m_%Y') + '/train/'

        if not(os.path.exists(time_folder_path)):
            os.makedirs(time_folder_path)

        time_history.save(os.path.join(time_folder_path,time_filename))





    def test(self,test_generator, steps, batch_size = 1,verbose=1):

         print("Testing {} network...".format(self.conv_base))

         y_pred = self.model.predict(
                test_generator,
                batch_size = batch_size,
                steps=steps,
                verbose= verbose,
         )

         print("Predicted labels: {} \n".format(len(y_pred[0].argmax(axis=1))))
         print("Actual labels: {} \n".format(len(test_generator.labels[:,[0,1,2]].argmax(axis=1))))

         return y_pred


    def save(self, filename):

        arch_filename = self.conv_base + filename
        arch_save_path = SAVE_PATH + '/' + self.conv_base + "/" + datetime.datetime.now().strftime('%d_%m_%Y') + '/train/'

        if not(os.path.exists(arch_save_path)):
            os.makedirs(arch_save_path)

        print("Saving architecture to {}".format(arch_save_path))

        self.model.save(os.path.join(arch_save_path, arch_filename))


    def save_weights(self,filename):


        weights_filename = self.conv_base + filename
        weights_save_path = SAVE_PATH + '/' + self.conv_base + "/" + datetime.datetime.now().strftime('%d_%m_%Y') + '/train/'

        if not(os.path.exists(weights_save_path)):
            os.makedirs(weights_save_path)

        print("Saving weights to {}".format(weights_save_path))

        self.model.save_weights(os.path.join(weights_save_path, weights_filename))




    def load_model(self, path_arch):
        self.model = keras.models.load_model(path_arch)

    def load_weights(self, path_weights, by_name = True, skip_mismatch=True):
        self.model.load_weights(path_weights, by_name = by_name, skip_mismatch=skip_mismatch)
