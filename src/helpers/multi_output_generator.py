
import numpy as np

class MultiOutputGenerator():

    def __init__(self,
                 generator,
                 dataframe,
                 directory=None,
                 image_data_generator=None,
                 x_col="filename",
                 y_col="class",
                 weight_col=None,
                 target_size=(256, 256),
                 color_mode='rgb',
                 classes=None,
                 class_mode='categorical',
                 batch_size=32,
                 shuffle=False,
                 seed=None,
                 data_format='channels_last',
                 save_to_dir=None,
                 save_prefix='',
                 save_format='png',
                 subset=None,
                 interpolation='nearest',
                 dtype='float32',
                 validate_filenames=True):
        'Initialization'

        self.keras_generator = generator.flow_from_dataframe(
                 dataframe,
                 directory=directory,
                 image_data_generator=image_data_generator,
                 x_col=x_col,
                 y_col=y_col,
                 weight_col=weight_col,
                 target_size=target_size,
                 color_mode=color_mode,
                 classes=classes,
                 class_mode=class_mode,
                 batch_size=batch_size,
                 shuffle=shuffle,
                 seed=seed,
                 data_format=data_format,
                 save_to_dir=save_to_dir,
                 save_prefix=save_prefix,
                 save_format=save_format,
                 subset=subset,
                 interpolation=interpolation,
                 dtype=dtype,
                 validate_filenames=validate_filenames
        )

    def getGenerator(self):

        while True:

            gnext = self.keras_generator.next()
            y = np.float32(gnext[1])
            yield (gnext[0],{'disease': y[:,[0,1,2]], 'severity':  y[:, [3,4,5,6,7]]})


    def getNext(self):
        return self.keras_generator.next()

    @property
    def labels(self):
        if self.keras_generator.class_mode in {"multi_output", "raw"}:
            return self.keras_generator._targets
        else:
            return self.keras_generator.classes

    @property
    def sample_weight(self):
        return self.keras_generator._sample_weight

    @property
    def filepaths(self):
        return self.keras_generator._filepaths


