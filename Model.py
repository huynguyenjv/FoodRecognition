import tensorflow as tf
import tensorflow.python.keras.backend as K
from tensorflow.python.keras import regularizers
from tensorflow.python.keras.applications.mobilenet_v2 import MobileNetV2
from tensorflow.python.keras.callbacks import ModelCheckpoint, CSVLogger
from tensorflow.python.keras.layers import Dense, Dropout
from tensorflow.python.keras.layers import GlobalAveragePooling2D
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.optimizers import SGD
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator

print(tf.__version__)
print(tf.test.gpu_device_name())

K.clear_session()

n_classes = 101
img_width, img_height = 299, 299
train_data_dir = 'food-101/train'
validation_data_dir = 'food-101/test'
nb_train_samples = 75750
nb_validation_samples = 25250
batch_size = 20

train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical')

validation_generator = test_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical')

mbv2 = MobileNetV2(weights='imagenet', include_top=False, input_shape=(299, 299, 3))
x = mbv2.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.2)(x)

predictions = Dense(101, kernel_regularizer=regularizers.l2(0.005), activation='softmax')(x)

model = Model(inputs=mbv2.input, outputs=predictions)
model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='categorical_crossentropy', metrics=['accuracy'])
checkpointer = ModelCheckpoint(filepath='best_model_3class_sept.hdf5', verbose=1, save_best_only=True)
csv_logger = CSVLogger('history.log')

history = model.fit_generator(train_generator,
                              steps_per_epoch=nb_train_samples // batch_size,
                              validation_data=validation_generator,
                              validation_steps=nb_validation_samples // batch_size,
                              epochs=10,
                              verbose=1,
                              callbacks=[csv_logger, checkpointer])

model.save('model_trained.h5')

