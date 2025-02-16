import os
import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
from pathlib import Path
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TrainingConfig:
    IMG_HEIGHT = 224
    IMG_WIDTH = 224
    BATCH_SIZE = 32
    EPOCHS = 30
    LEARNING_RATE = 0.0001
    FINE_TUNING_LEARNING_RATE = 0.00001
    NUM_CLASSES = 3
    
    def __init__(self, train_dir, val_dir):
        self.TRAIN_DIR = Path(train_dir)
        self.VAL_DIR = Path(val_dir)

def create_data_generators():
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=30,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest',
        validation_split=0.2  # Added validation split
    )

    val_datagen = ImageDataGenerator(rescale=1./255)
    
    return train_datagen, val_datagen

def create_model(config):
    base_model = VGG16(
        weights='imagenet',
        include_top=False,
        input_shape=(config.IMG_HEIGHT, config.IMG_WIDTH, 3)
    )
    base_model.trainable = False

    model = Sequential([
        Input(shape=(config.IMG_HEIGHT, config.IMG_WIDTH, 3)),
        base_model,
        Flatten(),
        Dense(256, activation='relu', kernel_regularizer=l2(0.01)),
        Dropout(0.5),
        Dense(config.NUM_CLASSES, activation='softmax')
    ])
    
    return model, base_model

def get_callbacks(model_path):
    return [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=model_path,
            monitor='val_accuracy',
            save_best_only=True,
            mode='max'
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=4,
            min_lr=1e-6
        ),
        tf.keras.callbacks.TensorBoard(
            log_dir='./logs',
            histogram_freq=1
        )
    ]

def train_model(config, model_path='best_model.keras'):
    logger.info("Starting model training...")
    
    # Create generators
    train_datagen, val_datagen = create_data_generators()
    
    # Set up data generators
    train_generator = train_datagen.flow_from_directory(
        config.TRAIN_DIR,
        target_size=(config.IMG_HEIGHT, config.IMG_WIDTH),
        batch_size=config.BATCH_SIZE,
        class_mode='categorical'
    )

    validation_generator = val_datagen.flow_from_directory(
        config.VAL_DIR,
        target_size=(config.IMG_HEIGHT, config.IMG_WIDTH),
        batch_size=config.BATCH_SIZE,
        class_mode='categorical'
    )
    
    # Create and compile model
    model, base_model = create_model(config)
    model.compile(
        optimizer=Adam(learning_rate=config.LEARNING_RATE),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Train the model
    history = model.fit(
        train_generator,
        epochs=config.EPOCHS,
        validation_data=validation_generator,
        callbacks=get_callbacks(model_path)
    )
    
    # Fine-tuning
    logger.info("Starting fine-tuning...")
    for layer in base_model.layers[-4:]:
        layer.trainable = True
    
    model.compile(
        optimizer=Adam(learning_rate=config.FINE_TUNING_LEARNING_RATE),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    fine_tune_history = model.fit(
        train_generator,
        epochs=10,
        validation_data=validation_generator,
        callbacks=get_callbacks(model_path.replace('.keras', '_fine_tuned.keras'))
    )
    
    return model, history, fine_tune_history

if __name__ == '__main__':
    config = TrainingConfig(
        train_dir=r"C:\Catvsdog\Catsdogs_dataset\Train",
        val_dir=r"C:\Catvsdog\Catsdogs_dataset\Val"
    )
    
    model, history, fine_tune_history = train_model(config)
    logger.info("Training completed successfully") 