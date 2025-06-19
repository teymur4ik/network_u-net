from models.unet_model import unet_model
from utils.preprocessing import load_data
from utils.augmentation import create_augmentation
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import config

def train():
    # Загрузка данных
    train_images, train_masks = load_data(config.TRAIN_PATH, config.TRAIN_MASKS_PATH)
    val_images, val_masks = load_data(config.VAL_PATH, config.VAL_MASKS_PATH)
    
    # Создание модели
    model = unet_model(config.INPUT_SHAPE)
    model.compile(optimizer=Adam(learning_rate=config.LEARNING_RATE),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    
    # Аугментация данных
    train_generator = create_augmentation(train_images, train_masks, config.BATCH_SIZE)
    
    # Коллбэки
    callbacks = [
        ModelCheckpoint(config.MODEL_SAVE_PATH, save_best_only=True),
        EarlyStopping(patience=config.PATIENCE, restore_best_weights=True)
    ]
    
    # Обучение
    history = model.fit(
        train_generator,
        steps_per_epoch=len(train_images) // config.BATCH_SIZE,
        epochs=config.EPOCHS,
        validation_data=(val_images, val_masks),
        callbacks=callbacks
    )
    
    return model, history

if __name__ == "__main__":
    model, history = train()