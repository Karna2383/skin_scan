import numpy as np
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D, Flatten, concatenate

def create_model() -> Model:
    # Image Branch
    image_input = Input(shape=(96, 96, 3))
    cnn = Conv2D(16, (3, 3), activation='relu')(image_input)
    cnn = MaxPooling2D(pool_size=(2, 2))(cnn)
    cnn = MaxPooling2D(2, 2)(cnn)
    cnn = Conv2D(32, (3, 3), activation='relu')(cnn)
    cnn = MaxPooling2D(2, 2)(cnn)
    cnn = Flatten()(cnn)
    cnn = Dense(64, activation='relu')(cnn)

    # Metadata Branch
    meta_input = Input(shape=(15,))
    dln = Dense(32, activation='relu')(meta_input)
    dln = Dense(16, activation='relu')(dln)

    # Combined Final Layers
    combined_layers = concatenate([dln, meta_input])
    HAM_combined_model = Dense(32)(combined_layers)
    HAM_combined_model = Dense(7, activation='softmax')(HAM_combined_model)

    return Model(inputs=[image_input, meta_input], outputs=HAM_combined_model)

def fit_model(X_images: np.array, X_metadata: np.array, y: np.array) -> Model:
    model = create_model()
    model.fit([X_images, X_metadata], y, epochs=20, batch_size=32, validation_split=0.2)
    return model

def predict(X_images: np.array, X_metadata: np.array, model: Model):
    prediction = model.predict([X_images, X_metadata])
    return prediction

