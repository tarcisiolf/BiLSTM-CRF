from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from model import bilstm_crf, CRF

def train(hyperparams, n_words, n_tags, text_sequences, tag_sequences_categorical):
    # Criar o modelo
    model = bilstm_crf(
        maxlen=hyperparams['max_len'], 
        n_tags=n_tags, 
        lstm_units=hyperparams['lstm_units'], 
        embedding_dim=hyperparams['embedding_dim'], 
        n_words=n_words, 
        mask_zero=True
    )

    model.summary()

    # Compilar o modelo
    model.compile(
        optimizer=Adam(learning_rate=hyperparams['learning_rate']), 
        loss=model.layers[-1].loss, 
        metrics=model.layers[-1].accuracy
    )

    # Definir callbacks
    callbacks = [
        ReduceLROnPlateau(monitor='loss', factor=0.1, patience=5, verbose=1),
        EarlyStopping(monitor='loss', min_delta=0, patience=5, verbose=1)
    ]

    # Treinar o modelo
    model.fit(
        text_sequences, 
        tag_sequences_categorical, 
        epochs=hyperparams['max_epochs'], 
        callbacks=callbacks, 
        verbose=1, 
        shuffle=True
    )

    return model
