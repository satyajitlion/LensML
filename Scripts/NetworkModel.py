import tensorflow as tf
import numpy as np

def quantile_loss(q, y_true, y_pred):
    """
    Quantile (pinball) loss.
    q=0.5 gives median regression, reduces skew errors.
    """
    error = y_true - y_pred
    return tf.reduce_mean(tf.maximum(q * error, (q - 1) * error))

class MinimumEpochEarlyStopping(tf.keras.callbacks.Callback):
    """Custom early stopping that ensures minimum training epochs"""
    def __init__(self, min_epochs=10, patience=10, monitor='val_loss'):
        super().__init__()
        self.min_epochs = min_epochs
        self.patience = patience
        self.monitor = monitor
        self.best_weights = None
        self.best_epoch = 0
        self.best_value = float('inf')
        self.wait = 0
        
    def on_epoch_end(self, epoch, logs=None):
        current_value = logs.get(self.monitor)
        
        if epoch < self.min_epochs:
            return
        
        if current_value < self.best_value:
            self.best_value = current_value
            self.best_epoch = epoch
            self.best_weights = self.model.get_weights()
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.model.stop_training = True
                print(f"\nEarly stopping at epoch {epoch+1}")
                print(f"Restoring weights from epoch {self.best_epoch+1}")
                if self.best_weights is not None:
                    self.model.set_weights(self.best_weights)
    
    def on_train_end(self, logs=None):
        if self.best_weights is not None:
            self.model.set_weights(self.best_weights)

# ==================== MODEL CREATION (updated input shapes) ====================

def create_single_model(loss_type='mse', quantile=0.6):
    """Model for single image lenses (4 inputs: img(2) + z(2), outputs: pot(1)+deflec(2)=3)"""
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(4,)),

        tf.keras.layers.Dense(32),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.Dropout(0.0),

        tf.keras.layers.Dense(64),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.Dropout(0.0),

        tf.keras.layers.Dense(32),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.Dropout(0.0),

        tf.keras.layers.Dense(16),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.Dropout(0.0),

        tf.keras.layers.Dense(3, activation='linear')
    ])

    if loss_type == 'quantile':
        loss = lambda yt, yp: quantile_loss(quantile, yt, yp)
    else:
        loss = 'mse'

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
        loss=loss,
        metrics=['mae']
    )
    return model

def create_double_model(loss_type='mse', quantile=0.6):
    """Model for double image lenses (7 inputs: img(4) + time(1) + z(2), outputs: pot(2)+deflec(4)=6)"""
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(7,)),
        
        tf.keras.layers.Dense(32, use_bias=False),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.Dropout(0.0),
        
        tf.keras.layers.Dense(64, use_bias=False),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.Dropout(0.0),
        
        tf.keras.layers.Dense(6, activation='linear')
    ])

    if loss_type == 'quantile':
        loss = lambda yt, yp: quantile_loss(quantile, yt, yp)
    else:
        loss = 'mse'

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss=loss,
        metrics=['mae']
    )
    return model

def create_quad_model(loss_type='mse', quantile=0.6):
    """Model for quad image lenses (13 inputs: img(8) + time(3) + z(2), outputs: pot(4)+deflec(8)=12)"""
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(13,)),

        tf.keras.layers.Dense(64),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.Dropout(0.0),

        tf.keras.layers.Dense(128),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.Dropout(0.0),

        tf.keras.layers.Dense(64),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.Dropout(0.0),

        tf.keras.layers.Dense(32),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.Dropout(0.0),

        tf.keras.layers.Dense(12, activation='linear')
    ])

    if loss_type == 'quantile':
        loss = lambda yt, yp: quantile_loss(quantile, yt, yp)
    else:
        loss = 'mse'

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
        loss=loss,
        metrics=['mae']
    )
    return model

# ==================== TRAINING FUNCTIONS (updated shape checks) ====================

def train_model_with_roto_translation_singles(model, x, y, rotation_prob=0.3, translation_prob=0.3, translation_scale=0.1):
    if x.shape[1] != 4:
        raise ValueError(f"Single model expects 4 input features, got {x.shape[1]}")
    
    early_stopping = MinimumEpochEarlyStopping(min_epochs=10, patience=10, monitor='val_loss')
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6, verbose=1)
    
    history = model.fit(
        x=x, 
        y=y,
        validation_split=0.1,
        batch_size=16,
        epochs=100,
        shuffle=True, 
        verbose=1,
        callbacks=[early_stopping, reduce_lr]
    )
    
    print(f"\n=== TRAINING SUMMARY ===")
    print(f"Stopped at epoch: {len(history.history['loss'])}")
    print(f"Best validation loss: {min(history.history['val_loss']):.6f}")
    print(f"Final training loss: {history.history['loss'][-1]:.6f}")
    
    return history

def train_model_with_roto_translation_doubles(model, x, y, rotation_prob=0.5, translation_prob=0.5, translation_scale=0.1):
    if x.shape[1] != 7:
        raise ValueError(f"Double model expects 7 input features, got {x.shape[1]}")
    
    early_stopping = MinimumEpochEarlyStopping(min_epochs=5, patience=15, monitor='val_loss')
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=1e-7, verbose=1)
    
    history = model.fit(
        x=x, 
        y=y,
        validation_split=0.1,
        batch_size=64,
        epochs=200,
        shuffle=True, 
        verbose=1,
        callbacks=[early_stopping, reduce_lr]
    )
    
    print(f"\n=== TRAINING SUMMARY ===")
    print(f"Stopped at epoch: {len(history.history['loss'])}")
    print(f"Best validation loss: {min(history.history['val_loss']):.6f}")
    
    return history

def train_model_with_roto_translation_quads(model, x, y, rotation_prob=0.3, translation_prob=0.3, translation_scale=0.1):
    if x.shape[1] != 13:
        raise ValueError(f"Quad model expects 13 input features, got {x.shape[1]}")
    
    early_stopping = MinimumEpochEarlyStopping(min_epochs=10, patience=10, monitor='val_loss')
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6, verbose=1)
    
    history = model.fit(
        x=x, 
        y=y,
        validation_split=0.1,
        batch_size=32,
        epochs=100,
        shuffle=True, 
        verbose=1,
        callbacks=[early_stopping, reduce_lr]
    )
    
    print(f"\n=== TRAINING SUMMARY ===")
    print(f"Stopped at epoch: {len(history.history['loss'])}")
    print(f"Best validation loss: {min(history.history['val_loss']):.6f}")
    print(f"Final training loss: {history.history['loss'][-1]:.6f}")
    
    return history

# ==================== SIMPLE TRAINING FUNCTIONS ====================

def train_simple_model(model, x, y, model_type='single'):
    """Simple training without augmentation for debugging"""
    print(f"\nTraining {model_type} model without augmentation...")
    
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=20,
        restore_best_weights=True,
        verbose=1
    )
    
    if model_type == 'single':
        batch_size = 16
    elif model_type == 'quad':
        batch_size = 32
    else:
        batch_size = 64
    
    history = model.fit(
        x=x,
        y=y,
        validation_split=0.1,
        batch_size=batch_size,
        epochs=100,
        shuffle=True,
        verbose=1,
        callbacks=[early_stopping]
    )
    
    return history

def train_with_cross_validation(model_creator, x_data, y_data, model_type='single', n_splits=3):
    """Train with cross-validation for small datasets"""
    from sklearn.model_selection import KFold
    
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    histories = []
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(x_data), 1):
        print(f"\nFold {fold}/{n_splits}:")
        
        x_train_fold, x_val_fold = x_data[train_idx], x_data[val_idx]
        y_train_fold, y_val_fold = y_data[train_idx], y_data[val_idx]
        
        model = model_creator()
        
        if model_type == 'single':
            history = train_simple_model(model, x_train_fold, y_train_fold, 'single')
        elif model_type == 'quad':
            history = train_simple_model(model, x_train_fold, y_train_fold, 'quad')
        else:
            history = train_model_with_roto_translation_doubles(model, x_train_fold, y_train_fold)
        
        histories.append(history)
    
    return histories

# ==================== ANALYSIS FUNCTIONS (unchanged) ====================

def analyze_training_history(history, model_name):
    """Analyze if model actually learned"""
    train_loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = len(train_loss)
    
    print(f"\n{'='*60}")
    print(f"ANALYSIS: {model_name}")
    print(f"{'='*60}")
    
    print(f"Epochs trained: {epochs}")
    print(f"Initial training loss: {train_loss[0]:.4f}")
    print(f"Final training loss: {train_loss[-1]:.4f}")
    print(f"Improvement: {((train_loss[0] - train_loss[-1]) / train_loss[0]) * 100:.1f}%")
    
    if epochs <= 1:
        print("Problem: Model stopped at epoch 1!")
        print("The model hasn't learned anything.")
        return False
    
    if (train_loss[0] - train_loss[-1]) / train_loss[0] < 0.1:
        print("Problem: Minimal improvement (<10%)")
        print("Model may not be learning effectively.")
        return False
    
    print("Model shows significant learning")
    return True

def compare_models_performance(models_dict, test_data_dict):
    """Compare multiple models' performance"""
    print(f"\n{'='*70}")
    print("MODEL PERFORMANCE COMPARISON")
    print(f"{'='*70}")
    
    results = []
    for name, (model, x_test, y_test) in models_dict.items():
        if len(x_test) > 0:
            loss, mae = model.evaluate(x_test, y_test, verbose=0)
            results.append({
                'Model': name,
                'Test Loss': loss,
                'Test MAE': mae,
                'Samples': len(x_test)
            })
    
    results.sort(key=lambda x: x['Test MAE'])
    
    for i, r in enumerate(results, 1):
        print(f"{i}. {r['Model']}:")
        print(f"   MAE: {r['Test MAE']:.4f}, Loss: {r['Test Loss']:.4f}, Samples: {r['Samples']}")
    
    return results

# ==================== BACKWARD COMPATIBILITY ====================

def train_model_with_rotation(model, x, y, augmentation_prob=0.5):
    """Backward compatibility (augmentation disabled)"""
    if x.shape[1] == 13:
        return train_model_with_roto_translation_quads(model, x, y, rotation_prob=0.0, translation_prob=0.0)
    elif x.shape[1] == 7:
        return train_model_with_roto_translation_doubles(model, x, y, rotation_prob=0.0, translation_prob=0.0)
    elif x.shape[1] == 4:
        return train_model_with_roto_translation_singles(model, x, y, rotation_prob=0.0, translation_prob=0.0)
    else:
        raise ValueError(f"Unknown input shape: {x.shape[1]}")

def train_model_with_translation(model, x, y, augmentation_prob=0.5):
    """Backward compatibility (augmentation disabled)"""
    if x.shape[1] == 13:
        return train_model_with_roto_translation_quads(model, x, y, rotation_prob=0.0, translation_prob=0.0)
    elif x.shape[1] == 7:
        return train_model_with_roto_translation_doubles(model, x, y, rotation_prob=0.0, translation_prob=0.0)
    elif x.shape[1] == 4:
        return train_model_with_roto_translation_singles(model, x, y, rotation_prob=0.0, translation_prob=0.0)
    else:
        raise ValueError(f"Unknown input shape: {x.shape[1]}")

def train_model(model, x, y):
    """Original training function without augmentation"""
    early_stopping = MinimumEpochEarlyStopping(min_epochs=5, patience=15, monitor='val_loss')
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=1e-7, verbose=1)
    
    if x.shape[1] == 4:
        batch_size = 16
    elif x.shape[1] == 7:
        batch_size = 64
    else:   # 13 for quads
        batch_size = 32
    
    history = model.fit(
        x=x, 
        y=y,
        validation_split=0.1,
        batch_size=batch_size,
        epochs=200,
        shuffle=True, 
        verbose=1,
        callbacks=[early_stopping, reduce_lr]
    )
    
    print(f"\n=== TRAINING SUMMARY ===")
    print(f"Stopped at epoch: {len(history.history['loss'])}")
    print(f"Best validation loss: {min(history.history['val_loss']):.6f}")
    
    return history

def create_model():
    """Legacy function - creates quad model."""
    print("Warning: create_model() creates a quad model. Use specific functions instead.")
    return create_quad_model()

#-------------- simpler models without augmentation -------------------#

def create_simple_double_model(loss_type='mse', quantile=0.6):
    """Minimal double model with 2 hidden layers (64) – input shape 7"""
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(7,)),
        
        tf.keras.layers.Dense(32, use_bias=False),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.Dropout(0.0),
        
        tf.keras.layers.Dense(64, use_bias=False),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.Dropout(0.0),
        
        tf.keras.layers.Dense(6, activation='linear')
    ])

    if loss_type == 'quantile':
        loss = lambda yt, yp: quantile_loss(quantile, yt, yp)
    else:
        loss = 'mse'

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss=loss,
        metrics=['mae']
    )
    return model

def train_simple_model_legacy(model, x, y):   # renamed to avoid conflict with the existing train_simple_model
    """Original training function without augmentation"""
    early_stopping = MinimumEpochEarlyStopping(min_epochs=5, patience=15, monitor='val_loss')
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=1e-10, verbose=1)
    
    if x.shape[1] == 4:
        batch_size = 16
    elif x.shape[1] == 7:
        batch_size = 64
    else:
        batch_size = 32
    
    history = model.fit(
        x=x, 
        y=y,
        validation_split=0.1,
        batch_size=batch_size,
        epochs=1000,
        shuffle=True, 
        verbose=1,
        callbacks=[early_stopping, reduce_lr]
    )
    
    print(f"\n=== TRAINING SUMMARY ===")
    print(f"Stopped at epoch: {len(history.history['loss'])}")
    print(f"Best validation loss: {min(history.history['val_loss']):.6f}")
    
    return history