# """
# Model_FSL_dynamic.py
# LSTM-based model for dynamic sign language recognition
# Similar to your Model_FSL.py structure
# """

# import tensorflow as tf
# from tensorflow import keras
# from tensorflow.keras import layers


# def create_dynamic_model(input_shape, num_classes):
#     """
#     Create LSTM-based model for dynamic sign recognition
    
#     Args:
#         input_shape: (sequence_length, features) e.g., (30, 126)
#         num_classes: Number of sign classes
    
#     Returns:
#         Compiled Keras model
#     """
    
#     model = keras.Sequential([
#         # Input layer
#         layers.Input(shape=input_shape),
        
#         # Masking layer to ignore padding
#         layers.Masking(mask_value=0.0),
        
#         # LSTM layers for temporal modeling
#         layers.LSTM(128, return_sequences=True, dropout=0.3),
#         layers.LSTM(128, return_sequences=True, dropout=0.3),
#         layers.LSTM(64, dropout=0.3),
        
#         # Dense layers
#         layers.Dense(128, activation='relu'),
#         layers.Dropout(0.5),
#         layers.Dense(64, activation='relu'),
#         layers.Dropout(0.3),
        
#         # Output layer
#         layers.Dense(num_classes, activation='softmax')
#     ], name='FSL_Dynamic_Model')
    
#     return model


# def get_model_config():
#     """Return model configuration"""
#     return {
#         'architecture': 'LSTM-based',
#         'lstm_units': [128, 128, 64],
#         'dense_units': [128, 64],
#         'dropout_rates': [0.3, 0.3, 0.3, 0.5, 0.3],
#         'activation': 'relu',
#         'output_activation': 'softmax'
#     }


# if __name__ == '__main__':
#     # Test model creation
#     print("Testing model creation...")
    
#     # Example: 30 frames, 126 features per frame, 7 classes
#     test_model = create_dynamic_model(input_shape=(30, 126), num_classes=7)
#     test_model.summary()
    
#     print("\n✅ Model created successfully!")
#     print(f"Total parameters: {test_model.count_params():,}")