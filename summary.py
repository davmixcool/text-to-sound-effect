import tensorflow as tf

# Load the fine-tuned model
wavenet_model = tf.keras.models.load_model('soundpen_model.h5')

# Print the model summary
wavenet_model.summary()

# Programmatically count the parameters
total_params = wavenet_model.count_params()

print(f'Total number of parameters in the model: {total_params}')
