import tensorflow as tf

# Load the trained model
wavenet_model = tf.keras.models.load_model('soundpen_model.h5')

# Compile the model with potentially adjusted parameters
wavenet_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), loss='mse')

# Fine-tune the model
wavenet_model.fit(X_audio_train, Y_train, epochs=5, batch_size=8, validation_data=(X_audio_val, Y_val))

# Save the fine-tuned model
wavenet_model.save('soundpen_model_finetuned.h5')

print("Model fine-tuning complete and saved.")