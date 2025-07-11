from keras.models import load_model

# Load your original H5 model (exported from newer Keras)
model = load_model("model/emotion_model.h5", compile=False)

# Save it again in the same H5 format, but now with current formatting
model.save("model/emotion_model_converted.h5", save_format="h5")

print("✅ Model successfully converted and saved as emotion_model_converted.h5")
