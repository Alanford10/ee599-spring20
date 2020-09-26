from tensorflow.keras.models import load_model
model = load_model('weights.49-0.56.hdf')
model.save_weights('weights.49-0.56.h5', overwrite=True)
