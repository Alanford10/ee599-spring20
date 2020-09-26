import numpy as np
from tensorflow.keras import Model
from model import get_model, get_streaming

streaming_model = get_streaming()
streaming_model.load_weights('./logs/weights.49-0.56.h5')
print(streaming_model.predict(np.random.rand(1,1,64)))
