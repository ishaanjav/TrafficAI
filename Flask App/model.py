import tensorflow.keras
from tensorflow.keras.applications import EfficientNetB0

reconstructed_model = keras.models.load_model("model.h5")

# from tensorflow.keras.models import load_model
# model = load_model('model_weights.h5')


# # from tensorflow.keras.models import load_model
# # from tensorflow.keras.models import model_from_json
# # import json
# # import tensorflow


# # with open('model.json', 'r') as f:
# #     model_json = json.load(f)

# # # model = tensorflow.keras.models.load_model("model.h5")
# # print(model_json)
# # model = model_from_json(model_json)
# # # model.load_weights('model.h5')


# # # # load model
# # # model = load_model('model.h5')
# # # # summarize model.
# # # print(model.summary())
# # # # load dataset
# # # # split into input (X) and output (Y) variables
# # # # evaluate the model
