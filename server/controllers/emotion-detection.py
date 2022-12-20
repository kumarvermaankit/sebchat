import sys
import os
import joblib
val=sys.argv[1]

#absolute path of the finalized_model.sav file

filepath = os.path.dirname(os.path.abspath(__file__))
loaded_model = joblib.load(filepath + "/finalized_model.sav")

# loaded_model = joblib.load("finalized_model.sav")
# sample = "What are you saying to me ?"
result = loaded_model.predict([val])

print(result)