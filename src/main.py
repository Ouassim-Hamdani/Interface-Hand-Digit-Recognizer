from utils import csv_to_numpy
from model import EnsembledModel
import os

def test_run():
    X,y = csv_to_numpy(os.path.join("data","test_gen.csv"))
    model = EnsembledModel()
    model.predict_visualize(X[:12])
if __name__=="__main__":
    test_run()