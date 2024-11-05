import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import math,os
from sklearn.metrics import accuracy_score,confusion_matrix
from utils import check_and_normalize,check_channels
class EnsembledModel:
    """
    An ensemble model that combines predictions from two pre-trained and fine-tuned models: VGG19 and ResNet50.

    Attributes:
        model (tf.keras.Model): The loaded and fine-tuned VGG19 model.
        model_2 (tf.keras.Model): The loaded and fine-tuned ResNet50 model.

    Methods:
        __init__(self, models_path="models"): 
            Initializes the EnsembledModel by loading the VGG19 and ResNet50 models from the specified path.

        predict(self, X, return_labels=False): 
            Predicts the class probabilities or labels for the given input data.

        evaluate(self, X, y): 
            Evaluates the model's performance on the given data using accuracy.

        predict_visualize(self, X): 
            Predicts the labels for the given input data and visualizes the predictions along with the input images.
    """
    def __init__(self,models_path="models"):
            """
            Initializes the EnsembledModel.

            Args:
                models_path (str): The path to the directory containing the saved models. Defaults to "models".
            """
            self.model = tf.keras.models.load_model(os.path.join(models_path,'VGG19.keras'))  # Loading Finetunned VGG19 Model
            self.model_2 = tf.keras.models.load_model(os.path.join(models_path,'RESNET50.keras'))  # Loading Finetunned ResNet50 Model
            self.model_3 = tf.keras.models.load_model(os.path.join(models_path,'CNN.keras'))

    def predict(self,X,return_labels=False):
        """
        Predicts the class probabilities or labels for the given input data.

        Args:
            X (numpy.ndarray): The input data as a NumPy array.
            return_labels (bool): Whether to return the predicted labels (True) or class probabilities (False). Defaults to False.

        Returns:
            numpy.ndarray: The predicted class probabilities or labels.
        """
        X = check_and_normalize(X)
        X = check_channels(X)
        y_pred_1 = self.model.predict(X)
        y_pred_2 = self.model_2.predict(X)
        y_pred_3 = self.model_3.predict(X)
        y_pred = (y_pred_1+y_pred_2+y_pred_3)/3
        if return_labels:
            return np.argmax(y_pred,axis=1)
        return y_pred
    
    
    def evaluate(self,X,y):
        """
        Evaluates the model's performance on the given data using accuracy.

        Args:
            X (numpy.ndarray): The input data.
            y (numpy.ndarray): The true labels.

        Returns:
            float: The accuracy score.
        """
        y_pred = self.predict(X,return_labels=True)
        return accuracy_score(y,y_pred)


    def predict_visualize(self,X):
        """
        Predicts the labels for the given input data and visualizes the predictions along with the input images.

        Args:
            X (numpy.ndarray): The input data.
        """
        
        h,w=[3,4]
        m=12
        if len(X)>12:
            print("Warning : You passed more then 12 Examples, Visualize only gonna treat & show 12 of them")
        elif len(X)<12:
            h = math.ceil(math.sqrt(len(X)))  # Calculate the ceiling of the square root
            w = math.ceil(len(X) / h)
            m = len(X)
        y_pred = self.predict(X,return_labels=True)
        _, axs = plt.subplots(h,w)
        axs = axs.flatten()
        for i in range(m):
            axs[i].imshow(X[i])
            axs[i].set_title(f"Prediction: {y_pred[i]}")
            axs[i].axis('off')
        plt.show()
    
    def cm(self,X,y):
        y_pred = self.predict(X,return_labels=True)
        return confusion_matrix(y,y_pred)