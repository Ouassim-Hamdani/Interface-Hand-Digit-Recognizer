import pandas as pd
import numpy as np
def csv_to_numpy(df,h=28,w=28,c=3,target='label'):
    """Converts a CSV file or pandas DataFrame into NumPy arrays suitable for image classification.

  This function reads image data from a CSV file or DataFrame, reshapes it into 
  image format, pads the images, converts them to RGB, normalizes the pixel values,
  and separates the features (X) and labels (y).

  Args:
    df: Either a string representing the path to a CSV file or a pandas DataFrame 
        containing the image data.
    h: The desired height of the images. Default is 28.
    w: The desired width of the images. Default is 28.
    c: The number of color channels in the output images (1 for grayscale, 3 for RGB). 
        Default is 3.
    target: The name of the column in the DataFrame containing the labels. 
        Default is 'label'.

  Returns:
    A tuple containing two NumPy arrays:
      - X: A NumPy array of shape (num_samples, h, w, c) representing the image data.
      - y: A NumPy array of shape (num_samples,) representing the labels.

  Raises:
    TypeError: If the input `df` is neither a string nor a pandas DataFrame.

  Example:
    >>> X, y = csv_to_numpy('image_data.csv', h=32, w=32)
  """
    if isinstance(df,str):
        df = pd.read_csv(df)
    elif isinstance(df,pd.core.frame.DataFrame):
        pass
    else:
        raise TypeError("Wrong Data Type passed, please use a string to a csv file or a pandas dataframe")
    X = df.drop(target,axis=1).to_numpy().reshape((len(df),h,w,1)) # turn
    X = np.pad(X, ((0, 0), (2, 2), (2, 2), (0, 0)), mode='constant') #pad
    X = np.repeat(X, c, axis=-1) #turn to 3 axis
    X=X/255.0 # normalize
    y = df[target].to_numpy()
    return X,y  

import numpy as np

def check_and_normalize(X_input):
  """
  Checks if the input image is normalized ( min-max scaling to 0-1).
  If not normalized, normalizes it by dividing by 255.

  Args:
    X_input: The input image as a NumPy array.

  Returns:
    The normalized image as a NumPy array.
  """

  if np.max(X_input) > 1:
    print("Image is not normalized. Normalizing...")
    X_input = X_input / 255.0
  else:
    print("Image is already normalized.")
  return X_input