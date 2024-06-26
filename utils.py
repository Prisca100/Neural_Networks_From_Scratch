import numpy as np
import matplotlib.pyplot as plt
import os
import zipfile
import pathlib
import shutil
import cv2
import random


def unzipping_files(zip_path:str, extracted_folder_name:str):
    zip_ref = zipfile.ZipFile(zip_path, "r")
    zip_ref.extractall(extracted_folder_name)
    zip_ref.close()


def create_img_dirs(images_dir_str:str, base_dir:str, categories):
    images_dir = pathlib.Path(images_dir_str)
    database_dir = pathlib.Path(base_dir)
    for category in categories:
        os.makedirs(database_dir / category, exist_ok=True)
    for category in categories:
        category_dir = images_dir/category
        dataset_category_dir = database_dir/category
        for img in category_dir.glob("*.jpg"):
            shutil.copy(img, dataset_category_dir/img.name)
    print("all images copied")

def train_test_split_data(data_dir, category, split_size=0.2):
    training_dir = data_dir / "training_set"
    test_dir = data_dir / "test_set"
    os.makedirs(training_dir / category, exist_ok=True)
    os.makedirs(test_dir / category, exist_ok=True)
    images = list((data_dir/category).glob("*.jpg"))

    random.shuffle(images)
    m = len(images)
    l_split = int((1-split_size ) * m)
    train_images = images[:l_split]
    test_images = images[l_split:]
    for img in train_images:
        shutil.copy(img,  training_dir / category / img.name)
    for img in test_images:
        shutil.copy(img, test_dir / category / img.name )


def to_numpy_array_supervised(database_dir:str, categories):
    """ Database dir has to contain properly labeled data. E.g
        |-dataset
        |---category_i
        |---category_2
    """
    database_dir = pathlib.Path(database_dir)
    # numpy_dir = os.makedirs(database_dir / "numpy_repr", exist_ok=True)
    numpy_dir = pathlib.Path(database_dir / "numpy_repr")
    print(numpy_dir)

    for category in categories:
        print(category)
        os.makedirs(numpy_dir / category, exist_ok=True)

    for category in categories:
        category_dir = database_dir / category
        numpy_category_dir = numpy_dir / category
        for img_path in category_dir.glob("*.jpg"):
            img = cv2.imread(str(img_path))
            img_array = np.array(img)
            np.save(numpy_category_dir / (img_path.stem + ".npy"), img_array)
    print("All images converted to numpy array")

    
# Activations
def sigmoid(Z):
    """returns sigmoid activation of input vector x"""
    A = 1/(np.exp(-Z) + 1)
    return A
    
def sigmoid_backward(dA, cache):
    """returns computed gradient dZ"""
    Z = cache
    s = sigmoid(Z)

    # s(1-s) =  sigmoid derivative
    dZ = dA * s*(1-s)
    assert(dZ.shape == Z.shape)
    return dZ

def tanH():
    pass

def relU(Z):
    A = np.maximum(0, Z)
    assert(A.shape == Z.shape)
    return A

def reLU_backward(dA, cache):
    Z =  cache
    dZ = np.array(dA, copy=True)
    dZ[Z<=0] = 0
    return dZ

def softMax(Z):
    Z_max = np.max(Z)
    exp_z = np.exp(Z-Z_max)
    A = exp_z / np.sum(exp_z)
    assert(A.shape == Z.shape)
    return A, Z

def softMax_backward(dA, cache):
    Z = cache
    s, _ = softMax(Z)
    s_diag = np.diag(s)
    s_outer = np.outer(s, s)
    jacobian = s_diag-s_outer
    dZ = dA * jacobian

    assert(dZ.shape == Z.shape)
    return dZ



def train_test_split(X, y, split_size, random_state=None):
    """Split size is test size"""
    if random_state is not None:
        np.random.seed(random_state)
    m_samples = X.shape[1]
    l_split = int((1-split_size ) * m_samples)

    indices = np.arange(m_samples)
    np.random.shuffle(indices)

    train_indices = indices[:l_split]
    test_indices = indices[l_split:]
    
    X_train, X_test = X[:, train_indices], X[:, test_indices]
    y_train, y_test = y[:, train_indices], y[:, test_indices]

    return X_train, y_train, X_test, y_test


# Image preprocessing
def image2Vector(image):
    if not isinstance(image,np.ndarray):
        image = np.array(image)
    print(image.shape)
    length, height, depth, m = image.shape
    if depth:
        return image.reshape(length*height*depth, m)
    return image.reshape(length*height,m)


# if __name__=="__main__":
#     # X = np.random.rand(3,100)
#     # y = np.random.choice([0,1], size=(1, 100))

#     # X_train, y_train, X_test, y_test = train_test_split(X, y, 0.2)
#     # print(f"y_train length: {y_train.shape[1]}, y length: {y.shape[1]}")

#     # unzipping_files("./archive.zip", "./content")
#     to_numpy_array_supervised("./dataset", ["cat", "dog"])