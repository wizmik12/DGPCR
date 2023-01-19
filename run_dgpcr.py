import os
import time
import numpy as np
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split

import tensorflow as tf
import gpflow
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="2"
#prevent tensorflow from allocating the entire GPU memory at once
config = tf.ConfigProto(intra_op_parallelism_threads=8, inter_op_parallelism_threads=8)

config.gpu_options.allow_growth=True
sess = tf.Session(config=config)
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())


from scipy.cluster.vq import kmeans
from dgpcr import DGPCR
import pickle

from utils_svgp import train_GP_model, batch_prediction_multi_classification, prediction_multi_dgp, eval_performance

#Results_folder
if not os.path.exists('./Results/'):
    os.makedirs('./Results/')

#Seed
for L in [2, 3, 4]:
    np.random.seed(0)
    tf.set_random_seed(1111)

    #Loading data
    X_train, y_train = np.load("./features_pretrained/X_train_512pool.npy").astype('float64'), np.load("./features_pretrained/annotations_ane.npy")
    X_test, y_test = np.load("./features_pretrained/X_test_512pool.npy").astype('float64'), np.load("./features_pretrained/y_test.npy").astype('float64')
    m,s=X_train.mean(0),X_train.std(0)
    X_train = (X_train - m)/s
    X_test = (X_test - m)/s
    y_test = np.argmax(y_test, 1).reshape(-1,1)
    X_train_sub, X_valid, y_train_sub, y_valid = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

    num_inducing=100
    Z = kmeans(X_train_sub, num_inducing)[0]
    minibatch_size = 1000 if X_train_sub.shape[0]>1000 else None

    kernels = [gpflow.kernels.RBF(X_train_sub.shape[1], variance=2.0, lengthscales =2.0, ARD=False)]
    for _ in range(1,L):
        kernels.append(gpflow.kernels.RBF(10, variance=2.0, lengthscales =2.0, ARD=False))
    print("Number of layers: ", len(kernels))
    m = DGPCR(X=X_train_sub, Y=y_train_sub, Z=Z, kernels=kernels, likelihood=gpflow.likelihoods.SoftMax(3), num_outputs=3, minibatch_size=minibatch_size)
    for layer in m.layers[:-1]:
        layer.q_sqrt = layer.q_sqrt.value * 1e-5

    #Como tenemos etiquetas de CR en train escogemos el test ("como validacion") para no afectar el resto del cï¿½digo
    trained_m = train_GP_model(m, X_train_sub, X_test, y_test, num_inducing)

    start = time.time()
    preds, probs = batch_prediction_multi_classification(m, prediction_multi_dgp, X_test, S=100)
    print("Time inference ", time.time()-start)

    y_test=np.load("./features_pretrained/y_test.npy").astype('float64')
    y_test_class = np.argmax(y_test,axis=1)

    np.save("probs", probs)
    metrics, mat, acc, NLL = eval_performance(preds, y_test_class, probs, output=None)
    print(metrics, '\n NLL ', NLL)
    roc_macro = roc_auc_score(y_test, probs, average='macro')
    print('AUC', roc_macro)

    with open('./Results/results_dgpcr_' + str(L) + '.txt', 'w') as the_file:
        the_file.write(metrics)
        the_file.write('\n\nAUC  ' + str(roc_macro) +'\nNLL ' + str(NLL))



    dir2Save = './models/'
    if not os.path.exists(dir2Save):
        os.makedirs(dir2Save)
    path = dir2Save + 'model_dgpcr_' + str(L) + '.pkl'
    with open(path, 'wb') as fp:
        pickle.dump(trained_m.read_trainables(), fp)
    saver = gpflow.saver.Saver()
    saver.save(dir2Save + "dgpcr_" + str(L) + ".gpflow", m)
