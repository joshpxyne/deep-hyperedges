import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import tensorflow as tf
import keras.backend as K
from keras.layers import Input, Dense, LSTM, GRU, Embedding, Lambda, concatenate, Add
from keras.models import Model, load_model
from keras import optimizers
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.utils.vis_utils import model_to_dot
from tqdm import tqdm,trange
from sklearn.utils import shuffle
from keras.utils import plot_model
from keras import metrics

# from https://datascience.stackexchange.com/questions/45165/how-to-get-accuracy-f1-precision-and-recall-for-a-keras-model
def recall_m(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

def precision_m(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

def DeepHyperedges(vertex_embedding_dimension,hyperedge_embedding_dimension,max_hyperedge_size,num_outputs,dataset_name):
    input_tensor = Input((hyperedge_embedding_dimension + (vertex_embedding_dimension*max_hyperedge_size),))
    layer_1 = {}
    layer_2 = {}
    layer_3 = {}
    layer_2_depth = 100
    layer_3_depth = 100
    dense_rho_depth = 30
    sum_list = []
    
    key = "layer_1_hyperedge"
    value = Lambda(lambda x: x[:,0:hyperedge_embedding_dimension], output_shape=((hyperedge_embedding_dimension,)))(input_tensor)
    layer_1[key] = value
    ### Split into embeddings
    for i in range(max_hyperedge_size):
        key = "layer_1_"+str(i)
        value = Lambda(lambda x: x[:,hyperedge_embedding_dimension+(vertex_embedding_dimension*i):hyperedge_embedding_dimension+(vertex_embedding_dimension*(i+1))], output_shape=((vertex_embedding_dimension,)))(input_tensor) # necessary, as lambdas will take the variable names (and not values) with them
        layer_1[key] = value

    ### Dense layer 1 for embeddings
    for i in range(max_hyperedge_size):
        key = "layer_2_"+str(i)
        value = Dense(layer_2_depth, activation='relu')(layer_1["layer_1_"+str(i)])
        layer_2[key] = value

    ### Dense layer 2 for embeddings
    for i in range(max_hyperedge_size):
        key = "layer_3_"+str(i)
        value = Dense(layer_3_depth, activation='relu')(layer_2["layer_2_"+str(i)])
        layer_3[key] = value

    ### Sum layer for embeddings
    for i in range(max_hyperedge_size):
        sum_list.append(layer_3["layer_3_"+str(i)])

    Adder = Lambda(lambda x: K.sum(x, axis=0))

    summed = Adder(sum_list)
    dense_rho = Dense(dense_rho_depth, activation='sigmoid')(summed)
    concat = concatenate([layer_1["layer_1_hyperedge"],dense_rho])
    final_dense = Dense(100, activation='relu')(concat)
    output_tensor = Dense(num_outputs, activation='softmax')(final_dense)

    deephyperedge_model = Model(inputs=input_tensor,outputs=output_tensor)
    
    adam = optimizers.Adam(lr=1e-4, epsilon=1e-3)
    sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    deephyperedge_model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=[metrics.mae, metrics.categorical_accuracy, f1_m])

    return deephyperedge_model

def MLP(input_dimension,num_outputs,dataset_name):
    input_tensor = Input((input_dimension,))
    dense_hidden_1 = Dense(100, activation='relu')(input_tensor)
    output_tensor = Dense(num_outputs, activation='softmax')(dense_hidden_1)
    MLP_model = Model(inputs=input_tensor,outputs=output_tensor)
    
    adam = optimizers.Adam(lr=1e-4, epsilon=1e-3)
    sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    MLP_model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=[metrics.mae, metrics.categorical_accuracy, f1_m])
    return MLP_model

def DeepSets(vertex_embedding_dimension,max_hyperedge_size,num_outputs,dataset_name):
    input_tensor = Input((vertex_embedding_dimension*max_hyperedge_size,))
    layer_1 = {}
    layer_2 = {}
    layer_3 = {}
    layer_2_depth = 100
    layer_3_depth = 100
    dense_rho_depth = 30
    sum_list = []
    
    ### Split into embeddings
    for i in range(max_hyperedge_size):
        key = "layer_1_"+str(i)
        value = Lambda(lambda x: x[:,vertex_embedding_dimension*i:vertex_embedding_dimension*(i+1)], output_shape=((vertex_embedding_dimension,)))(input_tensor) # necessary, as lambdas will take the variable names (and not values) with them
        layer_1[key] = value

    ### Dense layer 1 for embeddings
    for i in range(max_hyperedge_size):
        key = "layer_2_"+str(i)
        value = Dense(layer_2_depth, activation='relu')(layer_1["layer_1_"+str(i)])
        layer_2[key] = value

    ### Dense layer 2 for embeddings
    for i in range(max_hyperedge_size):
        key = "layer_3_"+str(i)
        value = Dense(layer_3_depth, activation='relu')(layer_2["layer_2_"+str(i)])
        layer_3[key] = value

    ### Sum layer for embeddings
    for i in range(max_hyperedge_size):
        sum_list.append(layer_3["layer_3_"+str(i)])

    Adder = Lambda(lambda x: K.sum(x, axis=0))

    summed = Adder(sum_list)
    dense_rho = Dense(dense_rho_depth, activation='sigmoid')(summed)
    final_dense = Dense(100, activation='relu')(dense_rho)
    output_tensor = Dense(num_outputs, activation='softmax')(final_dense)

    deepset_model = Model(inputs=input_tensor,outputs=output_tensor)
    
    adam = optimizers.Adam(lr=1e-4, epsilon=1e-3)
    sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    deepset_model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=[metrics.mae, metrics.categorical_accuracy, f1_m])

    return deepset_model
    