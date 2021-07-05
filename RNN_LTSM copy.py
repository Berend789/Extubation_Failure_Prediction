import gc
from time import time
import os
import math
import pickle
from keras.layers.core import Dropout
from keras.layers.recurrent import SimpleRNN


import numpy as np
import pandas as pd
#from tensorflow.python.keras.optimizer_v2 import RMSprop
from tensorflow.python.ops.gen_array_ops import Reshape
from LR_build import PadSequences
#from processing_utilities import PandasUtilities
from attention_function import attention_3d_block as Attention


from keras import backend as K
from keras.models import Model, Input, load_model  # model_from_json
from keras.layers import Masking, Flatten, Embedding, Dense, LSTM, TimeDistributed, Bidirectional, GRU, SimpleRNN
from keras.callbacks import TensorBoard, ModelCheckpoint
from keras.preprocessing.sequence import pad_sequences
from keras import regularizers
from keras.optimizer_v2 import rmsprop
#from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import RobustScaler, MinMaxScaler
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score, classification_report
from sklearn.metrics import recall_score, precision_score
from sklearn.model_selection import StratifiedKFold
import LR_build as pp

PATH = output_folder = os.path.join(os.getcwd(), 'Results_RNN')
FILE = 'processed_df.txt'


def get_synth_sequence(n_timesteps=12):
    """
    Returns a single synthetic data sequence of dim (bs,ts,feats)
    Args:
    ----
      n_timesteps: int, number of timesteps to build model for
    Returns:
    -------
      X: npa, numpy array of features of shape (1,n_timesteps,2)
      y: npa, numpy array of labels of shape (1,n_timesteps,1) 
    """

    X = np.array([[np.random.rand() for _ in range(n_timesteps)],
                  [np.random.rand() for _ in range(n_timesteps)]])
    X = X.reshape(1, n_timesteps, 2)
    y = np.array([0 if x.sum() < 0.5 else 1 for x in X[0]])
    y = y.reshape(1, n_timesteps, 1)
    return X, y


def return_data(synth_data=False, balancer=True, target='Reintub',
                return_cols=False, tt_split=0.65, val_percentage=0.75,
                cross_val=False, mask=False, dataframe=False,
                time_steps=12, split=True, pad=True):
    """
    Returns synthetic or real data depending on parameter
    Args:
    -----
        synth_data : synthetic data is False by default
        balance : whether or not to balance positive and negative time windows 
        target : desired target, supports MI, SEPSIS, VANCOMYCIN or a known lab, medication
        return_cols : return columns used for this RNN
        tt_split : fraction of dataset to use fro training, remaining is used for test
        cross_val : parameter that returns entire matrix unsplit and unbalanced for cross val purposes
        mask : 24 hour mask, default is False
        dataframe : returns dataframe rather than numpy ndarray
        time_steps : 12 by default, required for padding
        split : creates test train splits
        pad : by default is True, will pad to the time_step value
    Returns:
    -------
        Training and validation splits as well as the number of columns for use in RNN  
    """

    if synth_data:
        no_feature_cols = 2
        X_train = []
        y_train = []

        for i in range(10000):
            X, y = get_synth_sequence(n_timesteps=12)
            X_train.append(X)
            y_train.append(y)
        X_TRAIN = np.vstack(X_train)
        Y_TRAIN = np.vstack(y_train)

    else:
        f = open(os.path.join('./Results_LR_RF/', 'processed_df.txt'), 'rb')
        df = pickle.load(f)
        f.close()

        df = df.select_dtypes(exclude=['object'])
        # df = df.rename(columns={'Unnamed: 0': 'Admissionnumber'})
        if pad:
            pad_value = -100
            df = PadSequences().pad(df, 1, time_steps, pad_value=pad_value)
            df['Reintub'].replace(value=0, to_replace=-100, inplace=True)
            print('There are {0} rows in the df after padding'.format(len(df)))

        COLUMNS = list(df.columns)

        # Remove Admissionnumber / index number
        if 'pat_hosp_id' in COLUMNS:
            COLUMNS.remove('pat_hosp_id')

        if 'pat_datetime' in COLUMNS:
            COLUMNS.remove('pat_datetime')
            COLUMNS.remove(target)
            COLUMNS.remove('Adnum')
            COLUMNS.remove('Extubation_date')

        print(COLUMNS)
        if return_cols:
            return COLUMNS

        if dataframe:
            return (df[COLUMNS+[target]])

        # create 3D matrix of different shapes
        MATRIX = df[COLUMNS+[target]].values
        MATRIX = MATRIX.reshape(
            int(MATRIX.shape[0]/time_steps), time_steps, MATRIX.shape[1])

        # note we are creating a second order bool matirx
        bool_matrix = (~MATRIX.any(axis=2))
        MATRIX[bool_matrix] = np.nan
        #MATRIX = PadSequences().ZScoreNormalize(MATRIX)
        # restore 3D shape to boolmatrix for consistency
        bool_matrix = np.isnan(MATRIX)
        MATRIX[bool_matrix] = pad_value

        permutation = np.random.permutation(MATRIX.shape[0])
        MATRIX = MATRIX[permutation]
        bool_matrix = bool_matrix[permutation]

        X_MATRIX = MATRIX[:, :, 0:-1]
        Y_MATRIX = MATRIX[:, :, -1]

        x_bool_matrix = bool_matrix[:, :, 0:-1]
        y_bool_matrix = bool_matrix[:, :, -1]

        X_TRAIN = X_MATRIX[0:int(tt_split*X_MATRIX.shape[0]), :, :]
        Y_TRAIN = Y_MATRIX[0:int(tt_split*Y_MATRIX.shape[0]), :]
        Y_TRAIN = Y_TRAIN.reshape(Y_TRAIN.shape[0], Y_TRAIN.shape[1], 1)

        X_VAL = X_MATRIX[int(tt_split*X_MATRIX.shape[0]):int(val_percentage*X_MATRIX.shape[0])]
        Y_VAL = Y_MATRIX[int(tt_split*Y_MATRIX.shape[0]):int(val_percentage*Y_MATRIX.shape[0])]
        Y_VAL = Y_VAL.reshape(Y_VAL.shape[0], Y_VAL.shape[1], 1)

        x_val_boolmat = x_bool_matrix[int(
            tt_split*x_bool_matrix.shape[0]):int(val_percentage*x_bool_matrix.shape[0])]
        y_val_boolmat = y_bool_matrix[int(
            tt_split*y_bool_matrix.shape[0]):int(val_percentage*y_bool_matrix.shape[0])]
        y_val_boolmat = y_val_boolmat.reshape(
            y_val_boolmat.shape[0], y_val_boolmat.shape[1], 1)

        X_TEST = X_MATRIX[int(val_percentage*X_MATRIX.shape[0])::]
        Y_TEST = Y_MATRIX[int(val_percentage*X_MATRIX.shape[0])::]
        Y_TEST = Y_TEST.reshape(Y_TEST.shape[0], Y_TEST.shape[1], 1)

        x_test_boolmat = x_bool_matrix[int(
            val_percentage*x_bool_matrix.shape[0])::]
        y_test_boolmat = y_bool_matrix[int(
            val_percentage*y_bool_matrix.shape[0])::]
        print(y_test_boolmat.shape)
        y_test_boolmat = y_test_boolmat.reshape(
            y_test_boolmat.shape[0], y_test_boolmat.shape[1], 1)

        X_TEST[x_test_boolmat] = pad_value
        Y_TEST[y_test_boolmat] = pad_value

        if balancer:
            TRAIN = np.concatenate([X_TRAIN, Y_TRAIN], axis=2)
            print(np.where((TRAIN[:, :, -1] == 1).any(axis=1))[0])
            pos_ind = np.unique(
                np.where((TRAIN[:, :, -1] == 1).any(axis=1))[0])
            print(pos_ind)
            np.random.shuffle(pos_ind)
            neg_ind = np.unique(
                np.where(~(TRAIN[:, :, -1] == 1).any(axis=1))[0])
            print(neg_ind)
            np.random.shuffle(neg_ind)
            length = min(pos_ind.shape[0], neg_ind.shape[0])
            total_ind = np.hstack([pos_ind[0:length], neg_ind[0:length]])
            np.random.shuffle(total_ind)
            ind = total_ind
            X_TRAIN = TRAIN[ind, :, 0:-1]
            Y_TRAIN = TRAIN[ind, :, -1]
            Y_TRAIN = Y_TRAIN.reshape(Y_TRAIN.shape[0], Y_TRAIN.shape[1], 1)

    no_feature_cols = X_TRAIN.shape[2]

    if mask:
        print('MASK ACTIVATED')
        X_TRAIN = np.concatenate(
            [np.zeros((X_TRAIN.shape[0], 1, X_TRAIN.shape[2])), X_TRAIN[:, 1::, ::]], axis=1)
        X_VAL = np.concatenate(
            [np.zeros((X_VAL.shape[0], 1, X_VAL.shape[2])), X_VAL[:, 1::, ::]], axis=1)

    if cross_val:
        return (MATRIX, no_feature_cols)

    if split == True:
        return (X_TRAIN, X_VAL, Y_TRAIN, Y_VAL, no_feature_cols,
                X_TEST, Y_TEST, x_test_boolmat, y_test_boolmat,
                x_val_boolmat, y_val_boolmat)

    elif split == False:
        return (np.concatenate((X_TRAIN, X_VAL), axis=0),
                np.concatenate((Y_TRAIN, Y_VAL), axis=0), no_feature_cols)

    print(np.shape(X_TRAIN))
    print(np.shape(X_TEST))
    print(np.shape(X_VAL))
    print(np.shape(Y_TRAIN))
    print(np.shape(Y_TEST))
    print(np.shape(Y_VAL))


def build_model(no_feature_cols=None, time_steps=12, output_summary=False):
    """
    Assembles RNN with input from return_data function
    Args:
    ----
    no_feature_cols : The number of features being used AKA matrix rank
    time_steps : The number of days in a time block
    output_summary : Defaults to False on returning model summary
    Returns:
    ------- 
    Keras model object
    """
    print("time_steps:{0}|no_feature_cols:{1}".format(
        time_steps, no_feature_cols))
    input_layer = Input(shape=(12, 12))
    x = Attention(input_layer, time_steps)
    x = Masking(mask_value=0, input_shape=(12, 12))(input_layer)
    x = SimpleRNN(128, return_sequences=True)(x)
    x = LSTM(128, return_sequences=True)(x)
    preds = TimeDistributed(Dense(1, activation="sigmoid"))(x)
    model = Model(inputs=input_layer, outputs=preds)

    # , rho=0.9, epsilon=1e-08)
    RMS = rmsprop.RMSprop(learning_rate=0.001, rho=0.9, epsilon=1e-08)
    model.compile(optimizer=RMS, loss='binary_crossentropy', metrics=['acc'])

    if output_summary:
        model.summary()
    return model


def train(model_name="Ltsm_0", synth_data=False, target='Reintub',
          balancer=True, predict=False, return_model=False,
          n_percentage=1.0, time_steps=12, epochs=10):
    """
    Use Keras model.fit using parameter inputs
    Args:
    ----
    model_name : Parameter used for naming the checkpoint_dir
    synth_data : Default to False. Allows you to use synthetic or real data.
    Return:
    -------
    Nonetype. Fits model only. 
    """

    f = open(f'./pickled_objects/X_TRAIN_{target}.txt', 'rb')
    X_TRAIN = pickle.load(f)
    f.close()

    f = open('./pickled_objects/Y_TRAIN_{0}.txt'.format(target), 'rb')
    Y_TRAIN = pickle.load(f)
    f.close()

    f = open('./pickled_objects/X_VAL_{0}.txt'.format(target), 'rb')
    X_VAL = pickle.load(f)
    f.close()

    f = open('./pickled_objects/Y_VAL_{0}.txt'.format(target), 'rb')
    Y_VAL = pickle.load(f)
    f.close()

    f = open('./pickled_objects/x_boolmat_val_{0}.txt'.format(target), 'rb')
    X_BOOLMAT_VAL = pickle.load(f)
    f.close()

    f = open('./pickled_objects/y_boolmat_val_{0}.txt'.format(target), 'rb')
    Y_BOOLMAT_VAL = pickle.load(f)
    f.close()

    f = open('./pickled_objects/no_feature_cols_{0}.txt'.format(target), 'rb')
    no_feature_cols = pickle.load(f)
    f.close()

    X_TRAIN = X_TRAIN[0:int(n_percentage*X_TRAIN.shape[0])]
    Y_TRAIN = Y_TRAIN[0:int(n_percentage*Y_TRAIN.shape[0])]

    # build model
    model = build_model(no_feature_cols=no_feature_cols, output_summary=True,
                        time_steps=time_steps)

    # init callbacks
    tb_callback = TensorBoard(log_dir='./logs/{0}_.log'.format(model_name),
                              histogram_freq=0,
                              write_grads=False,
                              write_images=True,
                              write_graph=True)

    # Make checkpoint dir and init checkpointer
    checkpoint_dir = "./saved_models/{0}".format(model_name)

    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    checkpointer = ModelCheckpoint(
        filepath=checkpoint_dir+"/model.{epoch:02d}-{val_loss:.2f}.hdf5",
        monitor='val_loss',
        verbose=0,
        save_best_only=True,
        save_weights_only=False,
        mode='auto',
        save_freq='epoch')

    # fit
    model.fit(
        x=X_TRAIN,
        y=Y_TRAIN,
        batch_size=32,
        epochs=epochs,
        callbacks=[tb_callback],  # , checkpointer],
        validation_data=(X_VAL, Y_VAL),
        shuffle=True)

    model.save(os.path.join(os.getcwd(), 'saved_models',
                            model_name), save_format='tf')

    if predict:
        print('TARGET: {0}'.format(target))
        Y_PRED = model.predict(X_VAL)
        Y_PRED = Y_PRED[~Y_BOOLMAT_VAL]
        np.unique(Y_PRED)
        print(np.shape(Y_PRED))
        print(np.shape(Y_VAL))
        Y_VAL = Y_VAL[~Y_BOOLMAT_VAL]
        Y_PRED_TRAIN = model.predict(X_TRAIN)
        print(np.shape(Y_PRED))
        print(np.shape(Y_VAL))
        print('Confusion Matrix Validation')
        print(confusion_matrix(Y_VAL, np.around(Y_PRED)))
        print('Validation Accuracy')
        print(accuracy_score(Y_VAL, np.around(Y_PRED)))
        print('ROC AUC SCORE VAL')
        print(np.unique(Y_VAL))
        print(np.unique(Y_PRED))
        print(roc_auc_score(Y_VAL, np.around(Y_PRED)))
        print('CLASSIFICATION REPORT VAL')
        print(classification_report(Y_VAL, np.around(Y_PRED)))

    if return_model:
        return model


def return_loaded_model(model_name="ltsm_0"):

    loaded_model = load_model(os.path.join(os.getcwd(), 'saved_models',
                                           model_name))

    return loaded_model


def pickle_objects(target='Reintub', time_steps=12):

    (X_TRAIN, X_VAL, Y_TRAIN, Y_VAL, no_feature_cols,
     X_TEST, Y_TEST, x_boolmat_test, y_boolmat_test,
     x_boolmat_val, y_boolmat_val) = return_data(balancer=True, target=target,
                                                 pad=True,
                                                 split=True,
                                                 time_steps=time_steps)

    features = return_data(return_cols=True, synth_data=False,
                           target=target, pad=True, split=True,
                           time_steps=time_steps)

    output_folder = os.path.join(os.getcwd(), 'pickled_objects')
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    if not os.path.exists('./pickled_objects/X_TRAIN_{0}.txt'.format(target)):
        with open('./pickled_objects/X_TRAIN_{0}.txt'.format(target), 'w+'):
            pass
    f = open('./pickled_objects/X_TRAIN_{0}.txt'.format(target), 'wb')
    pickle.dump(X_TRAIN, f)
    f.close()

    if not os.path.exists('./pickled_objects/X_TRAIN_{0}.txt'.format(target)):
        with open('./pickled_objects/X_TRAIN_{0}.txt'.format(target), 'w+'):
            pass
    f = open('./pickled_objects/X_VAL_{0}.txt'.format(target), 'wb')
    pickle.dump(X_VAL, f)
    f.close()

    if not os.path.exists('./pickled_objects/Y_TRAIN_{0}.txt'.format(target)):
        with open('./pickled_objects/Y_TRAIN_{0}.txt'.format(target), 'w+'):
            pass
    f = open('./pickled_objects/Y_TRAIN_{0}.txt'.format(target), 'wb')
    pickle.dump(Y_TRAIN, f)
    f.close()

    if not os.path.exists('./pickled_objects/Y_VAL_{0}.txt'.format(target)):
        with open('./pickled_objects/Y_VAL_{0}.txt'.format(target), 'w+'):
            pass
    f = open('./pickled_objects/Y_VAL_{0}.txt'.format(target), 'wb')
    pickle.dump(Y_VAL, f)
    f.close()

    if not os.path.exists('./pickled_objects/X_TEST_{0}.txt'.format(target)):
        with open('./pickled_objects/X_TEST_{0}.txt'.format(target), 'w+'):
            pass
    f = open('./pickled_objects/X_TEST_{0}.txt'.format(target), 'wb')
    pickle.dump(X_TEST, f)
    f.close()

    if not os.path.exists('./pickled_objects/Y_TEST_{0}.txt'.format(target)):
        with open('./pickled_objects/Y_TEST_{0}.txt'.format(target), 'w+'):
            pass
    f = open('./pickled_objects/Y_TEST_{0}.txt'.format(target), 'wb')
    pickle.dump(Y_TEST, f)
    f.close()

    if not os.path.exists('./pickled_objects/x_boolmat_test_{0}.txt'.format(target)):
        with open('./pickled_objects/x_boolmat_test_{0}.txt'.format(target), 'w+'):
            pass
    f = open('./pickled_objects/x_boolmat_test_{0}.txt'.format(target), 'wb')
    pickle.dump(x_boolmat_test, f)
    f.close()

    if not os.path.exists('./pickled_objects/y_boolmat_test_{0}.txt'.format(target)):
        with open('./pickled_objects/y_boolmat_test_{0}.txt'.format(target), 'w+'):
            pass
    f = open('./pickled_objects/y_boolmat_test_{0}.txt'.format(target), 'wb')
    pickle.dump(y_boolmat_test, f)
    f.close()

    if not os.path.exists('./pickled_objects/x_boolmat_val_{0}.txt'.format(target)):
        with open('./pickled_objects/x_boolmat_val_{0}.txt'.format(target), 'w+'):
            pass
    f = open('./pickled_objects/x_boolmat_val_{0}.txt'.format(target), 'wb')
    pickle.dump(x_boolmat_val, f)
    f.close()

    if not os.path.exists('./pickled_objects/y_boolmat_val_{0}.txt'.format(target)):
        with open('./pickled_objects/y_boolmat_val_{0}.txt'.format(target), 'w+'):
            pass
    f = open('./pickled_objects/y_boolmat_val_{0}.txt'.format(target), 'wb')
    pickle.dump(y_boolmat_val, f)
    f.close()

    if not os.path.exists('./pickled_objects/no_feature_cols_{0}.txt'.format(target)):
        with open('./pickled_objects/no_feature_cols_{0}.txt'.format(target), 'w+'):
            pass
    f = open('./pickled_objects/no_feature_cols_{0}.txt'.format(target), 'wb')
    pickle.dump(no_feature_cols, f)
    f.close()

    if not os.path.exists('./pickled_objects/features_{0}.txt'.format(target)):
        with open('./pickled_objects/features_{0}.txt'.format(target), 'w+'):
            pass
    f = open('./pickled_objects/features_{0}.txt'.format(target), 'wb')
    pickle.dump(features, f)
    f.close()


if __name__ == "__main__":
    K.clear_session()
    # if not os.path.exists('./pickled_object/y_boolmat_val_{0}.txt'.format('Reintub')):
    pickle_objects(time_steps=12)
    K.clear_session()

    train(model_name='ltsm_final_no_mask_Rein_pad14', epochs=13,
          synth_data=False, predict=True, target='Reintub', time_steps=12)
    train(model_name='ltsm_final_no_mask_Rein_pad14_80_percent', epochs=13,
          synth_data=False, predict=True, time_steps=12,
          n_percentage=0.80)

    K.clear_session()

    train(model_name='ltsm_final_no_mask_Rein_pad14_60_percent', epochs=13,
          synth_data=False, predict=True,  time_steps=12,
          n_percentage=0.60)

    K.clear_session()

    train(model_name='ltsm_final_no_mask_Rein_pad14_40_percent', epochs=13,
          synth_data=False, predict=True, time_steps=12,
          n_percentage=0.40)

    K.clear_session()

    train(model_name='ltsm_final_no_mask_Rein_pad14_20_percent', epochs=13,
          synth_data=False, predict=True, time_steps=12,
          n_percentage=0.20)

    K.clear_session()

    train(model_name='ltsm_final_no_mask_Rein_pad14_10_percent', epochs=13,
          synth_data=False, predict=True, time_steps=12,
          n_percentage=0.10)

    K.clear_session()

    train(model_name='ltsm_final_no_mask_Rein_pad14_5_percent', epochs=13,
          synth_data=False, predict=True, time_steps=12,
          n_percentage=0.05)

    K.clear_session()
