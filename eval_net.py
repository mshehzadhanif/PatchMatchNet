import os
import argparse
import numpy as np
from sklearn.metrics import roc_curve
from scipy import interpolate
from tensorflow.python.keras.models import load_model
from tensorflow.python.keras.models import Model
from extended_imagedatagen import SimNetImageDataGenerator

parser = argparse.ArgumentParser('Patch Match Networks')
parser.add_argument('--data_dir', required=True, type=str)
parser.add_argument('--model_name', required=True, type=str)
parser.add_argument('--test_set', default='yosemite', type=str)
parser.add_argument('--test_matches', default='m50_100000_100000_0.txt', type=str)
parser.add_argument('--network_type', required=True, type=str)
BATCH_SIZE = 128

#normalize descriptor
def normalize_descriptor(x):
    norm = np.linalg.norm(x, ord=None, axis=1, keepdims=True)
    x_norm = x / norm
    return x_norm

#evaluate model
def eval_net(data_dir, model_name, test_set, test_matches, network_type):

    #load test data
    data_file_name = os.path.join(os.path.join(data_dir, test_set), test_set + '_data.npy')
    test_datagen = SimNetImageDataGenerator(
        data_file_name,
        test_matches,
        network_type,
        preprocessing = True)
    test_samples = test_datagen.get_total_number_of_samples()

    #load model
    model = load_model(model_name)

    # Evaluation
    print('Evaluation on Test Set')
    y_test = np.zeros((test_samples, 1), dtype=np.int32)
    y_pred = np.zeros((test_samples, 1), dtype=np.float32)
    total_batches = int(np.ceil(test_samples / BATCH_SIZE))

    #for embedding
    if network_type == 'siam_l2':
        x = model.get_layer('model_1').input
        y = model.get_layer('model_1').output
        siam_model = Model(inputs=x, outputs=y)

        i = 0
        for (batch_x, batch_y) in test_datagen.flow(batch_size=BATCH_SIZE, shuffle=False):
            y_test[i * BATCH_SIZE:(i + 1) * BATCH_SIZE, :] = batch_y[:, None]
            d1 = siam_model.predict_on_batch(batch_x[0])
            d2 = siam_model.predict_on_batch(batch_x[1])
            d1 = normalize_descriptor(d1)
            d2 = normalize_descriptor(d2)
            dist = np.sqrt(np.sum(np.power(d1 - d2, 2), axis=1))
            y_pred[i * BATCH_SIZE:(i + 1) * BATCH_SIZE, :] = dist[:,None]
            i += 1
            print('\rProcessing batch: %d/%d' % (i, total_batches), sep=' ', end = '', flush = True)
            if i == total_batches:
                break
        #for roc evaluation, reverse the polairty of labels
        y_test[y_test == -1] = 0
        y_test = 1 - y_test
    else:
        #evaluate 2ch, 2ch2stream or siam models
        if network_type == '2ch':
            i = 0
            for (batch_x, batch_y) in test_datagen.flow(batch_size=BATCH_SIZE, shuffle=False):
                y_test[i*BATCH_SIZE:(i+1)*BATCH_SIZE,:] = batch_y[:,None]
                y_pred[i * BATCH_SIZE:(i + 1) * BATCH_SIZE, :] = model.predict_on_batch(batch_x)
                i += 1
                print('\rProcessing batch: %d/%d' % (i, total_batches), sep=' ', end='', flush=True)
                if i == total_batches:
                    break
        else:
            i = 0
            for (batch_x, batch_y) in test_datagen.flow(batch_size=BATCH_SIZE, shuffle=False):
                y_test[i*BATCH_SIZE:(i+1)*BATCH_SIZE,:] = batch_y[:,None]
                y_pred[i * BATCH_SIZE:(i + 1) * BATCH_SIZE, :] = model.predict_on_batch([batch_x[0], batch_x[1]])
                i += 1
                print('\rProcessing batch: %d/%d' % (i, total_batches), sep=' ', end='', flush=True)
                if i == total_batches:
                    break

    # computing fpr95
    fpr, tpr, thr = roc_curve(y_test, y_pred, pos_label=1)
    fpr95 = float(interpolate.interp1d(tpr, fpr)(0.95))
    print("\nTest: fpr95 = %f" % (fpr95))

    return

# main function for train
if __name__ == '__main__':
    # command line arguments
    opt = parser.parse_args()

    # evaluation function
    eval_net(data_dir=opt.data_dir, model_name=opt.model_name, test_set=opt.test_set, test_matches=opt.test_matches, network_type=opt.network_type)
