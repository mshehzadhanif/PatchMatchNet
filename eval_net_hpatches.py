import os
from utils.hpatch import *
import cv2
import csv
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.models import load_model


# all types of patches
tps = ['ref','e1', 'e2','e3', 'e4', 'e5', \
       'h1','h2','h3','h4','h5',\
       't1','t2','t3','t4','t5']

#ocmputer descriptors on a sequence
def compute_descriptors(model, seq, tp, save_file_name):
    """compute the descriptor"""
    ids = range(0, int(seq.N))
    vis = np.empty((len(ids), 64, 64, 1), dtype=np.float32)
    # add the actual patches
    for idx in ids:
        #get patch
        im = get_patch(seq,tp,idx)
        #apply preprocessing
        im = im[0:64, 0:64]
        im = (im - np.mean(im, (0,1)))/255.0
        #store for processing
        vis[idx, :, :, 0] = im.astype(np.float32)

    #compute descriptors on the patches
    descr = model.predict(vis, batch_size=512)    

    #compute descriptor and save them in a cvs file
    np.savetxt(save_file_name, descr, fmt='%1.6f', delimiter=",")

# main function for train
if __name__ == '__main__':
   #load model
   model = load_model('model_siam_liberty.h5')
   x = model.get_layer('model_1').input
   y = model.get_layer('model_1').output
   siam_model = Model(inputs=x, outputs=y)

   #get directory list
   data_dir = '../data/hpatches-release'
   list_seq = os.listdir(data_dir)

   #create output directory
   save_dir = '../data/descriptors/patch-match-net'
   if not os.path.isdir(save_dir):
       os.makedirs(save_dir)

   #get all in the sequence
   i = 0
   for seq_name in list_seq:
       # load a sequence
       sname = os.path.join(data_dir, seq_name)
       print('processing sequence %d, path = %s' %(i, sname))
       seq = hpatch_sequence(sname)
       i += 1

       #create directory for descriptors
       save_seq_descr_name = os.path.join(save_dir, seq_name)
       if not os.path.isdir(save_seq_descr_name):
           os.makedirs(save_seq_descr_name)

       #traverse each attribute
       for tp in tps:
           # list of patch indices to visualise
           print('\tcomputing descriptors on attribute: %s' % (tp))
           save_file_name = os.path.join(save_seq_descr_name, tp + '.csv')
           compute_descriptors(siam_model, seq=seq, tp=tp, save_file_name=save_file_name)
