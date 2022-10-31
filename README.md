**Patch Match Networks: Improved Two-Channel and Siamese Networks for Image Patch Matching**
------------------------------------------------------------------------------------------
Hanif MS. Patch match networks: Improved two-channel and Siamese networks for image patch matching. Pattern Recognition Letters. 2019 Apr 1;120:54-61.

Read full article [here](https://www.sciencedirect.com/science/article/abs/pii/S0167865519300054).

Evaluation: UBC benchmark dataset
---------------------------------

Data preparation is required to create dataset files in numpy format (.npy extension) for each sequence in UBC dataset. Use create_dataset_file.py from Deep Compare repository. The resulting data file must be renamed as [sequence_name]_data.npy.

To use the evaluation code for UBC benchmark dataset using pre-trained models, please install all these requirements.

usage: python eval_net.py --data_dir=path_to_UBC_dataset --model_name=name_of_model --test_set='liberty' or 'yosemite' or 'notredame' --network_type='2ch' or '2ch2stream' or 'siam' or 'siam_l2'

extended_imagedatagen.py  (a data generator required for batch generation)

Evaluation: HPatches benchmark dataset
--------------------------------------

The code is destined for descriptor computation on HPatches benchmark dataset. Follow the instructions to setup HPatches toolkit, data directories and additional requirements from https://github.com/hpatches/hpatches-benchmark. Siamese network trained on Liberty sequence is used by default. Descriptors are stored under ../data/descriptors/patch-match-net  in HPatches descriptor format. 

usage: python eval_net_hpatches.py 

The code has been tested on Windows 7, 64-bit with Python 3.5.
