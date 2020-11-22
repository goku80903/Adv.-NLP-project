import os
import logging
root_dir = os.path.abspath('.')
log_root = os.path.join(root_dir,"./output.log")
logging.basicConfig(filename=log_root,filemode='w',format='%(levelname)s - %(message)s',level=logging.INFO)
try:
    os.mkdir('../data')
except:
    logging.error('The file "data" already exists,cannot be created')

try:
    os.mkdir('./Saved_model')
except:
    logging.error('The file "data" already exists,cannot be created')

rain_data_path = os.path.join(root_dir,"../finished_files/train.bin")
est_data_path = os.path.join(root_dir,"../finished_files/test.bin")
eval_data_path = os.path.join(root_dir,"../finished_files/val.bin")
train_data_path_final = os.path.join(root_dir,"../data/train.pkl")
test_data_path_final = os.path.join(root_dir,"../data/test.pkl")
eval_data_path_final = os.path.join(root_dir,"../data/val.pkl")
vocab_path = os.path.join(root_dir,"../finished_files/vocab")
model_path = os.path.join(root_dir,"./Saved_model/FirstModel_cuda.pt")

# Training problem
MAX_DEC_LEN = 100

## Hyperparameters
EPOCH = 40
batch_size = 8
ENC_EMB_DIM = 128
DEC_EMB_DIM = 128
ENC_HID_DIM = 256
DEC_HID_DIM = 256
N_LAYERS = 1
ENC_DROPOUT = 0.5
DEC_DROPOUT = 0.5

### Decoding parameters
BEAM_SIZE = 4