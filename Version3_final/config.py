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
model_path = os.path.join(root_dir,"./Saved_model/FinalModel50k_cuda_coverage.pt")
model_final_path = os.path.join(root_dir, "./Saved_model/FinalModel50k_cuda_final_coverage.pt")

# Pointer gen and coverage
IS_POINTER_GEN = True
IS_COVERAGE = True

# Training problem
MAX_DEC_LEN = 100
MAX_ENC_LEN = 400

## Hyperparameters
EPOCH = 35
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
