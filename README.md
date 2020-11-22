# Adv.-NLP-project
   Implementation of the paper [Get To The Point: Summarization with Pointer-Generator Networks by Abigail See](https://arxiv.org/pdf/1704.04368.pdf)
## Dataset
   The Dataset considered is same as the one used in the paper. It can be obtained from the [link](https://github.com/abisee/cnn-dailymail).

### Version 1
 The version 1 consists of the basic Bahandau attention model.
 
### Version 2
  This version is the baseline implementation of the model specifed in the paper. A varient of [Abstractive Text Summarization Using Sequence-to-Sequence RNNs and Beyond by Ramesh Nallapati](https://arxiv.org/pdf/1602.06023) without the large vocab trick. The vocab has been limited to a size of 50k by taking a minimum frequency of 27 while constructing the vocab. 
 
### Version 3 
  This version consists of the original model proposed in the paper. It is run seperately for pointer-gen and pointer-gen+coverage for experiment purposes.
  
## How to run
  Each of the repository can be independently run by the following:
  - Version 1: 
      Run train.py
  - Version 2:
      Run train.py
  - Version 3:
      Run train.py(with appropriate tags for pointer gen and coverage)
   
   P.S: The data after preprocessing and converted to pkl object(list of torchtext examples) is req to be present in data folder in the home directory in order to run the above
  
