
## Neural Network for Jet Classification

Pytorch Code implementation based on the paper by G. Louppe, K. Cho, C. Becot and K. Cranmer (arXiv:1702.00748)

-------------------------------------------------------

Data Processing:

Raw jet data:
https://desycloud.desy.de/index.php/s/llbX3zpLhazgPJ
 
The data has to be preprocessed before feeding it to the RecNN network. The python scripts which perform the preprocessing are located in Data Loader folder.Recursive

Requirements: ROOT(https://root.cern.ch), 
              PyROOT(https://root.cern.ch/pyroot), 
              Fastjet(https://github.com/scikit-hep/pyjet)

To run:

Step1. python read_h5.py

(This script will create numpy arrays of each jet)

Step2. python content.py

(This script will save jet trees in pickles. Make sure to modify the name of the input, output file names)

Finally there will be 3 pickles for train, validation, and test which are ready to be fed into the RecNN. 

---------------------------------------------------------------

batching.py: Batches the input data for the RecNN

rec_net.py: Network architecture using Pytorch

train_batch.py: Driving code for training and saving the model weights

test_batch.py: Outputs the predicted class of the jets.

For training:

Python train_batch.py filename_train filename_valid filename_model

filename_train, filename_valid are saved with .pickle extension, filename_model in .pt extension.




