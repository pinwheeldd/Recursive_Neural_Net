
## Recursive Neural Network for Jet Classification

Pytorch Code implementation based on the paper by G. Louppe, K. Cho, C. Becot and K. Cranmer (arXiv:1702.00748).

The goal is to classify boostd top jets and QCD jets.

-------------------------------------------------------

Data Processing:

Raw jet data (Boosted top and QCD jets): 
https://desycloud.desy.de/index.php/s/llbX3zpLhazgPJ
 
The jet data has to be preprocessed before feeding it to the RecNN network. The python scripts which perform the preprocessing are located in Data Loader folder. The jets are reclustered using kt algorithm. 


Requirements: ROOT(https://root.cern.ch), 
              PyROOT(https://root.cern.ch/pyroot), 
              Fastjet(https://github.com/scikit-hep/pyjet)
              Pytorch

##### To run:

Step1. python read_h5.py

(This script will create numpy arrays of each jet)

Step2. python content.py

(This script will save jet trees in pickles. Each entry in the data file is a dictionary belongs to each jet containing root jet features(pt, eta, phi, mass, energy), jet tree (each node has a left and right child or leaf), (px, py,pz, e) of all the constituents. Make sure to modify the name of the input, output file names)

Finally there will be 3 pickles for train, validation, and test which are ready to be fed into the RecNN. 

---------------------------------------------------------------

batching.py: Batches the input data for the RecNN

rec_net.py: Network architecture using Pytorch

train_batch.py: Driving code for training and saving the model weights

test_batch.py: Outputs the predicted class of the jets.

#### For training:



python train_batch.py filename_train filename_valid filename_model



filename_train, filename_valid are saved with .pickle extension, filename_model in .pt extension.




