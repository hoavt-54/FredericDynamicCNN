__author__ = 'Frederic Godin  (frederic.godin@ugent.be / www.fredericgodin.com)'
import theano
import theano.tensor as T
import numpy
import lasagne
import argparse

import DCNN
import dataUtils
import networks
import utils

parser = argparse.ArgumentParser(description='Train a DCNN on the binary Stanford Sentiment dataset as specified in the Kalchbrenner \'14 paper. All the default values are taken from the paper or the Matlab code.')
# training settings
parser.add_argument("--learning_rate",type=float, default=0.1, help='Learning rate')
parser.add_argument("--n_epochs",type=int,default=50,help="Number of epochs")
parser.add_argument("--valid_freq",type=int,default=10,help="Number of batches processed until we validate.")
parser.add_argument("--adagrad_reset",type=int,default=5,help="Resets the adagrad cumulative gradient after x epochs. If the value is 0, no reset will be executed.")
# input output
parser.add_argument("--vocab_size",type=int, default=50001, help='Vocabulary size')
parser.add_argument("--output_classes",type=int, default=3, help='Number of output classes')
parser.add_argument("--batch_size",type=int, default=4, help='Batch size')
# network paras
parser.add_argument("--word_vector_size",type=int, default=48, help='Word vector size')
parser.add_argument("--filter_size_conv_layers", nargs="+", type=int, default=[7,5],help="List of sizes of filters at layer 1 and 2, default=[10,7]")
parser.add_argument("--nr_of_filters_conv_layers", nargs="+", type=int, default=[6,14],help="List of number of filters at layer 1 and 2, default=[6,12]")
parser.add_argument("--activations",nargs='+', type=str,default=["tanh","tanh"],help="List of activation functions behind first and second conv layers, default [tanh, tanh]. Possible values are \"linear\", \"tanh\", \"rectify\" and \"sigmoid\". ")
parser.add_argument("--L2",nargs='+',type=float,default=[0.0001/2,0.00003/2,0.000003/2,0.0001/2],help="Fine-grained L2 regularization. 4 values are needed for 4 layers, namly for the embeddings layer, 2 conv layers and a final/output dense layer.")
parser.add_argument("--ktop",type=int,default=4,help="K value of top pooling layer DCNN")
parser.add_argument("--dropout_value", type=float,default=0.5,help="Dropout value after penultimate layer")

args = parser.parse_args()
hyperparas = vars(args)
print("Hyperparameters: "+str(hyperparas))

if len(hyperparas['filter_size_conv_layers'])!= 2 or len(hyperparas['nr_of_filters_conv_layers'])!=2 or len(hyperparas['activations'])!=2 or len(hyperparas["L2"])!=4 :
    raise Exception('Check if the input --filter_size_conv_layers, --nr_of_filters_conv_layers and --activations are lists of size 2, and the --L2 field needs a value list of 4 values.')


#######################
# LOAD  TRAINING DATA #
#######################
print('Loading the training data')

# load data, taken from Kalchbrenner matlab files
# we order the input according to length and pad all sentences until the maximum length
# at training time however, we will use the "length" array to shrink that matrix following the largest sentence within a batch
# in practice, this means that batches are padded with 1 or 2 zeros, or aren't even padded at all.
data_path = "../multinli_0.9/DCNN_format/"
train_y, train_sents1, train_sents2, train_lens1, train_lens2 = dataUtils.read_and_sort(data_path + 'train.txt')
dev_y, dev_sents1, dev_sents2, dev_lens1, dev_lens2 = dataUtils.read_and_sort(data_path + 'dev.txt')
test_y, test_sents1, test_sents2, test_lens1, test_lens2 = dataUtils.read_and_sort(data_path + 'test.txt')

# train data
n_train_batches = len(train_lens1) / hyperparas['batch_size']


#dev data
# to be able to do a correct evaluation, we pad a number of rows to get a multiple of the batch size
dev_sents1_extended = dataUtils.pad_to_batch_size(dev_sents1,hyperparas['batch_size'])
dev_sents2_extended = dataUtils.pad_to_batch_size(dev_sents2,hyperparas['batch_size'])
dev_y_extended = dataUtils.pad_to_batch_size(dev_y,hyperparas['batch_size'])
n_dev_batches = dev_sents1_extended.shape[0] / hyperparas['batch_size']
n_dev_samples = len(dev_y)
dataUtils.extend_lenghts(dev_lens1,hyperparas['batch_size'])
dataUtils.extend_lenghts(dev_lens2,hyperparas['batch_size'])

# test data
test_sents1_extended = dataUtils.pad_to_batch_size(test_sents1, hyperparas['batch_size'])
test_sents2_extended = dataUtils.pad_to_batch_size(test_sents2, hyperparas['batch_size'])
test_y_extended = dataUtils.pad_to_batch_size(test_y, hyperparas['batch_size'])
n_test_batches = test_sents1_extended.shape[0] / hyperparas['batch_size']
n_test_samples = len(test_y)
dataUtils.extend_lenghts(test_lens1,hyperparas['batch_size'])
dataUtils.extend_lenghts(test_lens2,hyperparas['batch_size'])

######################
# BUILD ACTUAL MODEL #
######################
print('Building the model')

# allocate symbolic variables for the data
SENT1_batch = T.imatrix('sent1')
SENT2_batch = T.imatrix('sent2')
y_batch = T.ivector('y')

# define/load the network
output_layer = networks.buildDCNNPaper(SENT1_batch, SENT2_batch, batch_size=hyperparas['batch_size'],vocab_size=hyperparas['vocab_size'],embeddings_size=hyperparas['word_vector_size'],filter_sizes=hyperparas['filter_size_conv_layers'],nr_of_filters=hyperparas['nr_of_filters_conv_layers'],activations=hyperparas['activations'],ktop=hyperparas['ktop'],dropout=hyperparas["dropout_value"],output_classes=hyperparas['output_classes'],padding='last')

# Kalchbrenner uses a fine-grained L2 regularization in the Matlab code, default values taken from Matlab code
# Training objective
l2_layers = []
for layer in lasagne.layers.get_all_layers(output_layer):
    if isinstance(layer,(DCNN.embeddings.SentenceEmbeddingLayer,DCNN.convolutions.Conv1DLayerSplitted,lasagne.layers.DenseLayer)):
        l2_layers.append(layer)

loss_train = lasagne.objectives.aggregate(lasagne.objectives.categorical_crossentropy(lasagne.layers.get_output(output_layer),y_batch),mode='mean') \
            + lasagne.regularization.regularize_layer_params_weighted(dict(zip(l2_layers,hyperparas["L2"])),lasagne.regularization.l2)

# validating/testing
loss_eval = lasagne.objectives.categorical_crossentropy(lasagne.layers.get_output(output_layer, deterministic=True),y_batch)
pred = T.argmax(lasagne.layers.get_output(output_layer , deterministic=True),axis=1)
correct_predictions = T.eq(pred, y_batch)

# In the matlab code, Kalchbrenner works with a adagrad reset mechanism, if the para --adagrad_reset has value 0, no reset will be applied
all_params = lasagne.layers.get_all_params(output_layer)
updates, accumulated_grads = utils.adagrad(loss_train, all_params, hyperparas['learning_rate'])
#updates = lasagne.updates.adagrad(loss_train, all_params, hyperparas['learning_rate'])


train_model = theano.function(inputs=[SENT1_batch,SENT2_batch,y_batch], outputs=loss_train,updates=updates, on_unused_input='warn')

valid_model = theano.function(inputs=[SENT1_batch,SENT2_batch,y_batch], outputs=correct_predictions, on_unused_input='warn')

test_model = theano.function(inputs=[SENT1_batch,SENT2_batch,y_batch], outputs=correct_predictions, on_unused_input='warn')



###############
# TRAIN MODEL #
###############
print('Started training')
print('Because of the default high validation frequency, only improvements are printed.')

best_validation_accuracy = 0
epoch = 0
batch_size = hyperparas["batch_size"]
while (epoch < hyperparas['n_epochs']):
    epoch = epoch + 1
    permutation = numpy.random.permutation(n_train_batches)
    batch_counter = 0
    train_loss=0
    for minibatch_index in permutation:
        train_sents1_input = train_sents1[minibatch_index*batch_size:(minibatch_index+1)*batch_size,0:train_lens1[(minibatch_index+1)*batch_size-1]]
        train_sents2_input = train_sents2[minibatch_index*batch_size:(minibatch_index+1)*batch_size,0:train_lens2[(minibatch_index+1)*batch_size-1]]
        train_y_input = train_y[minibatch_index*batch_size:(minibatch_index+1)*batch_size]
        train_loss+=train_model(train_sents1_input, train_sents2_input, train_y_input)
       
        #print '\n\n'
        #print train_lens1[(minibatch_index+1)*batch_size-1]
        #print train_sents1_input
        #print '\n\n'

        if batch_counter>0 and batch_counter % hyperparas["valid_freq"] == 0:
            accuracy_valid=[]
            for minibatch_dev_index in range(n_dev_batches):
                dev_sents1_input = dev_sents1_extended[minibatch_dev_index*batch_size:(minibatch_dev_index+1)*batch_size,0:dev_lens1[(minibatch_dev_index+1)*batch_size-1]]
                dev_sents2_input = dev_sents2_extended[minibatch_dev_index*batch_size:(minibatch_dev_index+1)*batch_size,0:dev_lens2[(minibatch_dev_index+1)*batch_size-1]]
                dev_y_input = dev_y_extended[minibatch_dev_index*batch_size:(minibatch_dev_index+1)*batch_size]
                accuracy_valid.append(valid_model(dev_sents1_input, dev_sents2_input, dev_y_input))

            #dirty code to correctly asses validation accuracy, last results in the array are predictions for the padding rows and can be dumped afterwards
            this_validation_accuracy = numpy.concatenate(accuracy_valid)[0:n_dev_samples].sum()/float(n_dev_samples)

            if this_validation_accuracy > best_validation_accuracy:
                print("Train loss, "+str( (train_loss/hyperparas["valid_freq"]))+", validation accuracy: "+str(this_validation_accuracy*100)+"%")
                best_validation_accuracy = this_validation_accuracy

                # test it
                accuracy_test= []
                for minibatch_test_index in range(n_test_batches):
                    sents1_input = test_sents1_extended[minibatch_test_index*batch_size:(minibatch_test_index+1)*batch_size,0:test_lens1[(minibatch_test_index+1)*batch_size-1]]
                    sents2_input = test_sents2_extended[minibatch_test_index*batch_size:(minibatch_test_index+1)*batch_size,0:test_lens2[(minibatch_test_index+1)*batch_size-1]]
                    y_input = test_y_extended[minibatch_test_index*batch_size:(minibatch_test_index+1)*batch_size]
                    accuracy_test.append(test_model(sents1_input, sents2_input, y_input))
                this_test_accuracy = numpy.concatenate(accuracy_test)[0:n_test_samples].sum()/float(n_test_samples)
                print("Test accuracy: "+str(this_test_accuracy*100)+"%")

            train_loss=0
        batch_counter+=1

    if hyperparas["adagrad_reset"] > 0:
        if epoch % hyperparas["adagrad_reset"] == 0:
            utils.reset_grads(accumulated_grads)

    print("Epoch "+str(epoch)+" finished.")



