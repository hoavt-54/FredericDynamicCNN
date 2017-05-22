__author__ = 'Frederic Godin  (frederic.godin@ugent.be / www.fredericgodin.com)'
import lasagne
import DCNN
import theano.tensor as T


def parseActivation(str_a):
    if str_a=="linear":
        return lasagne.nonlinearities.linear
    elif str_a=="tanh":
        return lasagne.nonlinearities.tanh
    elif str_a=="rectify":
        return lasagne.nonlinearities.rectify
    elif str_a=="sigmoid":
        return lasagne.nonlinearities.sigmoid
    else:
        raise Exception("Activation function \'"+str_a+"\' is not recognized")


def buildDCNNPaper(input1, input2, batch_size,vocab_size,embeddings_size=48,filter_sizes=[10,7],nr_of_filters=[6,12],activations=["tanh","tanh"],ktop=5,dropout=0.5,output_classes=2,padding='last'):

    prem_l_in = lasagne.layers.InputLayer(
        shape=(batch_size, None),
        input_var = input1
    )

    hypo_l_in = lasagne.layers.InputLayer(
        shape=(batch_size, None),
        input_var = input2
    )

    prem_l_embedding = DCNN.embeddings.SentenceEmbeddingLayer(
        prem_l_in,
        vocab_size,
        embeddings_size,
        padding=padding
    )

    hypo_l_embedding = DCNN.embeddings.SentenceEmbeddingLayer(
        hypo_l_in,
        vocab_size,
        embeddings_size,
        padding=padding,
        W=prem_l_embedding.get_em_weights()
    )




    prem_l_conv1 = DCNN.convolutions.Conv1DLayerSplitted(
        prem_l_embedding,
        nr_of_filters[0],
        filter_sizes[0],
        nonlinearity=lasagne.nonlinearities.linear,
        border_mode="full"
    )
    

    hypo_l_conv1 = DCNN.convolutions.Conv1DLayerSplitted(
        hypo_l_embedding,
        nr_of_filters[0],
        filter_sizes[0],
        nonlinearity=lasagne.nonlinearities.linear,
        border_mode="full"
    )


    prem_l_fold1 = DCNN.folding.FoldingLayer(prem_l_conv1)
    hypo_l_fold1 = DCNN.folding.FoldingLayer(hypo_l_conv1)
    

    prem_l_pool1 = DCNN.pooling.DynamicKMaxPoolLayer(prem_l_fold1,ktop,nroflayers=2,layernr=1)
    hypo_l_pool1 = DCNN.pooling.DynamicKMaxPoolLayer(hypo_l_fold1,ktop,nroflayers=2,layernr=1)

    prem_l_nonlinear1 = lasagne.layers.NonlinearityLayer(prem_l_pool1,nonlinearity=parseActivation(activations[0]))
    hypo_l_nonlinear1 = lasagne.layers.NonlinearityLayer(hypo_l_pool1,nonlinearity=parseActivation(activations[0]))



    prem_l_conv2 = DCNN.convolutions.Conv1DLayerSplitted(
        prem_l_nonlinear1,
        nr_of_filters[1],
        filter_sizes[1],
        nonlinearity=lasagne.nonlinearities.linear,
        border_mode="full"
    )

    hypo_l_conv2 = DCNN.convolutions.Conv1DLayerSplitted(
        hypo_l_nonlinear1,
        nr_of_filters[1],
        filter_sizes[1],
        nonlinearity=lasagne.nonlinearities.linear,
        border_mode="full"
    )



    prem_l_fold2 = DCNN.folding.FoldingLayer(prem_l_conv2)
    hypo_l_fold2 = DCNN.folding.FoldingLayer(hypo_l_conv2)
    
    prem_l_pool2 = DCNN.pooling.KMaxPoolLayer(prem_l_fold2,ktop)
    hypo_l_pool2 = DCNN.pooling.KMaxPoolLayer(hypo_l_fold2,ktop)

    
    prem_l_nonlinear2 = lasagne.layers.NonlinearityLayer(prem_l_pool2,nonlinearity=parseActivation(activations[1]))
    hypo_l_nonlinear2 = lasagne.layers.NonlinearityLayer(hypo_l_pool2,nonlinearity=parseActivation(activations[1]))

    prem_l_dropout2=lasagne.layers.DropoutLayer(prem_l_nonlinear2,p=dropout)
    hypo_l_dropout2=lasagne.layers.DropoutLayer(hypo_l_nonlinear2,p=dropout)
    

    prem_rep = lasagne.layers.DenseLayer(
        prem_l_dropout2,
        num_units=100,
        nonlinearity=lasagne.nonlinearities.tanh
        )
    
    hypo_rep = lasagne.layers.DenseLayer(
        hypo_l_dropout2,
        num_units=100,
        nonlinearity=lasagne.nonlinearities.tanh
        )
    #return hypo_rep

    #concat here
    #fc = lasagne.layers.ConcatLayer([prem_rep, hypo_rep])
    mul = lasagne.layers.ElemwiseMergeLayer([prem_rep, hypo_rep], T.mul)
    #fc = lasagne.layers.DropoutLayer(fc, p=dropout)
    #fc = lasagne.layers.DenseLayer(fc_input,
    #            num_units=300,
    #            nonlinearity=lasagne.nonlinearities.tanh
    #        )
    fc_drop = lasagne.layers.DropoutLayer(mul, p=dropout)

    out_put = lasagne.layers.DenseLayer(
        fc_drop,
        num_units=output_classes,
        nonlinearity=lasagne.nonlinearities.softmax
        )

    return out_put




def buildMaxTDNN1(batch_size,vocab_size,embeddings_size,filter_size,output_classes):

    l_in = lasagne.layers.InputLayer(
        shape=(batch_size, None),
    )

    l_embedding = DCNN.embeddings.SentenceEmbeddingLayer(l_in, vocab_size, embeddings_size)


    l_conv1 = DCNN.convolutions.Conv1DLayer(
        l_embedding,
        1,
        filter_size,
        nonlinearity=lasagne.nonlinearities.tanh,
        stride=1,
        border_mode="valid",

    )

    l_pool1 = lasagne.layers.GlobalPoolLayer(l_conv1,pool_function=T.max)

    l_out = lasagne.layers.DenseLayer(
        l_pool1,
        num_units=output_classes,
        nonlinearity=lasagne.nonlinearities.softmax,
        )

    return l_out


