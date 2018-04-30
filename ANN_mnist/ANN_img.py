from nltk.stem.porter import PorterStemmer         # for the porter stemmer
import re                                          # regular expression module - used to separate words in message
import numpy as np
from scipy.sparse import csr_matrix
import random
import time
import cPickle
import gzip


# Third-party libraries
import numpy as np
import matplotlib.pyplot as plt                    # for plotting results


learning_rate=0.01
MAX_ITERATIONS=500000



def load_data():
    f = gzip.open('mnist.pkl.gz', 'rb')
    training_data, validation_data, test_data = cPickle.load(f)
    f.close()
    return (training_data, validation_data, test_data)

def load_data_wrapper():
    tr_d, va_d, te_d = load_data()
    training_inputs = [(np.transpose(np.reshape(x, (784, 1)))) for x in tr_d[0]]              # getting 1 X 784 type array
    training_inputs=[np.insert(a,0,1,axis=1) for a in training_inputs]
    training_results = [vectorized_result(y) for y in tr_d[1]]
    training_data = (training_inputs, training_results)
    #validation_inputs = [(np.transpose(np.reshape(x, (784, 1)))) for x in va_d[0]]
    #validation_results=[vectorized_result(y) for y in va_d[1]]
    #validation_data = (validation_inputs, validation_results)
    test_inputs = [(np.transpose(np.reshape(x, (784, 1)))) for x in te_d[0]]
    test_inputs=[np.insert(a,0,1,axis=1) for a in test_inputs]
    test_results=[vectorized_result(y) for y in te_d[1]]
    test_data = (test_inputs, test_results)
    return (training_data, test_data)

def vectorized_result(j):
    e = np.zeros((1,10))
    e[0][j] = 1.0
    return e



def activation_fn(z):
    return 1.0/(1.0+np.exp(-z)) 
    
def activation_differential(z):
    return activation_fn(z)*(1-activation_fn(z))
    
    
def feed_forward(first_input,weights):
    
    '''
    Does the forward pass over the network and returns a
    list of ALL the outputs WITHOUT ACTIVATION FN APPLICATION of ALL layers.
    Params:
            sparse (scipy csr) array of 1-hot encodings of the message
            list of weights (numpy matrices)
    Returns:
            list of outputs (1Xn) (these will be useful for calculation of gradient with squared error cost)
    '''  
    output=[]
    output.append(first_input)
    # for rest of the operations, numpy arrays are available and used directly
    a=np.dot(first_input,weights[0])
    b=a
    a=np.insert(a,0,1,axis=1)     # inserting the fixed neuron - NOT included in the output list
    output.append(a)              # NOTE that the list of outputs is essenitally the values BEFORE APPLICATION OF ACTIVATION FN
    a=activation_fn(b)
    a=np.insert(a,0,1,axis=1)     # inserting the fixed neuron - NOTE that the activation was NOT to be applied on this value


    number_iterations=0
    for w in weights[1:]:
        number_iterations+=1
        a = (np.dot(a,w))
        b=a
        if number_iterations!=len(weights)-1:
            a=np.insert(a,0,1,axis=1)     # inserting the fixed neuron - NOT included in the output list
        output.append(a)
        a=activation_fn(b)
        a=np.insert(a,0,1,axis=1)

    return output
    
def delta_fn(weights,outputs,target_val):
    '''
    Finds and returns the delta values for ALL layers for every neuron.
    works for SQUARED ERROR function only
    params:
       weights: list of numpy matrices
       outputs: the list of ALL outputts of all layers 
       target_val: numpy array (1xn) of the target values
    returns:
        a list of numpy (1Xn) arrays, one for each layer
    '''
    delta=[]
    for i in reversed(range(1,len(weights)+1)):
        #sepcial case - the final layer
        if(i==len(weights)):
            #printing values
            #print "predicted: ",np.argmax(outputs[i]),
            #print "  real: ",np.argmax(target_val)

            diff=np.subtract(activation_fn(outputs[i]),target_val)
            diff=2*diff
            prev_del=np.multiply(diff,activation_differential(outputs[i]))
            delta.append(prev_del)
        else:
            weight_to_use=weights[i][1:]
            term_2=np.transpose(np.inner(weight_to_use,prev_del)) #an array of second terms of the backprop eqn
            term_1=activation_differential(outputs[i])[:,1:]#1-(np.square(outputs[i][:,1:]))
            prev_del=np.multiply(term_1,term_2)
            delta.append(prev_del)
    delta.reverse()
    return delta

    
def backprop(weights,first_input,learning_rate,target_val):
    '''
    does the backpropogation
    params:
        weights: list of numpy matrices
        first_input: the sparse array - the 1 hot vector encodings        
    Returns:
        new weights
    '''
    #doing the forward and the backward passes and getting values
    outputs=feed_forward(first_input,weights)
    delta=delta_fn(weights,outputs,target_val)
    weights_new=[None for i in weights]                  # of the same size as before  
    for l in reversed(range(0,len(outputs)-1)):          # layer number
        prod=np.dot(np.transpose(outputs[l]),delta[l])
        weights_new[l]=np.subtract(weights[l],(learning_rate*prod))

    return weights_new
    
def initiate_network(sizes):
    
    ''' 
    This function initializes the network -
    Params:
            a list of sizes (#nerurons) of each layer - INCLUDING INPUT and OUTPUT LAYERS
    Returns:
            a list of weight matrices (as numpy matrices)
    '''

    sizes=[size+1 for size in sizes]            # to account for x_0 - the first neruron in each layer with fixed output = 1
    weights = [np.random.randn(x,y-1) for x, y in zip(sizes[:-1], sizes[1:])]
    return weights
       

def classify(weights,inputs):
    '''
    does the classification for all inputs and returns the result as another BINARY ARRAY - true or false

    params:
        weights: the weights of the trained classifier
        inputs: the inputs in form of a LIST of one-hot encodings

    returns:
        a list of classifications
    ''' 
    results=[]
    for inp in inputs:
        outputs=feed_forward(inp,weights)
        final_result=outputs[-1]
        e = np.exp(final_result)
        dist = e / np.sum(e)
        res=np.zeros((1,10))
        res[0][np.argmax(dist)]=1
        results.append(res)

    return np.concatenate(results)

def error(real,predicted):
    '''
    params: both as np arrays
    returns: float value : ratio of misclassified to total
    '''
    errors= (np.count_nonzero(np.subtract(real,predicted))*0.5)/float(real.shape[0])  #for every erraneous row, 2 elements would be non-zero
    return errors
    
def train_net():
    '''
    The main controller function
    '''

    start_time=time.time()


    #np.seterr(all="ignore")
    training_data, test_data=load_data_wrapper()#one_hot_encodings, message_class_list
        
    '''
    ignoring the validation data (we don't need validation for now):
    
    '''
    one_hot_encodings_train=np.asarray(training_data[0])
    one_hot_encodings_test=np.asarray(test_data[0])


    message_class_list_train=training_data[1]
    message_class_list_test=test_data[1]

    first_l=one_hot_encodings_train[0].shape[1]-1

    # train and test sets obtained

    ###################   THE PARAMETERS FOR CHANGING MODE/ ARCHITECTURE  ##################
    
    cutoff=0.005
    sizes=[first_l,200,100,50,10]
    show_images=False                   # please use with only a small number of iterations - the process remains stuck till you close the image window
    ########################################################################################



    weights=initiate_network(sizes)

    errors_in=[]
    errors_out=[]
    number_iterations=0
    E_in=100

    while E_in > cutoff or number_iterations<100:
        number_iterations+=1
        if number_iterations>MAX_ITERATIONS:
            break


        # index deciding which training set to send
        index=random.randint(0,len(one_hot_encodings_train)-1)
        #index=number_iterations%50000


        input_encoding=one_hot_encodings_train[index]
        classification=message_class_list_train[index]


        # for displaying the image currently being trained on
        if show_images:
            val=np.argmax(classification)
            pixels=np.delete(input_encoding,0,axis=1)
            pixels=pixels.reshape(28,28)
            plt.title("clasification : "+str(val))
            plt.imshow(pixels, cmap='gray')
            plt.show()


        # training on it
        weights=backprop(weights,input_encoding,learning_rate,np.asarray(classification))


        if number_iterations%50000==0:
            # E_in and E_out calculation
            print "#iterations: ",number_iterations," : ",
            train_classification=classify(weights,one_hot_encodings_train)
            E_in=error(np.concatenate(message_class_list_train),train_classification)


            test_classification=classify(weights,one_hot_encodings_test)
            E_out=error(np.concatenate(message_class_list_test),test_classification)

            print "E_in: ",E_in," ; E_out: ",E_out

            errors_in.append(E_in)
            errors_out.append(E_out)


    #plotting this whole mess for easy visualization
    l1,l2=plt.plot(np.array(range(1,len(errors_in)+1)),np.array(errors_in),np.array(range(1,len(errors_in)+1)),np.array(errors_out))
    plt.setp(l1,linewidth=1,color='r')
    plt.setp(l2,linewidth=2,color='g')
    plt.legend([l1,l2],['E_in','E_out'])
    plt.xlabel('#iterations (X 50000 )')
    plt.ylabel('ERRORS')
    plt.savefig("img_mnist_classification.png")

    print "time taken: ", time.time()-start_time," sec"


train_net()
    
