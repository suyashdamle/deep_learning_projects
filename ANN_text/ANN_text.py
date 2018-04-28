from nltk.stem.porter import PorterStemmer         # for the porter stemmer
import re                                          # regular expression module - used to separate words in message
import numpy as np
from scipy.sparse import csr_matrix
import random
import matplotlib.pyplot as plt                    # for plotting results



learning_rate=0.05
MAX_ITERATIONS=40000

def get_preprocessed_data(file_name):
    
    '''
    This function returns a list of one-hot encodings of the messages,
    and the corresponding classifications of the messages

    Params: the file name to read from

    Returns: 
        the one-hot encodings - a list of scipy sparse CSR arrays of (1 X N) dimension
        the classifications of the messages as python (same index as that of the message)
    '''    
    
    # the list of stopwords imported
    stop_words=[]
    with open("stop_words.txt",'r') as sw_file:
        for line in sw_file:
            stop_words.append(line.split(" ")[0].strip())
    
    
    # generating a set of english stop words
    stop_words=set(stop_words)
    
    
    message_class_list=[]                               # the ordered list of all classifications - as 0(spam) or 1(ham)
    final_messages_list=[]                              # the ordered list of all messages AFTER PREPROCESSING 
    all_words=dict()                                    # a dictionary data structure with words as keys and indices as values
    stemmer = PorterStemmer()                           # stemmer object used for PORTER STEMMING 
    
    # NOTE: keeping index 0 for fixed neuron with output of 1
    word_index=1                                        # the index of new word to be inserted  in the dictionary                      
    
    # reading the file to get < type , message > tuples and pre-processing the messages:
    with open(file_name,'r') as input_file:
        for line in input_file:
            line=line.split('\t')
            
            message_class=line[0]
            if message_class=='spam':
                message_class_list.append(0)
            elif message_class=='ham':
                message_class_list.append(1)
            else:
                print "Exceptional Case encountered: ", line
                exit()
                
            # note that the message could itself have tabs
            message= '\t'.join(line[1:])
            
            message=re.findall(r"[\w']+", message)
    
            
            #preprocessing the message: removing stop words
            processed_message = [w.lower() for w in message if not w.lower() in stop_words]
            
            # applying PORTER STEMMING operation
            processed_message=[stemmer.stem(word) for word in processed_message]

            
            # adding the processed message words as a string in final_message_list
            if len(processed_message) != 0:
                final_messages_list.append(' '.join(processed_message))
            else:
                processed_message=['BLANK_MESSAGE']
                final_messages_list.append('BLANK_MESSAGE')
                
            # putting new words in dictionary
            all_words["FIRST_INPUT"]=0
            for word in processed_message:
                if word not in all_words:
                    all_words[word]=word_index
                    word_index+=1
                
    
    # now, once all words have been indexed, generating one-hot vector encodings for messages in final_message_list            
    
    # creating a sparse matrix - the One-Hot encodings and populating them
    
    one_hot_encodings=[]                                    # python list of one-hot encodings of the messages...
                                                            # ... The encodings are themselves SPARSE MATRICES
    for msg in final_messages_list:
        
        # lists for initializing sparse matrix
        row_index=[]
        col_index=[]
        data=[]
        
        # setting the value for first, fixed neuron
        row_index.append(0)
        col_index.append(0)
        data.append(1)
        
        for word in msg.split(' '):
            word_idx=all_words[word]
            if word_idx not in col_index:
                col_index.append(word_idx)
                data.append(1)
                row_index.append(0)
        # create a sparse matrix out of the 3 lists and put them in the one_hot_encodings list
        encoding=csr_matrix((data,(row_index,col_index)),shape=(1,len(all_words)))
        one_hot_encodings.append(encoding)
        
    # all one-hot encodings of the vectors obtained
    print "Pre-processing phase complete: "
    return one_hot_encodings, np.asarray(message_class_list)

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
    # the first layer ( ie, the input) has sparse array as outputs
    a=first_input.dot(weights[0])
    b=a
    a=np.insert(a,0,1,axis=1)     # inserting the fixed neuron - NOT included in the output list
    output.append(a)              # NOTE that the list of outputs is essenitally the values BEFORE APPLICATION OF ACTIVATION FN
    a=activation_fn(b)
    a=np.insert(a,0,1,axis=1)     # inserting the fixed neuron - NOT included in the output list

    # for rest of the operations, numpy arrays are available and used directly
    number_iterations=0
    for w in weights[1:]:
        number_iterations+=1
        a = (np.dot(a,w))
        b=a
        if number_iterations!=len(weights)-1:
            a=np.insert(a,0,1,axis=1)     # inserting the fixed neuron - NOT included in the output list
        output.append(a)
        a=activation_fn(b)
        a=a=np.insert(a,0,1,axis=1)

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
    #doing the forward and the backward passes and gettin values
    outputs=feed_forward(first_input,weights)
    delta=delta_fn(weights,outputs,target_val)
    weights_new=[None for i in weights]                  # of the same size as before  
    for l in reversed(range(0,len(outputs)-1)):          # layer number
        # special case : if the first layer of outputs is concerned:
        if l==0:
            prod=(outputs[l].transpose()).dot(delta[l])
        else:
            prod=np.dot(np.transpose(outputs[l]),delta[l])
        weights_new[l]=np.subtract(weights[l],(learning_rate*prod))
    return weights_new
    
def initiate_network(sizes):
    
    ''' 
    This function initializes the network -
    Params:
            a list of sizes (#nerurons) of each layer - INCLUDING INPUT and OUTPUT LAYERS
    Returns:
            a list of weight matrices
    '''

    sizes=[size+1 for size in sizes]            # to account for x_0 - the first neruron in each layer with fixed output = 1
    weights = [np.random.randn(x,y-1) for x, y in zip(sizes[:-1], sizes[1:])]
    return weights
       

def classify(weights,inputs,threshold):
    '''
    does the classification for all inputs and returns the result as another BINARY ARRAY - true or false

    NOTE:   in case of 2 output neurons, undecided result may be returned (ie, NONE) if threshold conditions do not hold;
            also, in this case, the first neuron is assumed to correspond to TRUE case, and the second to FALSE case
    params:
        weights: the weights of the trained classifier
        inputs: the inputs in form of a LIST of one-hot encodings
        threshold: the min value above which a TRUE value is returned for that input

    returns:
        a list of bools
    ''' 
    results=[]
    for inp in inputs:
        outputs=feed_forward(inp,weights)
        final_result=outputs[-1]
        if final_result.shape[1]==1:        # one op neuron
            if final_result[0][0]>=threshold:
                results.append(True)
            else:
                results.append(False)
        else:                               # 2 output neurons
            
            first_op=outputs[-1][0][0]
            sec_op=outputs[-1][0][1]
            if first_op*sec_op > 0:
                sum_array=abs(np.sum(outputs[-1]))
                final_layer=np.divide(outputs[-1],sum_array)
                if final_layer[0][0]>final_layer[0][1]:
                    results.append(True)
                else:
                    results.append(False)
            else:
                # dealing with one-negative output
                if first_op<0:
                    results.append(False)
                else:
                    results.append(True)



    return results

def squared_error(inputs,outputs):
    '''
    params: both as np arrays
    returns: integer value for squared error output
    '''
    return np.sum(np.square(np.subtract(inputs,outputs)))/float(inputs.shape[0])


    
def train_net():
    '''
    The main controller function
    '''
    np.seterr(all="ignore")
    one_hot_encodings, message_class_list=get_preprocessed_data("Assignment_2_data.txt")
        
    '''
    dividing into training and test sets in the ratio  8:2
    
    '''
    random.seed(0)

    total_data=len(one_hot_encodings)
    choice=np.arange(total_data)
    np.random.shuffle(choice)

    # we now have a list of randomized indices - rearranging the data according to these and taking first 80% as train set
    # this method ensures that the order of the message and the corresponding classification is preserved...
    #...the mapping of message to class is intact

    one_hot_encodings=[one_hot_encodings[i] for i in choice]
    message_class_list=message_class_list[choice]


    one_hot_encodings_train=np.asarray(one_hot_encodings[:int(0.8*total_data)])
    one_hot_encodings_test=np.asarray(one_hot_encodings[-int(0.2*total_data):])

    message_class_list_train=message_class_list[:int(0.8*total_data)]
    message_class_list_test=message_class_list[-int(0.2*total_data):]

    # train and test sets obtained

    # splitting the train test on the basis of the classification - for better representation in the final model
    one_hot_encodings_train_0_class=one_hot_encodings_train[message_class_list_train==0]    # set of messages with class=0
    one_hot_encodings_train_1_class=one_hot_encodings_train[message_class_list_train==1]    # set of messages with class=1
    first_l=one_hot_encodings_train[0].shape[1]-1




    ###################   THE PARAMETERS FOR CHANGING MODE/ ARCHITECTURE  ##################
    
    cutoff=0.01
    output_neurons=2
    sizes=[first_l,100,50,output_neurons]

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

        if number_iterations%100==0:
            print "#iterations: ",number_iterations

        # randomly deciding whether to send a class=0 or class=1 type message
        message_class_to_send=random.randint(0,1)#(0,len(one_hot_encodings)-1)

        # message type decided
        if message_class_to_send==0:
            random_no=random.randint(0,len(one_hot_encodings_train_0_class)-1)
            if output_neurons==1:
                classification=0
            else:
                classification=[0,1]

            input_encoding=one_hot_encodings_train_0_class[random_no]

        else:
            random_no=random.randint(0,len(one_hot_encodings_train_1_class)-1)
            if output_neurons==1:
                classification=1
            else:
                classification=[1,0]

            input_encoding=one_hot_encodings_train_1_class[random_no]




        # training on it
        weights=backprop(weights,input_encoding,learning_rate,np.asarray([classification]))

        if number_iterations%500==0:
            # E_in and E_out calculation
            train_classification=classify(weights,one_hot_encodings_train,0.8)
            E_in=squared_error(message_class_list_train,np.asarray(train_classification))


            test_classification=classify(weights,one_hot_encodings_test,0.8)
            E_out=squared_error(message_class_list_test,np.asarray(test_classification))

            errors_in.append(E_in)
            errors_out.append(E_out)
            #print test_classification.count(False)," / ",
            #print len(test_classification)
        


    #plotting the entire mess for easy visualization
    l1,l2=plt.plot(np.array(range(1,len(errors_in)+1)),np.array(errors_in),np.array(range(1,len(errors_in)+1)),np.array(errors_out))
    plt.setp(l1,linewidth=1,color='r')
    plt.setp(l2,linewidth=2,color='g')
    plt.legend([l1,l2],['E_in','E_out'])
    plt.xlabel('#iterations (X 500 )')
    plt.ylabel('ERRORS')
    plt.savefig("img_text_classification.png")

    
train_net()
    
