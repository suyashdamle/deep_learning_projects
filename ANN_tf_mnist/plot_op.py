import numpy as np
import matplotlib.pyplot as plt                    # for plotting results

file_name="op_1.txt"


E_in=[]
E_out=[]
epochs=[]
with open(file_name ,"r") as inp:
	for line in inp:
		params=line.split(';')
		for param in params:
			param=param.strip()
			vals=param.split(':')
			if vals[0].strip()=='epoch':
				epochs.append(vals[1].strip())
			elif vals[0].strip()=='E_in':
				E_in.append(vals[1].strip())
			elif vals[0].strip()=='E_out':
				E_out.append(vals[1].strip())

#plotting this whole mess for easy visualization
l1,l2=plt.plot(np.array(epochs),np.array(E_in),np.array(epochs),np.array(E_out))
plt.setp(l1,linewidth=1,color='r')
plt.setp(l2,linewidth=2,color='g')
plt.legend([l1,l2],['E_in','E_out'])
plt.xlabel('#epochs (X 20 )')
plt.ylabel('ERRORS')
plt.savefig("tf_mnist_classification.png")