import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from torch.autograd import Variable
from torch import optim
#from data_util import load_mnist
from sklearn.datasets import load_digits
import time
import argparse

parser = argparse.ArgumentParser(description='Word Embedding Models')
parser.add_argument('--model', default='pca', type=str,
	                    help='Select Word Embedding Model. Valid configurations are pca, pca3, max-pool, max-pool3, max-pool-pca and max-pool-pca3')
args = parser.parse_args()
if not args.model in ['pca','pca3','max-pool','max-pool3','max-pool-pca','max-pool-pca3']:
	print "Invalid Model Initialization:"
	print "Not one of: \n pca \n pca3 \n max-pool \n max-pool3 \n max-pool-pca \n max-pool-pca3"
	exit()

if(args.model=='pca'):
	filename='Xpca.p'
elif(args.model=='pca3'):
	filename='Xpca3.p'
elif(args.model=='max-pool'):
	filename='Xmax.p'
elif(args.model=='max-pool3'):
	filename='Xmax3.p'
elif(args.model=='max-pool-pca'):
	filename='Xmpca.p'
elif(args.model=='max-pool-pca3'):
	filename='Xmpca3.p'


def build_model(input_dim, output_dim):
    model = torch.nn.Sequential()
    model.add_module("linear",
                     torch.nn.Linear(input_dim, output_dim, bias=False))
    return model

def train(model, loss, optimizer, x_val, y_val):
    x = Variable(x_val, requires_grad=False)
    y = Variable(y_val, requires_grad=False)

    # Reset gradient
    optimizer.zero_grad()

    # Forward
    fx = model.forward(x)
    output = loss.forward(fx, y)

    # Backward
    output.backward()

    # Update parameters
    optimizer.step()

    return output.data[0]


def predict(model, x_val):
    x = Variable(x_val, requires_grad=False)
    output = model.forward(x)
    return output.data.numpy().argmax(axis=1)

def main():
	with open ('Y.p','r') as f:
		trY = np.load(f)
	with open(filename,'r') as f:
		trX = np.load(f)
	

	p = np.random.permutation(len(trX))
	trX=trX[p]
	trY=trY[p]
	test_length	= 2000
	
	print trX

	teX=trX[-test_length:]
	teY=trY[-test_length:]

	trX=trX[:-test_length]
	trY=trY[:-test_length]

	xscale=StandardScaler()
	trX=xscale.fit_transform(trX)
	teX=xscale.transform(teX)
	print teX.shape
	print trX.shape

	trX = torch.from_numpy(trX).float()
	teX = torch.from_numpy(teX).float()
	trY = torch.from_numpy(trY).long()
	teY = torch.from_numpy(teY).long()

	n_examples, n_features = trX.size()
	n_classes = 2
	model = build_model(n_features, n_classes)
	loss = torch.nn.CrossEntropyLoss(size_average=True)
	optimizer = optim.SGD(model.parameters(), lr=0.00001, momentum=0.9)
	batch_size = 100
	s=time.time()
	#print trX[0],trY[0]
	prev_cost=0
	for i in range(1000):
		cost = 0.
		num_batches = n_examples // batch_size
		for k in range(num_batches):
			start, end = k * batch_size, (k + 1) * batch_size
			cost += train(model, loss, optimizer,
				trX[start:end], trY[start:end])
		if np.abs(prev_cost-cost) <=0.00001:
			break
		prev_cost=cost
		print("Epoch %d, cost = %f"
		      % (i + 1, cost / num_batches))
	e=time.time()
	print("time =",e-s)
	res = predict(model,teX).reshape(-1,1)
	teY=teY.numpy()
	acc=0
	tp=0
	fp=0
	fn=0
	for i in range(test_length):
		if res[i]==teY[i]:
			acc+=1
			if(res[i]==1):
				tp+=1
		elif res[i]==1:
			fp+=1
		elif res[i]==0:
			fn+=1
			
	print "acc",acc
	print "tp",tp
	print "fp",fp
	print "fn",fn
	acc=acc*1.0/(test_length)

	prec= tp*1.0/(tp+fp)
	recall = tp*1.0/(tp+fn)
	print "accuracy ", acc
	print "precision ", prec
	print "recall " , recall
	print "F1 score "  , 2*(prec*recall)/(prec+recall)

main()