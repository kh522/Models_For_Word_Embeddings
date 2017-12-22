import numpy as np
import pickle
import time
import pca
from sklearn.decomposition import PCA
from collections import defaultdict
import torch
from sklearn.preprocessing import StandardScaler
from torch.autograd import Variable
from torch import optim
import torch.nn as nn
#from data_util import load_mnist
from sklearn.datasets import load_digits
import time
import flops_counter as fc
import torch.nn.functional as F

word_vec={}
start=time.time()
with open ('glove.6B.300d.txt','r') as f:
	for line in f:
		word, vec = line.split(' ')[0], np.asarray(map(float,line.split(' ')[1:]))
#		word_vec[word]=vec
		word_vec[word]=np.asarray(vec).reshape(-1,1)

l=word_vec['.'].shape[0]
end=time.time()
print('loading time',end-start)

num_sent=0
data_matrix=[]
n_examples=10000
batch_size=1000
test_length=2000

Y=[]
with open('train_Y.dat','r') as f:
	for line in f:
		if float(line) >= 0.5:
			Y.append(1)
		else:
			Y.append(0)
Y=np.asarray(Y)
default_val=np.zeros(l).reshape(-1,1)
#print(word_vec['.'].shape)
#print(default_val.shape)
W=[]
sentences=[]

with open('train.dat','r') as f:
	for line in f:
		sent=line.split('@@@')
		sentence=[]
		for word in sent:
			if word[:2]=='~~'or word[-2:]=='~~':
				pass
			else:
				sentence.append(word)
		sentences.append(sentence)


class conv(nn.Module):
	def __init__(self):
		super(conv,self).__init__()
		self.conv1 = nn.Conv2d(1,3, (2,1),1)

	def forward(self,x):
		out=self.conv1(x)
		#out=F.max_pool2d(F.relu(self.conv1(x)),(19,1))
		return out

def build_model(input_dim, output_dim):
    # We don't need the softmax layer here since CrossEntropyLoss already
    # uses it internally.
    model = torch.nn.Sequential()
    model.add_module("linear",
                     torch.nn.Linear(input_dim, output_dim, bias=False))
    return model

def train(model, loss, optimizer, x_val, y_val):
    #x = Variable(x_val, requires_grad=False)
    x=x_val
    y = Variable(y_val, requires_grad=False)
    # Reset gradient
    optimizer.zero_grad()
    # Forward
    fx = model.forward(x)
    output = loss.forward(fx, y)
    # Backward
    s=time.time()
    output.backward(retain_graph=True)
    e=time.time()
    #print("time to compute backprop", float(e-s)/60)
    # Update parameters
    optimizer.step()
    return output.data[0]

def predict(model, x_val):
    #x = Variable(x_val, requires_grad=False)
    output = model.forward(x_val)
    return output.data.numpy().argmax(axis=1)


cnn=conv()
model = build_model(300,2)
loss = torch.nn.CrossEntropyLoss(size_average=True)
optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=0.)

rnns = nn.LSTM(300,1,1)
Y_=torch.from_numpy(Y).long()
num_epochs=2000
for epochs in range(num_epochs):
	
	for num_sent, sentence in enumerate(sentences):
		s=[]
		W=[]
		if num_sent>=n_examples:
			break

		for i,word in enumerate(sentence):		
			x=word_vec.get(word,default_val).reshape(1,-1)

			if s==[]:
				s=x
			else:
				s=np.concatenate((s,x),axis=0)
		
		#print s.shape

		s=Variable(torch.from_numpy(s.reshape(1,1,-1,300)).float(),requires_grad=False)
		out = cnn.forward(s)
		val=out[0][0].data.numpy()
		s=val[0]
		for x in val:
			s[s>0] = np.fmax(np.abs(x)[s>0],np.abs(s)[s>0])
			s[s<=0] = -1*np.fmax(np.abs(x)[s<=0],np.abs(s)[s<=0])
		W=Variable(torch.from_numpy(s.reshape(1,-1)))
		cost = 0.
		#e_time=time.time()
		#print("time for 1 sent",(e_time-s_time)/60)
		for k in range(1):
			#print("starting backprop, batch: ",k,"out of ", num_batches)

			#start, end = k * batch_size, (k + 1) * batch_size
			cost += train(model, loss, optimizer,
				W, Y_[num_sent:num_sent+1])
		#print("finished sent")
	print("Epoch %d, cost = %f"
		      % (epochs + 1, cost/2200))


	W=[]
	for num_sent, sentence in enumerate(sentences):
		s=[]
		if num_sent<=n_examples:
			continue
		if num_sent > n_examples+test_length:
			break
		for i,word in enumerate(sentence):
			x=word_vec.get(word,default_val).reshape(1,-1)
			# if i==0:
			# 	x=default_val
			# 	x=np.concatenate((x,word_vec.get(word,default_val)),axis=0)
			# 	x=np.concatenate((x,word_vec.get(sentence[i+1],default_val)),axis=0)
			# if i==len(sentence)-1:
			# 	x=word_vec.get(sentence[i-1],default_val)
			# 	x=np.concatenate((x,word_vec.get(word,default_val)),axis=0)
			# 	x=np.concatenate((x,default_val),axis=0)
			# else:
			# 	x=word_vec.get(sentence[i-1],default_val)
			# 	x=np.concatenate((x,word_vec.get(sentence[i],default_val)),axis=0)
			# 	x=np.concatenate((x,word_vec.get(sentence[i+1],default_val)),axis=0)
			# if s==[]:
			# 	s=x
			# else:
			# 	#max-pool
			# 	s[s>0] = np.fmax(np.abs(x)[s>0],np.abs(s)[s>0])
			# 	s[s<=0] = -1*np.fmax(np.abs(x)[s<=0],np.abs(s)[s<=0])	
			if s==[]:
				s=x
			else:
				s=np.concatenate((s,x),axis=0)
		s=Variable(torch.from_numpy(s.reshape(1,1,-1,300)).float(),requires_grad=False)
		out = cnn.forward(s)
		val=out[0][0].data.numpy()
		s=val[0]
		for x in val:
			s[s>0] = np.fmax(np.abs(x)[s>0],np.abs(s)[s>0])
			s[s<=0] = -1*np.fmax(np.abs(x)[s<=0],np.abs(s)[s<=0])
		out=Variable(torch.from_numpy(s.reshape(1,-1)))
		#print("sentence shape",out.squeeze())
		# try:
		# 	if W==[]:
		# 		W=out[-1].view(1,-1)		
		# 	else:
		# 		W=torch.cat((W,out[-1].view(1,-1)),0)
		# except:
		# 	W=torch.cat((W,out[-1].view(1,-1)),0)

		# if num_sent%100==0:
		# 	print "W shape", W.size()
		# num_sent+=1	
		#for torch versions
		try:
			if W==[]:
				W=out[-1].view(1,-1)		
			else:
				W=torch.cat((W,out[-1].view(1,-1)),0)
		except:
			W=torch.cat((W,out[-1].view(1,-1)),0)

		if num_sent%100==0:
			print W.size()

	# try:
	# 	W_=torch.from_numpy(W).float()
	# except:
	# 	pass
	res = predict(model,W).reshape(-1,1)

	acc=0
	tp=0
	fp=0
	fn=0
	for i in range(test_length):
		if res[i]==Y[n_examples+i]:
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
	try:
		acc=acc*1.0/(test_length)
		prec= tp*1.0/(tp+fp)
		recall = tp*1.0/(tp+fn)
		f1=2*(prec*recall)/(prec+recall)
	except:
		print "Divide By Zero"
		continue
	print "accuracy ", acc
	print "precision ", prec
	print "recall " , recall
	print "F1 score "  , f1