import numpy as np
import time
import pca
from sklearn.decomposition import PCA
from collections import defaultdict
import argparse
from cvxpy import *
import cvxpy as cvx


def init():
	#Loading in arguments
	global args,word_vec,sentences,Y
	parser = argparse.ArgumentParser(description='Word Embedding Models')
	parser.add_argument('--model', default='pca', type=str,
	                    help='Select Word Embedding Model. Valid configurations are pca, pca3, avg, avg3, max-pool, max-pool3, max-pool-pca and max-pool-pca3')

	args = parser.parse_args()
	if not args.model in ['pca','pca3','avg','avg3','max-pool','max-pool3','max-pool-pca','max-pool-pca3']:
		print "Invalid Model Initialization:"
		print "Not one of: \n pca \n pca3 \n avg \n avg3 \n max-pool \n max-pool3 \n max-pool-pca \n max-pool-pca3"
		exit()
	#Reading from pre-trained word-embeddings
	start=time.time()
	with open ('glove.6B.300d.txt','r') as f:
		for line in f:
			word, vec = line.split(' ')[0], np.asarray(map(float,line.split(' ')[1:]))
			word_vec[word]=np.asarray(vec).reshape(-1,1)
	end=time.time()
	print('loading time',end-start)

	Y=[]
	with open('train_Y.dat','r') as f:
		for line in f:
			if float(line) >= 0.5:
				Y.append(1)
			else:
				Y.append(0)
	Y=np.asarray(Y)
	
	#converting sentences to numpy arrays
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

def max_pool_pca(s):
	length=s.shape[0]
	z=Variable(length,length)
	wi=np.dot(s,s.T)
	objective = Minimize(trace(z))
	constraints = [z>=wi]# [norm(A) <= norm(B)] 
	prob = Problem(objective, constraints)
	prob.solve()
	s=np.asarray(z.value)
	return np.diag(s)

def main():
	global args,word_vec,sentences,Y
	word_vec={}
	init()
	print "Currently training: " + args.model
	W=[]
	l=word_vec['.'].shape[0]
	default_val=np.zeros(l).reshape(-1,1)

	for num_sent,sentence in enumerate(sentences):
		s=[]
		for i,word in enumerate(sentence):
			
			if args.model=="pca" or args.model=="max-pool" or args.model=="avg" or args.model=="max-pool-pca":
				x=word_vec.get(word,default_val)
			else:
				if i==0:
					x=default_val
					x=np.concatenate((x,word_vec.get(word,default_val)),axis=0)
					x=np.concatenate((x,word_vec.get(sentence[i+1],default_val)),axis=0)
				if i==len(sentence)-1:
					x=word_vec.get(sentence[i-1],default_val)
					x=np.concatenate((x,word_vec.get(word,default_val)),axis=0)
					x=np.concatenate((x,default_val),axis=0)
				else:
					x=word_vec.get(sentence[i-1],default_val)
					x=np.concatenate((x,word_vec.get(sentence[i],default_val)),axis=0)
					x=np.concatenate((x,word_vec.get(sentence[i+1],default_val)),axis=0)
			if s==[]:
				s=x
			else:
				if args.model=='max-pool' or args.model=="max-pool3":
					s[s>0] = np.fmax(np.abs(x)[s>0],np.abs(s)[s>0])
					s[s<=0] = -1*np.fmax(np.abs(x)[s<=0],np.abs(s)[s<=0])
				elif args.model=='avg' or args.model=='avg3':
					s=x+s
				else:
					s=np.concatenate((s,x),axis=1)
		
		if (args.model=='pca' or args.model=='pca3'):	
			p = PCA(n_components=3)
			s = p.fit_transform(s)

		if (args.model=='avg' or args.model=='avg3'):	
			s=s/(len(sentence))

		if (args.model=="max-pool-pca" or args.model=="max-pool-pca3"):
			s=max_pool_pca(s)

		s=s.reshape(1,-1)
		if W==[]:
			W=s		
		else:
			W=np.concatenate((W,s),axis=0)
		if num_sent%100==0:
			print W.shape
		num_sent+=1	

	with open ('Y.p','w') as f:
		print 'y',Y.shape
		np.save(f,Y)


	if(args.model=='pca'):
		filename='Xpca.p'
	elif(args.model=='pca3'):
		filename='Xpca3.p'
	elif(args.model=='max-pool'):
		filename='Xmax.p'
	elif(args.model=='max-pool3'):
		filename='Xmax3.p'
	elif(args.model=='avg'):
		filename='Xavg.p'
	elif(args.model=='avg3'):
		filename='Xavg3.p'
	elif(args.model=='max-pool-pca'):
		filename='Xmpca.p'
	elif(args.model=='max-pool-pca3'):
		filename='Xmpca3.p'

	with open(filename,'w') as f:
		np.save(f,W)

	with open ('Y.p','r') as f:
		print "loading y",np.load(f).shape
	 
	with open(filename,'r') as f:
		print "loading x",np.load(f).shape


if __name__ == '__main__':
	main()