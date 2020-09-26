import pickle
import matplotlib.pyplot as plt
import os

KEY = ['loss', 'acc', 'val_loss', 'val_acc']
def read_history(filename):
	f = open(filename,'rb')
	history = pickle.load(f)
	return history

def save_history_plot(history):
	for index, val in enumerate(KEY):
		plt.subplot(2, 2,index+1)
		plt.plot(history[val])
		plt.xlabel('epoches')
		plt.ylabel(val)
		plt.title(val)
		plt.grid(True)
		plt.tight_layout()
	plt.savefig('scheme.png')
	plt.show()

res = read_history('p3_history.pickle')
save_history_plot(res)
