import pickle
import matplotlib.pyplot as plt
import numpy as np
import os

KEY = ['loss', 'accuracy', 'val_loss', 'val_accuracy']
def read_history(filename):
	f = open(filename,'rb')
	history = pickle.load(f)
	return history

def save_history_plot(history):
	for index, val in enumerate(KEY):
		plt.subplot(1, 2,index+1)
		plt.plot(history[val])
		plt.xlabel('epoches')
		plt.ylabel(val)
		plt.title(val)
		plt.grid(True)
		plt.tight_layout()
	plt.savefig('scheme.png')
	plt.show()

res = read_history('history.pickle')
print(res)
#save_history_plot(res)

plt.plot(0.01 * np.exp(-0.1 * np.arange(80)), label='learningrate')
plt.xlabel('epoches')
plt.title('learningrate')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
