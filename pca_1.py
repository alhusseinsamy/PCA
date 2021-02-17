import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt

def pca(data):
	rows = 0
	cols = 0

	for row in data:
		rows+=1

	for column in data.T:
		cols+=1

	mn = np.mean(data.T[0])
	conc = data.T[0] - mn
	conc = conc.reshape(len(conc), 1)

	# print(conc)


	for column in data.T[1:]:
		mean = np.mean(column)
		col = column - mean

		col = col.reshape(len(col), 1)

		# print(col.shape)

		conc = np.hstack((conc, col))

	ret = np.zeros([cols, cols], dtype = float)
	for c in conc:
		c1 = c.reshape(cols,1)

		res = c * c1
		ret += res

	ret = ret/rows

	w, v = LA.eig(ret)
	# print(v)

	return v, conc, data


with open('Data.txt') as f:
	data = []
	for line in f:
		tmp = line.split(' ')
		arr = []
		for x in tmp:
			arr.append(x.replace('\n', ''))
		data.append(arr)

	data = np.array(data).astype(np.float)

	vecs, dta, original_data = pca(data)

	fig = plt.figure()
	ax = fig.add_axes([0.1,0.1,0.8,0.8])

	ax.scatter(dta.T[0], dta.T[1], c='r')
	


	ax.quiver(vecs[0][0], vecs[1][0], color='green', scale=4)
	ax.quiver(vecs[0][1], vecs[1][1], color='blue', scale=4)

	plt.savefig('Data_PCA.png')

	plt.show()

	






