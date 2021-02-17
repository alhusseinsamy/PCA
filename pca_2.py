import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt
import pandas as pd

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

	# print(v.shape)

	return v, conc, data, rows, cols


df = pd.read_excel('EPL.xlsx')

new_df = df.drop(['HomeTeam', 'AwayTeam', 'FTR'], axis=1)	

data = np.array(new_df)

vecs, dta, original_data, rows, cols = pca(data)

ident = 1

for v in vecs:
	plt.figure()
	v1 = v.T
	data_transposed = original_data.T
	# print(data_transposed.shape)

	dd = np.dot(v1,data_transposed)

	dd = dd.reshape(rows,)

	# print(dd.shape)

	home = []
	away = []
	for i in range(0, len(dd)):
		if(df['FTR'][i] == 'H'):
			home.append(dd[i])
		else:
			away.append(dd[i])
	# print(home[0])	
	# print(len(away))		


	num_bins = 10
	plt.hist([home,away], num_bins, edgecolor='white', color=['red', 'blue'], linewidth=1, alpha=1)

	plt.savefig('Proj_PC'+str(ident))
	ident+=1

x = [0,1,2,3,4,5,6,7]
differences = []

for v in vecs:
	v1 = v.T
	data_transposed = original_data.T

	dd = np.dot(v1,data_transposed)

	dd = dd.reshape(rows,)


	home = []
	away = []
	for i in range(0, len(dd)):
		if(df['FTR'][i] == 'H'):
			home.append(dd[i])
		else:
			away.append(dd[i])		

	home = np.array(home)
	mean_home = np.mean(home)	
	away = np.array(away)
	mean_away = np.mean(away)
	diff = abs(mean_home - mean_away)
	differences.append(diff)


fig = plt.figure()

plt.scatter(x, differences, c='r')

plt.savefig('Distance.png')

plt.show()






	# dd = dd.reshape(281,)

	# home = []
	# away = []