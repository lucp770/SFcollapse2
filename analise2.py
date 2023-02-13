import os
import numpy as np
from matplotlib import pyplot as plt
import utilities

# uma funcao do segundo grau
def funcao(x):
	resultado = x**2 - 2*x 
	return resultado

def is_inclination_positive(ponto, funcao, delta =1e-6):
	x0 = ponto
	x1 = ponto + delta
	y0 = funcao(x0)
	y1 = funcao(x1)

	inclination = (y1-y0)/(x1-x0)

	if inclination > 0 : return True
	else: return False

x = np.linspace(-5,5,100)
eixo = [0 for i in x]

pontox = -2
pontoy = funcao(pontox)

print(is_inclination_positive(pontox,funcao))

y = funcao(x)

plt.plot(x,y)
plt.plot(x, eixo, '-k')
plt.scatter(pontox,pontoy)
plt.show()
