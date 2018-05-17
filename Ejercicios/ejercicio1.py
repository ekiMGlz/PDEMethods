import  sys
sys.path.append('../')
import  latexStrings  as ls
from IPython.display import Latex
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pdesolver

#Definimos el PVIF en eq, y la solucion exacta en u
u = lambda x, t: np.exp(-np.pi*t)*np.sin(np.pi*x)
eq= {}
eq['D'] = 1/np.pi
eq['ic'] = lambda x : np.sin(np.pi*x)
eq['bcL'] = lambda t : 0
eq['bcR'] = lambda t : -np.pi*np.exp(-np.pi*t)
Ix = [0, 1]
It = [0, 1]

#Resolvemos con M = 20, N = 256
M = 20
N = 256

W, X, T = pdesolver.explicitHeat(eq, Ix, It, M, N)
U = np.array([[u(x,t) for t in T] for x in X])
Error = max(abs(U[:,256]-W[:,256]))
print('Maximo Error en T=1: '+ str(Error))

#Plot de la solucion exacta
MeshT, MeshX = np.meshgrid(T, X)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_wireframe(MeshT, MeshX, U)

ax.text2D(0, 1, "Solucion Exacta", transform=ax.transAxes)
ax.set_xlabel('Tiempo (t)')
ax.set_ylabel('Espacio (x)')
ax.set_zlabel('u(t,x)')

#plt.savefig('../Graficas/fig1_1.png', dpi = 200)
plt.show()

#Plot de la solucion aproximada
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_wireframe(MeshT, MeshX, W)

ax.text2D(0, 1, "Solucion Metodo Explicito", transform=ax.transAxes)
ax.set_xlabel('Tiempo (t)')
ax.set_ylabel('Espacio (x)')
ax.set_zlabel('u(t,x)')

#plt.savefig('../Graficas/fig1_2.png', dpi = 200)
plt.show()