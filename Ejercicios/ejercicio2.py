import  sys
sys.path.append('../')
import  latexStrings  as ls
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pdesolver


#Definimos el PVIF en eq, asi como su solucion exacta u
u = lambda x, t: np.exp(-np.pi*t)*np.sin(np.pi*x)
eq= {}
eq['D'] = 1/np.pi
eq['ic'] = lambda x : np.sin(np.pi*x)
eq['bcL'] = lambda t : 0
eq['bcR'] = lambda t : -np.pi*np.exp(-np.pi*t)
Ix = [0, 1]
It = [0, 1]

#Resolvemos con M = 20, N = 10
M = 20
N = 10

W, X, T = pdesolver.implicitHeat(eq, Ix, It, M, N)
U = np.array([[u(x,t) for t in T] for x in X])

#Plot de la solucion exacta
MeshT, MeshX = np.meshgrid(T, X)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_wireframe(MeshT, MeshX, U)

ax.text2D(0, 1, "Solucion Exacta", transform=ax.transAxes)
ax.set_xlabel('Tiempo (t)')
ax.set_ylabel('Espacio (x)')
ax.set_zlabel('u(t,x)')

#plt.savefig('../Graficas/fig2_1.png', dpi = 200)
plt.show()

#Plot de la aproximacion
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_wireframe(MeshT, MeshX, W)

ax.text2D(0, 1, "Solucion Metodo Implicito", transform=ax.transAxes)
ax.set_xlabel('Tiempo (t)')
ax.set_ylabel('Espacio (x)')
ax.set_zlabel('u(t,x)')

#plt.savefig('../Graficas/fig2_2.png', dpi = 200)
plt.show()

#Tabla de errores con M = [10, 20, 40], N = [16, 64, 256]
h = [10, 20, 40]
k = [16, 64, 256]
data = []

for i in range(3):
    M = h[i]
    N = k[i]
    W, X, T = pdesolver.implicitHeat(eq, Ix, It, M, N)
    U = u(X, 1)
    maxError = max(abs(U-W[:,-1]))
    eoc = 'NaN' if i == 0 else np.log(maxError/prevError)/np.log(h[i-1] / M)
    data.append([ maxError, eoc ])
    prevError = maxError

data = [['1/10', '1/16'] + data[0], ['1/20', '1/64'] + data[1], ['1/40', '1/256'] + data[2]]
header = ['h', 'k', 'Error', 'erc']
print(ls.latexTable(header, data, 'cc|rr'))

#Tabla de errores con M = [10, 20, 40], N = [10, 20, 40]
h = [10, 20, 40]
k = [10, 20, 40]
data = []

for i in range(3):
    M = h[i]
    N = k[i]
    W, X, T = pdesolver.implicitHeat(eq, Ix, It, M, N)
    U = u(X, 1)
    maxError = max(abs(U-W[:,-1]))
    eoc = 'NaN' if i == 0 else np.log(maxError/prevError)/np.log(h[i-1] / M)
    data.append([ maxError, eoc ])
    prevError = maxError

data = [['1/10', '1/10'] + data[0], ['1/20', '1/20'] + data[1], ['1/40', '1/40'] + data[2]]
header = ['h', 'k', 'Error', 'erc']
print(ls.latexTable(header, data, 'cc|rr'))

#Tabla de errores con M = 10, N = [25, 50, 100, 200]
M = 10
k = [25, 50, 100, 200]
data = []

for i in range(4):
    N = k[i]
    W, X, T = pdesolver.implicitHeat(eq, Ix, It, M, N)
    U = u(X, 1)
    maxError = max(abs(U-W[:,-1]))
    eoc = 'NaN' if i == 0 else np.log(maxError/prevError)/np.log(k[i-1] / N)
    data.append([ maxError, eoc ])
    prevError = maxError

data = [['1/25'] + data[0], ['1/50'] + data[1], ['1/100'] + data[2], ['1/200'] + data[3]]
header = ['k', 'Error', 'erc']
print(ls.latexTable(header, data, 'c|rr'))