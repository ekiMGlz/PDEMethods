import numpy as np
import scipy.linalg as linear

def latexMatrix(M, matrixName, eq=True, complx=False, form='%0.0f', cmplxForm='%+0.0fi'):
    s=''
    if eq:
        s ='\\[ \n' + matrixName + ' = \n'
    s += '\\begin{pmatrix} \n'
    [rows, cols] = M.shape
    if complx:
        f = form+cmplxForm
        for i in range(rows):
            for j in range(cols):
                s += (f % (M[i,j].real, M[i,j].imag)) + ' '
                if not j+1 == cols:
                    s += '& '           
            s += '\\\\ \n'
    else:
        for i in range(rows):
            for j in range(cols):
                s += (form % (M[i,j].real)) + ' '
                if not (j+1 == cols):
                    s += '& '
            s += '\\\\ \n'
    s += '\\end{pmatrix}'
    if eq:
        s += '\n \\]'
    return s

def latexVector(v, vecName, eq=True, complx=False, form='%0.0f', cmplxForm='%+0.0fi'):
    s=''
    if eq:
        s ='\\[ \n \\vec{' + vecName + '} = \n'
    s += '\\begin{pmatrix} \n'
    if complx:
        f = form+cmplxForm
        for x in np.nditer(v):
            s += (f % (x.real,x.imag) + ' \\\\ \n')
    else:
        for x in np.nditer(v):
            s += (form % (x.real) + ' \\\\ \n')
    s += '\\end{pmatrix}'
    if eq:
        s += '\n \\]'
    return s

def latexList(l, listName, eq=True, complx=False, form='%0.0f', cmplxForm='%+0.0fi'):
    s=''
    if eq:
        s ='\\[ \n ' + listName + ' = \n'
    s += '\\{ \n'
    if complx:
        f = form+cmplxForm
        for x in l:
            s += (f % (x.real,x.imag) + ', ')
    else:
        for x in l:
            s += (form % (x.real) + ', ')
    s = s[:-2]
    s += '\\}'
    if eq:
        s += '\n \\]'
    return s
    
def latexTable(header, data, colformat):
    s=''
    s += '\\begin{tabular}{'+colformat+'} \n \\hline \n'
    for entry in header:
        s += str(entry) + ' '
        s += '& '
    s = s[:-2] + '\\\\ \hline \n'
    
    for row in data:
        for entry in row:
            s += str(entry) + ' '
            s += '& '
        s=s[:-2]
        s += '\\\\ \n'
    s += '\\hline \n \\end{tabular}'
    
    return s