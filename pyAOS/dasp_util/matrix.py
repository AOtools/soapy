""" Module matrix.py : simplification to the call of BLAS functions
for matrix/vector multiplication and matrix/matrix functions

For performance issues, it is better that the matrices are declared with
the \"fortran\" keyword with the array, empty or zeros commands

Calls BLAS level 2 and 3 functions stored in the scipy.lib.blas.fblas modules
"""

import scipy.lib.blas.fblas as fblas
import numpy,scipy.linalg
##General dictionary storing prefix for BLAS
##functions names as a function of typecode
prefDict={'f':'s','d':'d','F':'c','D':'z'}


def generalMatrixVectorMultiply(mat,vect,alpha=1.,trans=0):
    """
    performs general matrix vector multiplication alpha*mat*vect
    mat : general matrix (real or complex)
    vect : vector
    alpha : multiply coefficient (default 1)
    trans : if 1, computes the product alpha*transpose(mat)*vect
    """
    blasFuncName='gemv'
    cmd='fblas.'+prefDict[mat.dtype.char]+blasFuncName+'(alpha,mat,vect,trans=trans)'
    y=eval(cmd)
    return y

def symmetricMatrixVectorMultiply(mat,vect,alpha=1.,lower=0):
    """
    performs symmetric matrix vector multiplication alpha*mat*vect
    mat : symmetric matrix (real)
    vect : vector
    alpha : multiply coefficient (default 1)
    lower : if 0, only the upper part of the matrix is referenced
    """
    d=vect.dtype.char#if this raises an error, convert vect to numpy...
    blasFuncName='symv'
    cmd='fblas.'+prefDict[mat.dtype.char]+blasFuncName+'(alpha,mat,vect,lower=lower)'
    y=eval(cmd)
    return y

def hermitianMatrixVectorMultiply(mat,vect,alpha=1.,lower=0):
    """
    performs hermitian matrix vector multiplication alpha*mat*vect
    mat : hermitian matrix (complex)
    vect : vector
    alpha : multiply coefficient (default 1+0j)
    lower : if 0, only the upper part of the matrix is referenced
    """
    blasFuncName='hemv'
    cmd='fblas.'+prefDict[mat.dtype.char]+blasFuncName+'(alpha,mat,vect,lower=lower)'
    y=eval(cmd)
    return y

def triangularMatrixVectorMultiply(mat,vect,lower=0,trans=0,unitdiag=0):
    """
    performs general matrix vector multiplication alpha*mat*vect
    mat : general matrix (real or complex)
    vect : vector
    lower : if 0, only the upper part of the matrix is referenced
    trans : if 1, computes the product alpha*transpose(mat)*vect
    unitdiag : if 1, assumes mat is unit triangular
    """
    blasFuncName='trmv'
    cmd='fblas.'+prefDict[mat.dtype.char]+blasFuncName+'(alpha,mat,vect,lower=lower,trans=trans,unitdiag=unitdiag)'
    y=eval(cmd)
    return y


def generalMatrixMatrixMultiply(mat1,mat2,alpha=1.,trans_a=0,trans_b=0):
    """
    performs general matrix matrix multiplication alpha*mat1*mat2
    mat1 and mat2 : general matrices (real or complex)
    vect : vector
    trans_a : if 1, computes the product alpha*transpose(mat1)*mat2
    trans_b : if 1, computes the product alpha*mat1*transpose(mat2)
    """
    blasFuncName='gemm'
    cmd='fblas.'+prefDict[mat1.dtype.char]+blasFuncName+'(alpha,mat1,mat2,trans_a=trans_a,trans_b=trans_b)'
    y=eval(cmd)
    return y


def dot(a,v):
    """
    dot(a,v) returns matrix-multiplication between a and b.  
    The product-sum is over the last dimension of a and the 
    second-to-last dimension of b.
    Calls BLAS optimized functions
    """
    nbDim=len(v.shape)
    if nbDim==1: ##v is a vector
        f=generalMatrixVectorMultiply
    else: ##v is a matrix
        f=generalMatrixMatrixMultiply

    return f(a,v)


def fastMultiply(a,b,c=None,minsize=128):
    """Uses a O(N^log2(7)) algorithm, rather than O(N^3).  From
    Numerical recipes
    For a 2048x2048 matrix multiply, this is approx 150x faster than numpy.dot!
    The best minsize parameter will probably depend on what machine this is
    run on.  128 seems to work for the Cray.
    """
    if a.shape[0]%2!=0 or a.shape[1]%2!=0 or b.shape[0]%2!=0 or b.shape[0]%2!=0 or a.shape[1]<minsize or b.shape[0]<minsize:
        return numpy.dot(a,b)
    if c==None:
        c=numpy.zeros((a.shape[0],b.shape[1]),max(a.dtype,b.dtype))
    ax=a.shape[1]/2
    ay=a.shape[0]/2
    bx=b.shape[1]/2
    by=b.shape[0]/2
    cy=c.shape[0]/2
    cx=c.shape[1]/2
    Q1=fastMultiply(a[:ay,:ax]+a[ay:,ax:],b[:by,:bx]+b[by:,bx:],minsize=minsize)
    Q2=fastMultiply(a[ay:,:ax]+a[ay:,ax:],b[:by,:bx],minsize=minsize)
    Q3=fastMultiply(a[:ay,:ax],b[:by,bx:]-b[by:,bx:],minsize=minsize)
    Q4=fastMultiply(a[ay:,ax:],b[by:,:bx]-b[:by,:bx],minsize=minsize)
    Q5=fastMultiply(a[:ay,:ax]+a[:ay,ax:],b[by:,bx:],minsize=minsize)
    Q6=fastMultiply(a[ay:,:ax]-a[:ay,:ax],b[:by,:bx]+b[:by,bx:],minsize=minsize)
    Q7=fastMultiply(a[:ay,ax:]-a[ay:,ax:],b[by:,:bx]+b[by:,bx:],minsize=minsize)
    c[:cy,:cx]=Q1+Q4-Q5+Q7
    c[cy:,:cx]=Q2+Q4
    c[:cy,cx:]=Q3+Q5
    c[cy:,cx:]=Q1+Q3-Q2+Q6
    return c

def fastInverse(a,c=None,minsize=128,minsizemult=128,invfn=numpy.linalg.inv):
    """Uses a O(N^log2(7)) algorithm rather than O(N^3).  From Numerical
    recipes, page 102
    Note, there are some cases where this doesn't appear to work - for example a matrix extended by a few diagonal elements... so probably not too safe to use!.
    """
    if a.shape[0]%2!=0 or a.shape[0]<minsize:
        return invfn(a)
    if a.shape[0]!=a.shape[1]:
        raise Exception("Must be square matrix %s"%str(a.shape))
    ax=ay=cy=cx=a.shape[0]/2
    if type(c)==type(None):
        c=numpy.zeros(a.shape,a.dtype)

    R1=fastInverse(a[:ay,:ax],minsize=minsize,minsizemult=minsizemult,invfn=invfn)
    R2=fastMultiply(a[ay:,:ax],R1,minsize=minsizemult)
    R3=fastMultiply(R1,a[:ay,ax:],minsize=minsizemult)
    R4=fastMultiply(a[ay:,:ax],R3,minsize=minsizemult)
    R5=R4-a[ay:,ax:]
    R6=fastInverse(R5,minsize=minsize,minsizemult=minsizemult,invfn=invfn)
    c[:cy,cx:]=fastMultiply(R3,R6,minsize=minsizemult)
    c[cy:,:cx]=fastMultiply(R6,R2,minsize=minsizemult)
    R7=fastMultiply(R3,c[cy:,:cx],minsize=minsizemult)
    c[:cy,:cx]=R1-R7
    c[cy:,cx:]=-R6
    return c

def cholskyInverse(a):
    """perform a cholsky decomposition of a SPD matrix, and then
    get the inverse from this.
    """
    cho=scipy.linalg.cho_factor(a)
    inv=scipy.linalg.cho_solve(cho,numpy.identity(a.shape[0]))
    return inv

def luInverse(a):
    """perform a LU decomposition and then the inverse fro this.
    """
    lu=scipy.linalg.lu_factor(a)
    inv=scipy.linalg.lu_solve(lu,numpy.identity(a.shape[0]))
    return inv
