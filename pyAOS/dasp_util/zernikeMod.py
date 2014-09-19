## Automatically adapted for numpy Mar 10, 2006 by 

## Automatically adapted for numpy Mar 09, 2006 by

"""
Module zernike
New functions to create Zernike Polynomials (ZP) from F Assemat,

Conversion of accelerated Yorick functions

Zernike Polynomials follow definition given in Noll-1976 JOSA article
"""

import numpy as na
#import util.dot as quick # not necessary, as the line where used is commented out
#import scipy#agbhome
from .matrix import symmetricMatrixVectorMultiply#agbhome
import scipy.linalg as nala#agbhome
from . import tel
from scipy.special import gamma,gammaln
import time
# try:
#     import Numeric
# except:
#     print "Unable to import Numeric - continuing"

def makeThetaGrid(npup,natype=na.float32):
    """
    makeThetaGrid(dpix,natype)
    returns a dpix*dpix Numarray array with a grid of angles (in radians)
    from the center of the screen

    default return type : Float32
    """
    tabx=na.arange(npup)-float(npup/2.)+0.5 ##RWW's convention
    grid=na.arctan2(tabx[:,na.newaxis],tabx[na.newaxis,:,])
    return grid.astype(natype)

def nm(j,sign=0):
    """
    returns the [n,m] list giving the radial order n and azimutal order
    of the zernike polynomial of index j
    if sign is set, will also return a 1 for cos, -1 for sine or 0 when m==0.
    """
    n = int((-1.+na.sqrt(8*(j-1)+1))/2.)
    p = (j-(n*(n+1))/2.)
    k = n%2
    m = int((p+k)/2.)*2 - k
    if sign==0:
        return [n,m]
    else:#determine whether is sine or cos term.
        if m!=0:
            if j%2==0:
                s=1
            else:
                s=-1
            #nn,mm=nm(j-1)
            #if nn==n and mm==m:
            #    s=-1
            #else:
            #    s=1
        else:
            s=0
        return [n,m,s]


def fastZerCoeffs(n,m,natype=na.float64):
    """
    returns the array of the Knm(s) coefficients for the definition of
    Zernike polynomials (from Noll-1976)

    We use the following properties to improve computation speed :
    - K_mn(0) = n! / ((n+m)/2)! / ((n-m)/2)!
    - Knm(s+1) =  -Knm(s) * ((n+m)/2-s)*((n-m)/2-s)/(s+1)/(n-s) 
    """
    #result of the function
    result=na.zeros(n+1,natype)
    
    ## We first compute Knm(0), giving the coefficient of the highest polynomial
    st =  2 ## start index for dividing by ((n-m)/2)!
    coef = 1.00
    #print "fastZerCpeffs",int((n+m)/2.+1.5),n+1
    for i in range(int((n+m)/2.+1.5),n+1):
        ## we compute n! / ((n+m)/2)!
        if (st<=int(((n-m)/2.)+0.5) and (i%st==0)):
            j = i/float(st)
            st+=1
            coef *= j
        else:
            coef *= i

    ## We then divide by ((n-m)/2) ! (has already been partially done)
    #print "factorial from",st,int((n-m)/2.+1.5)
    for i in range(st,int((n-m)/2.+1.5)):
        coef /= float(i)
    
    ##We fill the array of coefficients
    result[n] = na.floor( coef + 0.5)  ## for K_nm(s=0), ie n=m

    ##We use the recurrent relation shown in the function documentation
    for i in range(1,int((n-m)/2.+1.5)):
      coef *= -((n+m)/2.-i+1)*((n-m)/2.-i+1)
      coef /= float(i)
      coef /= float((n-i+1))
      result[n-2*i] = na.floor( coef + 0.5 )

    return result

## def Rnm(n,m,a,r):
##     """
##     Returns the radial part of the Zernike polynomials of the Zernike
##     polynomial with radial order n and azimutal order m

##     n,m : radial and azimutal order
##     a : array of Knm(s) coefficients given by fastZerCoeffs
##     r : radial coordinate in the pupil squared (from tel.makeCircularGrid(dosqrt=0))

##     Use of informations in section 5.3 of Numerical Recipes to accelerate
##     the computataion of polynomials
##     """
##     if n>1 :
##         r2 = r*r

##     p=a[n]
##     for i in range(n-2,m-1,-2):
##         p=p*r2+na.array(a[i])
  
##     if m==0:
##         return p
##     elif m==1:
##         p*=r
##     elif m==2:
##         p*=r2
##     else:
##         p*=(r**m)
  
##     return p

def Rnm(n,m,a,r,dosquare=1):
    """
    Returns the radial part of the Zernike polynomials of the Zernike
    polynomial with radial order n and azimutal order m

    n,m : radial and azimutal order
    a : array of Knm(s) coefficients given by fastZerCoeffs
    r : radial coordinate in the pupil squared (from tel.makeCircularGrid(dosqrt=0))

    Use of informations in section 5.3 of Numerical Recipes to accelerate
    the computataion of polynomials
    """
    if dosquare:
        if n>1 :
            r2 = r*r
    else:#r already squared (on large pupils, sqrting and then squaring again can lead to errors)
        if n>1 :
            r2=r
        r=na.sqrt(r)


    p=a[n]
    for i in range(n-2,m-1,-2):
        p=p*r2+a[i]#I think this is the part that causes numerical precision errors for large pupils...
  
    if m==0:
        return p
    elif m==1:
        p*=r
    elif m==2:
        p*=r2
    else:#This can be a problem in terms of numerical precision?  Probably not...
        p*=(r2**(m/2.))
  
    return p

def makegammas(nzrad,returntype="numpy"):
    """Make gamma matrices (Noll 1976)
    eg makegammas(9) will return a 55x55 (thats (9+1)*(9+2)/2) matrix.
    Code from Tim Butterley, date 20070111.
    nzrad is the radial order.
    """
    natype=na
    n=[0]
    m=[0]
    tt=[1]
    trig=0

    for p in range(1,nzrad+1):
        for q in range(p+1):
            if(na.fmod(p-q,2)==0):
                if(q>0):
                    n.append(p)
                    m.append(q)
                    trig=not(trig)
                    tt.append(trig)
                    n.append(p)
                    m.append(q)
                    trig=not(trig)
                    tt.append(trig)
                else:
                    n.append(p)
                    m.append(q)
                    tt.append(1)
                    trig=not(trig)
    nzmax=len(n)

    #for j in range(nzmax):
    #    print j+1, n[j], m[j], tt[j]
    if returntype=="numpy":
        gamx=na.zeros((nzmax,nzmax),na.float64)
        gamy=na.zeros((nzmax,nzmax),na.float64)
    else:
        raise Exception("Numeric not supported anymore")
        #gamx=Numeric.zeros((nzmax,nzmax),Numeric.Float64)
        #gamy=Numeric.zeros((nzmax,nzmax),Numeric.Float64)
        

    # Gamma x
    for i in range(nzmax):
        for j in range(i+1):
            
            # Rule a:
            if (m[i]==0 or m[j]==0):
                gamx[i,j] = natype.sqrt(2.0)*natype.sqrt(float(n[i]+1)*float(n[j]+1))
            else:
                gamx[i,j] = natype.sqrt(float(n[i]+1)*float(n[j]+1))
            
            # Rule b:
            if m[i]==0:
                if ((j+1) % 2) == 1:
                    gamx[i,j] = 0.0
            elif m[j]==0:
                if ((i+1) % 2) == 1:
                    gamx[i,j] = 0.0                
            else:
                if ( ((i+1) % 2) != ((j+1) % 2) ):
                    gamx[i,j] = 0.0
            
            # Rule c:
            if abs(m[j]-m[i]) != 1:
                gamx[i,j] = 0.0

            # Rule d - all elements positive therefore already true

    # Gamma y
    for i in range(nzmax):
        for j in range(i+1):
            
            # Rule a:
            if (m[i]==0 or m[j]==0):
                gamy[i,j] = natype.sqrt(2.0)*natype.sqrt(float(n[i]+1)*float(n[j]+1))
            else:
                gamy[i,j] = natype.sqrt(float(n[i]+1)*float(n[j]+1))

            # Rule b:
            if m[i]==0:
                if ((j+1) % 2) == 0:
                    gamy[i,j] = 0.0
            elif m[j]==0:
                if ((i+1) % 2) == 0:
                    gamy[i,j] = 0.0                
            else:
                if ( ((i+1) % 2) == ((j+1) % 2) ):
                    gamy[i,j] = 0.0
            
            # Rule c:
            if abs(m[j]-m[i]) != 1:
                gamy[i,j] = 0.0

            # Rule d:
            if m[i]==0:
                pass    # line 1
            elif m[j]==0:
                pass    # line 1
            elif m[j]==(m[i]+1):
                if ((i+1) % 2) == 1:
                    gamy[i,j] *= -1.    # line 2
            elif m[j]==(m[i]-1):
                if ((i+1) % 2) == 0:
                    gamy[i,j] *= -1.    # line 3
            else:
                pass    # line 4
    return gamx,gamy

def lnGamma(x,natype=na.complex64):
    """returns a Complex array with the complex logarithm of the array x
    Normally the gammaln function applies only to positive numbers
    We generalise here the function to the case where x is negative, by adding i*pi

    Input parameters :
        - x : input variable (scalar or array)
        - natype : Numeric Complex data type (Complex32 by default) for scalar values
    
    """
    ##Allocation of the returned array
    ##Check if x is a scalar variable
    if na.isscalar(x):
        res=na.array(x,dtype=na.complex_)
        x2=na.array(x,dtype=natype)
    else: ##x is a Numeric array (x2 is used as a reference to input array)
        res=na.zeros(x.shape,dtype=na.complex_)
        x2=x
            
    ##We compute lngamma for positive values
    idx=na.nonzero(na.where(x2.ravel()>=0,1,0))[0]
    ##print x.dtype.char
    if idx.shape!=(0,):
        na.put(res.ravel(),idx,gammaln(na.take(x2.flat,idx))+0j)
    
    ##We compute lngamma for negative values
    idx=na.nonzero(na.where(x2.flat<=0,1,0))[0]
    if idx.shape!=(0,): ##we use the following property : gamma(x)*gamma(-x)
        z=-1.*na.take(x2.flat,idx) ##we take the opposite of the x
        temp=-na.log(z)-na.log(na.sin(na.pi*z)+0j)-gammaln(z)
        temp+=na.log(na.pi)+1j*na.pi
        na.put(res.ravel(),idx,temp)
    ##we return the res array
    if na.isscalar(x): ##we have a scalar value
        res.astype(natype)
        return res
    else: ## x is a Numeric array.
        if (x.dtype==na.single):
            res=res.astype(na.csingle)
        elif (x.dtype==na.float_):
            res=res.astype(na.complex_)
        else:
            res=res.astype(natype)
        res.shape=x.shape
        return res

def hypPfQ(a,b,z,k=50,natype=na.float32):
    """ Computes the generalised hypergeometric function pFq(a;b;z)
    See page 166 of F Assemat's thesis for the definition of the Taylor expansion

    Input parameters : 
        - P : number of elements of a
        - Q : number of elements of b
        - z : input (SCALAR !!!)
        - k : maximum index for Taylor expansion (50 by default)
        - natype : Numeric Float data type (Float32 by default) of the returned value
    """    
    ##dimensions of a and b
    p=len(a)
    q=len(b)

    ##computation of the product of the gamma functions in front of the sum
    s1=na.sum(lnGamma(b))
    s2=na.sum(lnGamma(a))
    C=na.exp(s1-s2).real

    ##array of k values going from 0 to k
    tabk=na.arange(k+1)

    ##we create the arrays of the a+k (2D array), and compute the sum of their logarithm (k-array)
    taba_k=a[:,na.newaxis]+tabk[na.newaxis,:,];lnA1=na.sum(lnGamma(taba_k),axis=0)

    ##we create the arrays of the b+k (2D array), and compute the sum of their logarithm (k-array)
    tabb_k=b[:,na.newaxis]+tabk[na.newaxis,:,];lnA2=na.sum(lnGamma(tabb_k),axis=0)

    ##we create the arrays of the lnGamma of tabk
    lnA3=tabk*na.log(z)-lnGamma(tabk+1.)

    ##we compute the return value of the function, store it into a Numeric array with the good data type
    res=na.sum(na.exp(lnA1-lnA2+lnA3)).real*C
    res=na.array(res,natype)
    return res

def matCovZernikeL0(jmax,dlo,dro,natype=na.float32,kmax=50,diagOnly=0):
    """returns covariance matrix of Zernike Polynomials 1 to jmax
    for Von-Karman turbulence (finite outer scale)

    jmax :  maximum index of ZP to take into account
    dlo  : (D/L0) ratio
    dro  : (D/ro) ratio
    natype : Numeric data type of the output array (by default Float32)
    kmax : maximum number to take into account in the computation of hypergeometric function

    ATTENTION !!!!!!!!!!!!!!!!!!!!
    the output array has jmax*jmax elements
    as piston has no infinite variance in the case of Von-Karman turbulence
    """

    ## We compute the square of the pi*D/l0 ratio, as this quantity
    ## is used in the computation of the matrix
    piDfo=na.pi*dlo
    piDfoS=piDfo*piDfo

    ## Factor at the beginning of expression
    fTurb=2*gamma(11./6.);
    fTurb/=(na.pi)**(3./2.);
    fTurb*=((24./5)*gamma(6./5))**(5./6);
    fTurb*=(dro)**(5./3.);

    ## radial order of jmax
    nmax=nm(jmax)[0]

    ## we first compute a (nmax+1)x(nmax+1) array,
    ## storing the numbers only function of n1 and n2
    matNM=na.zeros((nmax+1,nmax+1),dtype=natype)
    for n1 in range(0,nmax+1):
        ##print "n1=",n1
        for n2 in range(0,n1+1):
            ##print "n2=",n2
            ##Equation 2.42, page 166 of F Assemat's thesis
            a=na.array([(3.+n1+n2)/2,2.+(n1+n2)/2.,1.+(n1+n2)/2.,5./6-(n1+n2)/2.])
            a11=gamma(a);
            a=na.array([(3.+n1+n2),2.+n1,2.+n2]);
            a12=gamma(a);
            a1=na.product(a11)/na.product(a12);
            a=na.array([(3.+n1+n2)/2,2.+(n1+n2)/2.,1.+(n1+n2)/2.]);
            b=na.array([3.+n1+n2,2.+n1,2.+n2,(3.*n1+3.*n2+1)/6]);
            a2=hypPfQ(a,b,piDfoS,k=kmax);
            a2=a2*(piDfo)**((3.*(n1+n2)-5)/3);
            s=a1*a2;

            b11=gamma(na.array([(3.*n1+3.*n2-5)/6,7./3,17./6,11./6]));
            b12=gamma(na.array([(3.*n1+3.*n2+23)/6.,(3.*n1-3.*n2+17)/6.,(3.*n2-3.*n1+17)/6]));
            b1=na.product(b11)/na.product(b12);
            a=na.array([7./3,17./6,11./6]);
            b=na.array([(3.*n1+3.*n2+23.)/6,(3.*n1-3.*n2+17)/6,(3.*n2-3.*n1+17.)/6,(11.-3.*n1-3.*n2)/6]);
            b2=hypPfQ(a,b,piDfoS,k=kmax);
            s+=b1*b2;
            s*=na.sqrt((n1+1)*(n2+1));

            ##we fill the matNM matrix
            matNM[n1,n2]=s*fTurb
            matNM[n2,n1]=s*fTurb

    ##display of the matNM matrix
    ##print "Display of matNM"
    ##print matNM
    
    ##Allocation of the output array
    if diagOnly:
        result=na.zeros((jmax,),dtype=natype)
    else:
        result=na.zeros((jmax,jmax),dtype=natype)

    ##As the matrix is symmetric (because it's a covariance matrix),
    ##we just need to compute upper triangular values
    for j1 in range(1,jmax+1): ##we go among the 1st dimension
        n1,m1=nm(j1) ##radial and azimutal orders of 1st ZP
        ##print "j1",j1
        if diagOnly:
            K=(-1.)**(n1-m1)
            ## we take the value in matNM (stores values which are function of n1 and n2)
            fact=matNM[n1,n1]
            ##we fill the matrix
            result[j1-1]=K*fact
        else:
            for j2 in range(1,j1+1): ##we go among the second dimension
                delta=0
                n2,m2=nm(j2) ##radial and azimutal orders of 2d ZP
                ##print "j2",j2
                if (m1==m2): ##the polynomials have the same azimutal orders
                    if (m1==0) and (m2==0):
                        ##print "m=0"
                        delta=1
                    elif ((j1%2==0)and(j2%2==0))or((j1%2==1)and(j2%2==1)):##j1 and j2 have the same parity
                        ##print "j1 and j2 have same parity"
                        delta=1
                    if delta:
                        ##we use formulae from Noll-1976 JOSA paper
                        K=(-1.)**((n1+n2-2.*m1)/2.)
                        ## we take the value in matNM (stores values which are function of n1 and n2)
                        fact=matNM[n1,n2]
                        ##we fill the matrix
                        result[j1-1,j2-1]=result[j2-1,j1-1]=K*fact

    return result

def calcZern(nz,coords,output=None):
    """compute zernike array for zernike number nz, doing the computation at locations specified by coords array (shape ncoord,2) corresponding to a list of x,y coords.
    piston corresponds to nz==1
    """
    if type(output)==type(None):
        output=na.zeros((coords.shape[0],),na.float32)
    if nz<=1:
        output[:,]=1
        return output
    rad,az,trig=nm(nz,1)#get radial and azimuthal order.
    n=rad
    m=az
    #if trig==1, is a cos term, else if ==-1 is a sign term, else is not a trig term.
    ##loop to fill the cube of zernike polynomials
    rGrid=na.sqrt(coords[:,0]**2+coords[:,1]**2)
    thetaGrid=na.arctan2(coords[:,1],coords[:,0])
    ##computation of the starting azimutal order
    ##we compute the radial part of the polynomial
    ##print "m=%d" % m
    a=fastZerCoeffs(n,m)
    Z=Rnm(n,m,a,rGrid)*na.array(na.sqrt(n+1))
    ##we make the other computations
    if m==0: ##azimutal order = 0 => no trigonometric part
        ##we add one ZP when m=0
        #print "j=%d" % j
        #na.put(self.zern[jmax.index(j-1),:,:,].ravel(),idxPup,Z)
        #j+=1
        pass
    else: ##azimutal order >0 : there is a trigonometric part
        Z*=na.array(na.sqrt(2.))
        ##we add the two zernike polynomials per azimutal order
        if trig==-1: ## j is odd : multiply by sin(m*theta)
            #print "j=%d" % j
            Z*=na.sin(m*thetaGrid)
            #na.put(self.zern[jmax.index(j-1),:,:,].ravel(),idxPup,Z*na.sin(m*thetaGridFlat))
        else: ## j is even : multiply by cos(m*theta)
            #print "j=%d" % j
            Z*=na.cos(m*thetaGrid)
            #na.put(self.zern[jmax.index(j-1),:,:,].ravel(),idxPup,Z*na.cos(m*thetaGridFlat))
            
    output[:,]=Z.astype(output.typecode())
    return output


def normalise(modes,scaleto=1.):
    """Normalise zernikes (or anything) to a value (default 1) such that integration of the function squared equals this value.
    """
    for i in range(modes.shape[0]):
        #now give orthonormal scaling...
        modes[i]/=na.sqrt(na.sum(modes[i]*modes[i])/scaleto)


### Zernike Polynomials class #############################################
class Zernike:
    """
    Class Zernike : new class to define Zernike Polynomials
    Assumes full circular apertures

    Fields :
    - jmax : index of the maximum zernike polynomial stored in the zern array, or a list of indexes to be computed.
    - npup : number of pixels of the zern array
    - zern : Numeric (Float32 or 64 or 128) array storing the Zernike Polynomials maps
             shape=(jmax,npup,npup) or (len(jmax),npup,npup)
    - invGeoCovMat : inverse of the geometric covariance matrix (used to give the
                     real expansion of the input phase)
    - pupfn : Ubyte array storing the circular binary array defining pupil function
    - puparea : area in PIXELS of the pupil function
    - idxPup : 1D array with the index of the pixels belonging to the pupil
    - natype : Numeric Data type of the zern array (Float32/64)
    - zern2D : Numeric Float32/64 array storing the zern cube
    as a 2D array (each 2D ZP is converted as a 1D array), with only the values belonging to the pupil
    Note, this module is approx 10x slower than cmod.zernike for creation of a single zernike on a large pupil when using float64.
    This module is 3.5x slower with float32 than with float64.
    Note, there are floating point precision issues even with float128.  For eg pupil=600, jmax=2016, this appears fuzzy if using float64.  Okay for float128, but then numpy.sum(zern[]*zern[]) should equal pup.sum, but doesn't.  Just be warned that high order zernikes may not be correct!
    """
    def __init__(self,pupil,jmax,natype=na.float64,removePiston=0,computeInv=1):
        """ Constructor for the Zernike class

        Parameters :
        - pupil : Pupil object storing pupil geometry
        - jmax : maximum index of the Zernike polynomial to store into the zern array, either a int, or a list of ints to be generated.
        - natype : output array data type (by default Float32):
        """
        if type(pupil)==na.ndarray:# or type(pupil)==Numeric.ArrayType:#ArrayType:
            self.pupfn=pupil#na.array(pupil.copy())
        elif type(pupil)==type(1):
            self.pupfn=tel.Pupil(pupil,pupil/2.,0).fn
        else:
            self.pupfn=pupil.fn#na.array(pupil.fn.copy()) ##copy of the pupil function
        self.npup=self.pupfn.shape[0] ##number of pixels for one zernike polynomial
        if natype==na.float32 and self.npup>100:
            print("WARNING: zernikeMod - using float32 for large pupils can result in error.")

        self.natype=natype ##type of output array
        
        if type(jmax)==type(1):
            jmax=range(jmax)
        #jmax.sort()
        #print "Creation of Zernike Polynomials"
        self.createZernikeCube(jmax)
        self.zern2D=self.zern.view()
        self.zern2D.shape=(len(jmax),self.npup*self.npup)
        
        if computeInv:
            #print "Computation of the inverse of the geometric covariance mx"
            t1=time.time()
            self.createInvGeometricCovarianceMatrix(removePiston)
            dt=time.time()-t1
            #print "Final Zernike takes",dt

                
    def createZernikeCube(self,jmax):
        """Fills the zern array (cube of Zernike Polynomials)
        jmax : maximum index of the Zernike Polynomial to store into zern
        """
        ## grid of distance to center normalised to 1,
        ## because ZP are defined on unit radius pupils
        rGrid=tel.makeCircularGrid(self.npup,natype=self.natype,dosqrt=0)/self.npup**2*4
        #rGrid=rGrid.astype(self.natype) ##to keep the same datatype

        ##1D-index of pixels belonging to the pupil
        self.idxPup=idxPup=na.nonzero(self.pupfn.ravel())
        self.puparea=len(idxPup)

        ## grid of angles
        thetaGrid=makeThetaGrid(self.npup,natype=self.natype) ##grid of angles

        ##modification of jmax
        self.jmax=jmax
        ##we extract only the pixels corresponding to the pupil, to improve computation time
        try:
            rGridFlat=na.take(rGrid.ravel(),idxPup)
        except:
            rGridFlat=na.take(rGrid.flat,idxPup)
        thetaGridFlat=na.take(thetaGrid.ravel(),idxPup)

        ##we allocate the cube of Zernike Polynomials
        retShape=(len(jmax),self.npup,self.npup)
        self.zern=na.zeros(retShape,self.natype)

        ##we look for the radial order corresponding to jmax
        nmax=nm(max(jmax)+1)[0]
        ##print "nmax=%d" % nmax

        ##we put the piston
        if 0 in jmax:#piston required.
            self.zern[jmax.index(0),:,:,]=self.pupfn.copy().astype(self.natype)
        
        ##loop to fill the cube of zernike polynomials
        j=2

        ##we go through the radial orders 1 to nmax-1
        for n in range(1,nmax):
            ##computation of the starting azimutal order
            ## print "n=%d" % n
            if n%2:##n is odd : m starts at 1
                mstart=1
            else: ##n is even : mstart at 0
                mstart=0
            m=mstart      
            while (m<=n):
                ##we compute the radial part of the polynomial
                #Used to call Rnm here...
                ##print "m=%d" % m
                ##we make the other computations
                if m==0: ##azimutal order = 0 => no trigonometric part
                    ##we add one ZP when m=0
                    #print "j=%d" % j
                    if j-1 in jmax:
                        a=fastZerCoeffs(n,m,self.natype)
                        Z=Rnm(n,m,a,rGridFlat,dosquare=0)*na.array(na.sqrt(n+1))
                        #print j,n,m,a
                        na.put(self.zern[jmax.index(j-1),:,:,].ravel(),idxPup,Z)
                    j+=1
                else: ##azimutal order >0 : there is a trigonometric part
                    if ((j-1) in jmax) or (j in jmax):#generate the data...
                        a=fastZerCoeffs(n,m,self.natype)
                        Z=Rnm(n,m,a,rGridFlat,dosquare=0)*na.array(na.sqrt(n+1))
                        Z*=na.array(na.sqrt(2.))
                    ##we add the two zernike polynomials per azimutal order
                    for cnt in range(2):               
                        if j%2: ## j is odd : multiply by sin(m*theta)
                            #print "j=%d" % j
                            if j-1 in jmax:
                                #print j,n,m,a
                                na.put(self.zern[jmax.index(j-1),:,:,].ravel(),idxPup,Z*na.sin(m*thetaGridFlat))
                        else: ## j is even : multiply by cos(m*theta)
                            #print "j=%d" % j
                            if j-1 in jmax:
                                #print j,n,m,a
                                na.put(self.zern[jmax.index(j-1),:,:,].ravel(),idxPup,Z*na.cos(m*thetaGridFlat))
                        j+=1
                
                ##we increase the azimuthal order
                m+=2

        ##we do the last radial order
        n=nmax
        ##computation of the starting azimutal order
        ## print "n=%d" % n
        if n%2:##n is odd : m starts at 1
            mstart=1
        else: ##n is even : mstart at 0
            mstart=0
        m=mstart
        while (m<=n): ##we go through the azimutal orders
            ##we compute the radial part of the polynomial
            ##print "m=%d" % m
            #a=fastZerCoeffs(n,m)
            #Z=Rnm(n,m,a,rGridFlat,dosquare=0)*na.array(na.sqrt(n+1))
            ##we make the other computations
            if m==0: ##azimutal order = 0 => no trigonometric part
                if j>max(jmax)+1: ##we leave the while loop if jmax is reached
                    break
                else:
                    ##we add one ZP when m=0
                    #print "j=%d" % j
                    if j-1 in jmax:
                        a=fastZerCoeffs(n,m,self.natype)
                        Z=Rnm(n,m,a,rGridFlat,dosquare=0)*na.array(na.sqrt(n+1))
                        #print "zernikeMod put2",j,n,m,a
                        na.put(self.zern[jmax.index(j-1),:,:,].ravel(),idxPup,Z)
                    j+=1
            else: ##azimutal order >0 : there is a trigonometric part
                if ((j-1) in jmax) or j in jmax:
                    if j<=max(jmax)+1:
                        a=fastZerCoeffs(n,m,self.natype)
                        Z=Rnm(n,m,a,rGridFlat,dosquare=0)*na.array(na.sqrt(n+1))
                        Z*=na.array(na.sqrt(2.))
                ##we add the two zernike polynomials per azimutal order
                for cnt in range(2):
                    if j%2: ## j is odd : multiply by sin(m*theta)
                        if j>max(jmax)+1: ##we leave the current for loop if jmax is reached
                            break
                        else:
                            #print "Adding term in sinus"
                            #print "j=%d" % j
                            if j-1 in jmax:
                                #print j,n,m,a
                                na.put(self.zern[jmax.index(j-1),:,:,].ravel(),idxPup,Z*na.sin(m*thetaGridFlat))
                            j+=1
                    else: ## j is even : multiply by cos(m*theta)
                        if j>max(jmax)+1: ##we leave the while loop if jmax is reached
                            break
                        else:
                            #print "Adding term in cosinus"
                            #print "j=%d" % j
                            if j-1 in jmax:
                                #print j,n,m,a
                                na.put(self.zern[jmax.index(j-1),:,:,].ravel(),idxPup,Z*na.cos(m*thetaGridFlat))
                            j+=1
                               
                if (j>max(jmax)+1): ##we leave the while loop if jmax is reached
                    break


            ##we increment the azimutal order of 2
            m+=2
    
    def createInvGeometricCovarianceMatrix(self,removePiston=0):
        """creates the geometric covariance matrix,
        computes its inverse and store it into the invGeoCovMat array

        This matrix is required to compute the real expansion of
        an input phase on the ZP stored in the zern array,
        because they don't define an orthonormal basis (see p170 of Francois Assemat's thesis)  Actually pg 163 I think.

        Fills the invGeoCovMat array
        If removePiston==1, will remove the piston term...
        """
        if removePiston!=0:
            removePiston=1
        ##copy of the jmax and of the area in pixels of the pupil
        jmax=self.jmax[:]
        popped=0
        if removePiston and (0 in self.jmax):
            jmax.pop(0)
            popped=1
        #jmax=self.jmax-removePiston
        area=self.puparea

        ##Allocation of the geometric covariance array
        geoCovMat=na.zeros((len(jmax),len(jmax)),self.natype)

        ##we just use the pixels belonging to the pupil
        idxPup=self.idxPup

        #loop over the zernike polynomials to fill the geometric covariance matrix
        for j in range(len(jmax)):
            ##we take the values of the current ZP into the pupil
            zj=self.zern2D[j+popped]#+removePiston]
            ##we take the values of the others ZP into the pupil
            zj2=self.zern2D[j+popped:,]#+removePiston:,]
            ##we compute the numeric scalar product
            col=(zj2*zj).sum(axis=1)/float(area)#zj2*zj[na.newaxis:,]
            #col=na.sum(col,axis=1)
            #col=na.sum(col,axis=1)/area
            col=col.astype(self.natype)
            
            ##we fill the matrix
            geoCovMat[j,j:,]=col
            geoCovMat[j:,j]=col

        ##we store it as a field of the class (for debugging purpose)
        self.geoCovMat=geoCovMat
        #print "inverting geoCovMat"
        ##we compute the inverse of the covariance matrix and store it into the invGeoCovMat field
        ##As the covariance matrix is symetric, we use Cholesky decomposition to inverse it
        try:
            ch=nala.cho_factor(geoCovMat)
        except:
            ch=None
            print("cholsky decomposition of geometric covariance matrix failed.")
            print( "Possibly, try using fewer zernikes for your pupil size.  I hate solicitors")
        if type(ch)!=type(None):
            self.invGeoCovMat=nala.cho_solve(ch,na.identity(len(jmax),dtype=self.natype))
        else:
            print ("trying scipy.linalg.inv",geoCovMat.shape)
            self.invGeoCovMat=nala.pinv(geoCovMat)#agbhome
    def giveZernikeExpansion(self,phi):
        """Gives the coefficients a_i of the phase screen phi projected over the Zernike polynomials 1 to self.jmax
        First computes the numeric scalar product b_i=1/puparea sum(phi*self.zern[j,:,:,])
        Then computes the vector a=invGeoCovMat*b, thus taking into account the projection matrix
        """
        ##pupil's area in pixels
        area=self.puparea
        
        ##we compute the numeric scalar product
        col=(self.zern2D*phi.flat).sum(axis=1)/float(area)#self.zern*phi[na.newaxis:,];col=na.sum(col,axis=1);col=na.sum(col,axis=1)/area
        col=col.astype(self.natype)
        
        ##we do the product between the inverse of the geometric covariance function
        ##and the vect array
        ##as the geometric covariance matrix is symetric, its inverse is symetric
        ##we therefore use the function in the matrix package
        result=symmetricMatrixVectorMultiply(self.invGeoCovMat,col)
        #result=quick.dot(self.invGeoCovMat,col)#agbhome
        result.astype(self.natype)

        ##we return the result
        return result
        
    def createPhase(self,a):
        """Creates a phase from the array of coefficients a, ie
        output=sum_{i=0}^(jmax-1) a[i]*self.zern[i,:,:,]
        """

        ##we do a test of the length of a
        la=len(a)

        if (la<len(self.jmax)): ##we have less elements than Zernike polynomials
            anew=na.zeros(len(self.jmax),self.natype)
            anew[:la]=a
        elif (la>len(self.jmax)):
            anew=a[:len(self.jmax)]
        else:
            anew=a

        # output array
        res=na.zeros((self.npup,self.npup),dtype=self.natype)
        # we add the ZPs ponderated by their coefficients: most efficient way of doing it
        for p in self.jmax:#range(self.jmax):
            res+=self.zern[p]*anew[p]
        
        ##computation of the result
        #res=na.sum(anew[:,na.newaxis,na.newaxis]*self.zern)
        return res
    

