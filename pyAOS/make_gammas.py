
#from Numeric import *
from numpy import *
import FITS
#from gist import *

# Make gamma matrices (Noll 1976)
def makegammas(nzrad):
    n=[0]
    m=[0]
    tt=[1]
    trig=0

    for p in range(1,nzrad+1):
        for q in range(p+1):
            if(fmod(p-q,2)==0):
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
        #print j+1, n[j], m[j], tt[j]

    gamx=zeros((nzmax,nzmax),"float32")
    gamy=zeros((nzmax,nzmax),"float32")

    # Gamma x
    for i in range(nzmax):
        for j in range(i+1):

            # Rule a:
            if (m[i]==0 or m[j]==0):
                gamx[i,j] = sqrt(2.0)*sqrt(float(n[i]+1)*float(n[j]+1))
            else:
                gamx[i,j] = sqrt(float(n[i]+1)*float(n[j]+1))

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
                gamy[i,j] = sqrt(2.0)*sqrt(float(n[i]+1)*float(n[j]+1))
            else:
                gamy[i,j] = sqrt(float(n[i]+1)*float(n[j]+1))

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



#    FITS.Write(gamx, 'gamma_x.fits')
#    FITS.Write(gamy, 'gamma_y.fits')
    return gamx,gamy 

#makegammas(25)

