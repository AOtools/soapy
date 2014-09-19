#! /usr/bin/env python
'''
Module containing code to analyse WFS slopes.
Accepts slopes - either from darc using "subscribe" class, or from file -
and performs some basic analysis (using "wfs" class) functions such as
calculating r0 and temporal and spatial power spectrums
'''
import FITS
import numpy

#import circle #Andrews C code to generate circles and zernikes
import circlePy as circle    #alternative circle module written in Python. Slower but more reliable

import circle #Andrews code to generate circles and zernikes
###import circlePy as circle    #alternative circle module written in Python. Slower but more reliable
import zerns #Import module from DASP to make zernikes
import sys

import pylab
import make_gammas
import progressbar
try:
    import cmod.sor
    import util.sor
    import util.tel

except:
    print("Unable to import cmod and util DASP modules.\
          SORRecon function will be unavailable.")

class wfs:
    '''
    Class which actually analyses Wavefront Sensor, giving PowerSpectrum
    on a block of data or in real-time.
    use subscribe class with this to get real-time data
    '''

    def __init__(self,sim=None,dir=None):

        if dir !=None:
            sys.path.append(dir)
            import conf
            confDict = conf.parameters

            self.subaps = confDict[subaps]
            self.WL = confDict[waveLengths]
            #self.pxlsPerSubap


        if sim != None:
            self.subaps=sim.subaps[wfs]
            self.WL=sim.waveLengths[wfs]

            self.subapDiam= sim.telDiam/self.subaps
            self.noSubaps=sim.wfss[wfs].activeSubaps
            #Find coords of subaps in use
            self.mask=circle.circleGen(self.subaps/2.,self.subaps)
            self.maskCoords=numpy.where(self.mask==1)
            self.pxlsPerSubap = sim.pxlsPerSubap[wfs]
            self.FR = 1./sim.loopTime
            self.subapFOV = sim.subapFOV[wfs]

            slopes = sim.slopes
            self.workingSlopes = numpy.zeros( (sim.slopes.shape[0], sim.slopes.shape[1]/2.,2) )
            slopes.shape = sim.slopes.shape[0], 2, sim.slopes.shape[1]/2.

            for i in xrange(sim.slopes.shape[0]):
                self.workingSlopes[i] = slopes[i].T
            self.workingSlopes.shape = sim.slopes.shape[0],sim.slopes.shape[-1]*2

            self.subapThreshold = sim.subapThreshold

    def resetSubaps(self):
        '''
        Must be in form (frames, 2*noSubaps), where subaps can be
        (xsubaps,ysubaps,2)
        '''
        self.workingSubaps=self.rawSubaps

    def abStats(self,frames=500):
        slopes=self.sub.getData(stream="bobrtcCentBuf",frames=frames)
        staticSlopes=slopes.mean(0)
        self.workingSubaps-=staticSlopes


    def getSlopes(self,frames):
        self.rawSlopes=self.sub.getData(stream="bobrtcCentBuf",frames=frames)
        self.workingSlopes=self.rawSlopes.copy()

    def loadSlopes(self,fname):
        self.rawSlopes=FITS.Read(fname)[1]
        self.workingSlopes=self.rawSlopes.copy()

    def maskSlopes(self):

        slopes=self.workingSlopes.copy()
        slopes.shape=slopes.shape[0],self.subaps,2
        maskedSlopes=numpy.empty((slopes.shape[0],self.maskCoords[0].shape[0],2))
        for i in xrange(slopes.shape[0]):
            for dim in xrange(2):
                slopes[i,:,dim]=slopes[i,:,dim][self.maskCoords]
        self.workingSlopes=slopes[coords]

    def removeTT(self):
        slopes=self.workingSlopes.copy()
        slopes.shape=slopes.shape[0],self.noSubaps,2
        for i in range(self.noSubaps):
            slopes[:,i,:]-=slopes.mean(1)
        slopes.shape=self.workingSlopes.shape
        self.workingSlopes=slopes

    def TipTilt(self,slopes):
        '''
        Returns the average Tip and Tilt Values for a given set of subapertures
        '''
        slopes.shape=self.noSubaps,2
        tipTilt=slopes.mean(0)
        return tipTilt

    def spatialPowerSpectrum(self):
        slopes=self.slopes2d()

        PowerSpec=numpy.empty(slopes.shape,dtype="complex64")

        for frame in xrange(slopes.shape[0]):
            for dim in xrange(2):
                PowerSpec[frame,:,:,dim]=numpy.fft.fftshift(numpy.fft.fft2(slopes[frame,:,:,dim]))

        del(slopes)
        self.PowerSpec=PowerSpec
        self.meanPowerSpec=abs(PowerSpec.mean(0))
        PowerSpec1d=self.radialAvg(self.meanPowerSpec)

        return PowerSpec1d

    def plotSPS(self):
        SPS=self.spatialPowerSpectrum()
        X=numpy.arange(SPS.shape[0],dtype="float32")

        fig=pylab.figure()
        ax=fig.add_subplot(111)
        ax.plot(numpy.log10(X),numpy.log10(SPS[:,0]),label="X")
        ax.plot(numpy.log10(X),numpylog10(SPS[:,1]),label="Y")

        ax.legend()

        pylab.xlabel("Spatial Frequency")
        pylab.ylabel("Power")


    def plotTPS(self,FR=135):
        TPS=self.temporalPowerSpectrum()
        X=numpy.arange(TPS.shape[0],dtype="float32")
        f=X/FR

        fig=pylab.figure()
        ax=fig.add_subplot(111)
        ax.plot(numpy.log10(f),numpy.log10(TPS[:,0]),label="X")
        ax.plot(numpy.log10(f),numpy.log10(TPS[:,1]),label="Y")

        ax.legend()

        pylab.xlabel("Frequency (log10(Hz))")
        pylab.ylabel("Power")



    def temporalPowerSpectrum(self):
        '''
        Acts on block of working slopes and return temporal power spectra for
        both X and Y dimensions
        '''
        slopes=self.workingSlopes.copy()
        slopes.shape=slopes.shape[0],self.noSubaps,2
        powerSpectra=numpy.fft.fftshift(numpy.fft.fft(slopes,axis=0),axes=0)
        meanPS=powerSpectra.mean(1)
        return abs(meanPS[1+meanPS.shape[0]/2:])

    def radialAvg(self,data):

        avg=numpy.empty((int(data.shape[0]/2),data.shape[-1]))
        for dim in xrange(data.shape[-1]):
            for radii in xrange(int(data.shape[0]/2)):
                ring=self.ring(radii,data.shape[0])
                avg[radii,dim]=(data[:,:,dim]*ring).sum()/ring.sum()
        return avg





    def PSGradient(self,dim="x"):
        '''
        Calculates the gradiant of sections of the calculated log/log temporal power spectrum
        '''
        if dim == "x":
            PS =  self.plotTPS()[:,0]
        if dim == "y":
            PS = self.temporalPowerSpectrum()
        X = numpy.arange(PS[:,0])

        v = X/self.FR #convert to frequency


        fig = pylab.figure()
        ax = pylab.add_subplot()

    def ring(self,radius,size):

        #Make 2 circles, 1 bigger then the other....
        circA=circle.circleGen(radius,size)
        circB=circle.circleGen(radius+1,size)
        #"invert"(ones outside the circle, noughts in) one, and sum, then "invert" the sum!
        #A useful ring...hooray!
        ring=abs(circA+abs(circB-1)-1)

        return ring

    def slopes2d(self):

        slopes=self.workingSlopes.copy()
        twoDSlopes=numpy.zeros((slopes.shape[0],self.subaps,self.subaps,2))

        P = progressbar.ProgressBar()
        P.start()
        for i in xrange(slopes.shape[0]):
            n=0
            for x in xrange(self.subaps):
                for y in xrange(self.subaps):
                    for dim in xrange(2):
                        if self.mask[x,y]==1:
                            twoDSlopes[i,x,y,dim]=slopes[i,n]
                            n+=1
            P.update(100.*i/slopes.shape[0])

        P.finish()

        del(slopes)
        self.twoDSlopes=twoDSlopes
        return twoDSlopes


    def r0(self,error=False):
        '''
        Calculates r0 from an data set of slopes,
        Also returns error on measurement if "error" True
        '''

        slopes=self.workingSlopes.copy()

        arcSecPerPxl = self.subapFOV/self.pxlsPerSubap

        radPerPxl=(arcSecPerPxl/3600.)*numpy.pi/180.

        slopesAngle=slopes*radPerPxl
        slopeVar=slopesAngle.var(0)
        del(slopes)
        r0=(slopeVar**(-3/5.)) * (self.WL**(6./5)) * (self.subapDiam**(-1./5)) * (0.162**(3./5))
        self.r0Array=r0
        meanr0=r0.mean()

        if error:
            err = r0.std()/numpy.sqrt(r0.size)
            return meanr0,err

        else:
            return meanr0

    def r02d(self):
        '''
        Generates a 2d array of r0 values for each subaperture
        '''


        self.r0()
        r0Array=self.r0Array
        n=0
        twoDr0=numpy.zeros((self.subaps,self.subaps,2))
        for x in xrange(self.subaps):
            for y in xrange(self.subaps):
                for dim in xrange(2):
                    if self.mask[x,y]==1:
                        twoDr0[x,y,dim]=self.r0Array[n]
                        n+=1

        self.twoDr0=twoDr0
        return twoDr0


    def reconSOR(self):
        '''
        Reconstruct loaded slopes using Successive Over-relaxation method.
        SOR module first written for DASP simulation
        '''

        print("Re-formatting Slopes....")
        slopes2d = self.slopes2d()

        print("preparing SOR model...")
        #SOR parameters
        conv = 0.01
        maxiters = 1000

        pupil = util.tel.Pupil( self.pxlsPerSubap * self.subaps, self.pxlsPerSubap *self.subaps/2.,0)
        mask = util.sor.createSorMask(pupil,self.pxlsPerSubap,0.5).ravel()
        avPist = (mask>0).astype("i").sum()
        tmparr = numpy.zeros( (2,self.subaps,self.subaps) ,"f")
        sorstr = cmod.sor.init( mask,avPist , conv , maxiters , tmparr )

        phaseArray = numpy.zeros( (slopes2d.shape[0],self.subaps,self.subaps),"d" )


        print("Reconstruction Phase....")
        P = progressbar.ProgressBar()
        P.start()
        for i in xrange(slopes2d.shape[0]):
            xFrame = slopes2d[i,:,:,0]
            yFrame = slopes2d[i,:,:,1]

            cmod.sor.fit(xFrame,yFrame,phaseArray[i],sorstr)

            P.update(100*i/slopes2d.shape[0])
        P.finish()

        return phaseArray





    def zernikeArray(self,n,size):
        '''
        Generates an array of zernike polynomials sorted in the Noll j arrangement
        '''
        J = numpy.arange(n+2).sum()
        zernikes = numpy.empty( (size,size,J) )

        j=0
        for n in xrange(n+1):
            if n%2==0:
                m_lower = 0
            else:
                m_lower = 1
            for m in xrange(m_lower,n+2,2):
                zernikes[:,:,j] = circle.zernike(-m,n,size)
                j+=1
                if m!=0:
                    Z =  circle.zernike(m,n,size)
                    zernikes[:,:,j] = Z/numpy.sqrt( (Z**2).sum())
                    j+=1

        zernikes /= numpy.sqrt( (zernikes**2).sum(0) ) #normalise the polynomials
        return zernikes


    def makeSubaps(self,data):
        '''
        downsampls an array of 2d images into smaller blocks, analogous to subaps on a SH sensor
        input data shape is (dim1,dim2,frames)
        output is (subaps,subaps,frames)
        '''

        subaps = self.subaps
        newData=numpy.empty( (subaps,subaps,data.shape[2]) )

        xSubapSize = data.shape[0]/subaps
        ySubapSize = data.shape[1]/subaps


        for x in xrange(subaps):
            for y in xrange(subaps):

                newData[x,y,:] = data[xSubapSize*x:xSubapSize*(x+1),ySubapSize*y:ySubapSize*(y+1),:].mean(0).mean(0)

        return newData


    def remapSubaps(self,data):
        '''
        Takes WFS data in format (frames,slopes) and remaps them back to the 2d
        pupil from which they came, giving both x and y frames
        return shape (frames,2,subaps,subaps)
        '''

        data=data.copy()
        data.shape = data.shape[0],data.shape[1]/2,2
        subaps = self.subaps

        mask = circle.circleGen(subaps/2.,subaps)
        newData=numpy.zeros(( data.shape[0],2,subaps,subaps) )

        j=0
        for x in xrange(subaps):
            for y in xrange(subaps):

                if mask[x,y] == 1:

                    newData[:,:,x,y] = data[:,j,:]
                    j+=1

        return newData




    def reconZernikePowerSpectrum(self,J):
        '''
        Generate a zernike breakdown powerspectrum by first reconstructing the wavefront,
        then comparing with an array of zernike modes.
        '''

        wavefront = self.reconSOR()     #First reconstruct wavefront
        wavefront -=wavefront.mean(0)   #And remove piston.

        zernikes = zerns.zernikeArray(J,wavefront.shape[1])

        coEffs = numpy.empty( (wavefront.shape[0], J) )

        for i in xrange(wavefront.shape[0]):
            for j in xrange(J):
                coEffs[i,j] = ((wavefront[i]*zernikes[j]).sum()) / zernikes[0].sum()
        PS = coEffs.var(0)
        return PS

    def zernikeSpectrumR0(self,PS,error=False):

        #Load up reference noll matrix for D/r0 =1  with which we will compare our PS
        noll = FITS.Read("noll.fits")[1]
        nollPS = noll.diagonal()[2:PS.shape[0]-1] #Only get the required zernikes(don't include TT)

        ratio = PS[3:]/nollPS       #Find ratio to Ref zernike variances, don't include TT or piston

        Dr0 = ratio.mean()**(3./5)         #Use ratio and handy scaling law (Noll 1976) to get D/r0

        if error:
            error = ratio.var()

        return Dr0


    def errorEstimation(self,n):
        '''
        Generate an estimate of noise for the "reconZernikePowerSpectrum" function
        split data up into 10 chunks, then find variance. Use this to get error for
        sample 10 times bigger.

        ***ONLY WORKS WITH DATA SETS WHICH ARE MULTIPLES OF 10***
        '''

        wavefront = self.reconSOR()
        wavefront -=wavefront.mean(0)

        wavefront.resize(10,wavefront.shape[0]/10,self.subaps**2)
        zernikes = self.zernikeArray(n,self.subaps)
        iZernikes = numpy.linalg.pinv(zernikes.reshape(self.subaps**2,zernikes.shape[2]))

        PSs = numpy.empty((10,zernikes.shape[2]))

        for i in xrange(10):

            PSs[i] = iZernikes.dot(wavefront[i].reshape(wavefront.shape[1],self.subaps**2).T).var(1)

        error = (PSs.var(0)/10.)**0.5

        return error

    def gradZernikePowerSpectrum(self,n):
        '''
        Generate a breakdown of zernike modes, using the gradients of zernike mode
        This requires no reconstruction of the wavefront.
        INPUT: int n, the max radial order of Zernike  modes desired to be analysis
        OUTPUT: numArray  PowerSpectrum, a 2xJ array, with x and y power spectrums respectively
        '''
        print("remap slopes")
        slopes = self.remapSubaps(self.workingSlopes) #remaps the subaps onto a circular pupil
        print("Make Gammas")
        GX,GY = make_gammas.makegammas(n)   #Generate the Noll gamma matrix for zernike gradients
        J = GX.shape[0]
        print("Make Zerns")
        print((J,self.subaps))
        Z = zerns.zernikeArray(J,self.subaps)
        G = numpy.zeros( (2,J,J) )
        G[0] = GX
        G[1] = GY

        print("Get co-effs")
        GZ = numpy.empty( (2,J,self.subaps,self.subaps) ) #Use Gamma matrices to get zernike gradients in X and Y
        for dim in xrange(2):
            for i in xrange(J):
                for j in xrange(J):
                    GZ[dim,i] += Z[j]*G[dim,i,j]

        coEffs = numpy.zeros( (2,slopes.shape[0],J) )   #initialise a matrix to hold all zernike co-efficients for each frame
        for dim in xrange(2):                           #Cycle through frames, and multiply and integrate with each zernike gradient to get co-efficients
            for i in xrange(slopes.shape[0]):
                for j in xrange(J):
                    coEffs[dim,i,j] = (slopes[i,dim]*Z[j]).sum()

        PS = numpy.zeros( (2,J) )                       #Arrange in to one array, and get variance of Zernike o-efficients.
        PS[0] = coEffs[0].var(0)
        PS[1] = coEffs[1].var(0)

        return PS


    def covMap(self,size = None):
        '''
        Generates Covariance maps for a set of slopes using a 2d FFT method
        Returns XX, YY, XY covariance Maps respectively
        '''

        slopes = self.workingSlopes.copy()
        subaps = self.subaps
        if size == None:
            size = 64
        slopes2d = self.remapSubaps(slopes)
        fslopes2d = numpy.fft.fft2(slopes2d,s=(size,size))

        XXCorrelation = fslopes2d[:,0].conjugate()*fslopes2d[:,0]
        YYCorrelation = fslopes2d[:,1].conjugate()*fslopes2d[:,1]
        XYCorrelation = fslopes2d[:,0].conjugate()*fslopes2d[:,1]

        XXCovMap = numpy.fft.fftshift(numpy.fft.ifft2(XXCorrelation))
        YYCovMap = numpy.fft.fftshift(numpy.fft.ifft2(YYCorrelation))
        XYCovMap = numpy.fft.fftshift(numpy.fft.ifft2(XYCorrelation))


        covMap = numpy.zeros( (3,size,size) )

        covMap[0] = abs(XXCovMap.mean(0))
        covMap[1] = abs(YYCovMap.mean(0))
        covMap[2] = abs(XYCovMap.mean(0))

        return covMap


#    def wavefrontPS(self,n):
#        '''
#        Uses reconstructed phase to dot with zernike modes and generate power spectrum
#        '''
#
#        phase = self.reconSOR()
#        zernikeArray = self.zernikeArray(n,self.subaps)



