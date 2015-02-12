#Copyright Durham University and Andrew Reeves
#2014

# This file is part of pyAOS.

#     pyAOS is free software: you can redistribute it and/or modify
#     it under the terms of the GNU General Public License as published by
#     the Free Software Foundation, either version 3 of the License, or
#     (at your option) any later version.

#     pyAOS is distributed in the hope that it will be useful,
#     but WITHOUT ANY WARRANTY; without even the implied warranty of
#     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#     GNU General Public License for more details.

#     You should have received a copy of the GNU General Public License
#     along with pyAOS.  If not, see <http://www.gnu.org/licenses/>.

'''
A Module to perform FFTs, wrapping a variety of FFT Backends in a common
interface.
Currently supports either pyfftw (requires FFTW3), the scipy fftpack or some GPU algorithms
'''

import numpy
from . import logger
from multiprocessing import cpu_count,Process,Queue


try:
    import pyfftw
    PYFFTW_AVAILABLE=True
except:
    PYFFTW_AVAILABLE=False

try:
    import reikna.cluda as cluda
    import reikna.fft
    REIKNA_AVAILABLE=True
except:
    REIKNA_AVAILABLE=False

try:
    import scipy.fftpack as fftpack
    SCIPY_AVAILABLE = True
except:
    SCIPY_AVAILABLE = False

class FFT(object):
    '''
    Class for performing FFTs in a variety of ways, with the same API.

    Once the class has been initialised, FFTs going in the same direction and 
    using the same padding size can be performed with re-initialising. The 
    inputSize set is actually the padding size, any array smaller than this can
    then be transformed. 
    
    Usually, its best to best to pass the data when performing the fft, either
    calling the class directly (``fftobj(fftData)``) or calling the ``fft`` method of the 
    class. If though, you're certain the array to transform is C-contiguous, 
    and its size is the same as ``inputSize``, then you can set:: 
        
        fftObj.inputData = fftData
    
    then::
        
        outputData = fftObj()
        
    where ``fftData`` is the data to be transformed. This is faster, as it avoids
    an array copying operation, but is dangerous as the FFT may fail if the 
    input data is not correct.


    Parameters:
        inputSize (tuple): The size of the input array, including any padding
        axes (tuple, optional): The axes to transform. defaults to the last.
        mode (string, optional): Which FFT library to use, can by ``'pyfftw'``, ``'scipy'`` or ``'gpu'``. Defaults to ``'pyfftw'``.
        dtype (string, optional): The data type to transform, defaults to ``'complex64'``
     '''

    def __init__(self,inputSize, axes=(-1,),mode="pyfftw",dtype="complex64",
                    direction="FORWARD",fftw_FLAGS=("FFTW_MEASURE","FFTW_DESTROY_INPUT"),
                    THREADS=None, loggingLevel=None):
        self.axes = axes
        self.direction=direction

        if loggingLevel:
            logger.setLoggingLevel(loggingLevel)

        if mode=="gpu" or mode=="gpu_ocl" or mode=="gpu_cuda":
            if mode == "gpu":
                mode = "gpu_ocl"
            if REIKNA_AVAILABLE:
                if mode=="gpu_ocl":
                    try:
                        reikna_api = cluda.ocl_api()
                        self.reikna_thread = reikna_api.Thread.create()
                        self.FFTMODE="gpu"
                    except:
                        logger.warning("no reikna opencl available. \
                                            will try cuda")
                        mode = "gpu_cuda"
                if mode=="gpu_cuda":
                    try:
                        reikna_api = cluda.cuda_api()
                        self.reikna_thread = reikna_api.Thread.create()
                        self.FFTMODE="gpu"
                    except:
                        logger.warning("no cuda available. \
                                Switching to pyfftw")
                        mode = "pyfftw"
            else:
                logger.warning("No gpu algorithms available\
                        switching to pyfftw")
                mode = "pyfftw"

        if mode=="pyfftw":
            if PYFFTW_AVAILABLE:
                self.FFTMODE = "pyfftw"
            else:
                logger.warning("No pyfftw available. \
                                Defaulting to scipy.fftpack")
                mode = "scipy"

        if mode=="scipy":
            if SCIPY_AVAILABLE:
                self.FFTMODE = "scipy"
            else:
                logger.warning("No scipy available - fft won't function.")


        if self.FFTMODE=="gpu":
            if direction=="FORWARD":
                self.inverse=1
            elif direction=="BACKWARD":
                self.inverse=0

            self.inputData = numpy.zeros( inputSize, dtype=dtype)
            inputData_dev = self.reikna_thread.to_device(self.inputData)
            self.outputData_dev = self.reikna_thread.array(inputSize,
                                                     dtype=dtype)

            logger.info("Generating and compiling reikna gpu fft plan...")
            reikna_ft = reikna.fft.FFT(inputData_dev, axes=axes)
            self.reikna_ft_c = reikna_ft.compile(self.reikna_thread)
            logger.info("Done!")

        if self.FFTMODE=="pyfftw":
            if THREADS==None:
                THREADS=cpu_count()

            #fftw_FLAGS Set the optimisation level of fftw3,
            #(more optimisation takes longer - but gives quicker ffts.)
            #Can be FFTW_ESTIMATE, FFTW_MEASURE, FFT_PATIENT, FFTW_EXHAUSTIVE
            n = pyfftw.simd_alignment

            self.inputData = pyfftw.n_byte_align_empty( inputSize,n,
                                dtype)
            self.inputData[:] = numpy.zeros( inputSize, dtype=dtype)
            self.outputData = pyfftw.n_byte_align_empty(inputSize,n,
                                dtype)
            self.outputData[:] = numpy.zeros( inputSize,dtype=dtype)

            logger.info("Generating fftw3 plan....\nIf this takes too long, change fftw_FLAGS.")
            logger.debug("currently set to: {})".format(fftw_FLAGS))
            if direction=="FORWARD":
                self.fftwPlan = pyfftw.FFTW(self.inputData,self.outputData,
                                axes=axes, threads=THREADS,flags=fftw_FLAGS)
            elif direction=="BACKWARD":
                self.fftwPlan = pyfftw.FFTW(self.inputData,self.outputData,
                                direction='FFTW_BACKWARD', axes=axes,
                                threads=THREADS,flags=fftw_FLAGS)
            logger.info("Done!")


        elif self.FFTMODE=="scipy":

            self.direction=direction
            self.inputData = numpy.zeros(inputSize,dtype=dtype)
            self.size=[]
            for i in range(len(self.axes)):
                self.size.append(inputSize[self.axes[i]])


    def __call__(self,data=None):

        return self.fft(data)


    def fft(self,data=None):
        """
        Perform the fft of `data`.
        
        Parameters:
            data (ndarray, optional): The data to transform. Optional as sometimes it can be faster to access ``inputData`` directly, though if and only if the data will be c-contiguous.

        Returns:
            ndarray: The transformed data
        """
        if self.FFTMODE=="pyfftw":

            if data!=None:
                self.inputData[:] = data
            if self.direction=="FORWARD":
                return self.fftwPlan()
            elif self.direction=="BACKWARD":
                return self.fftwPlan()

        elif self.FFTMODE=="gpu":

            if data!=None:
                self.inputData[:] = data

            inputData_dev = self.reikna_thread.to_device(self.inputData)

            self.reikna_ft_c(self.outputData_dev, inputData_dev, self.inverse)

            self.outputData = self.reikna_thread.from_device(
                                        self.outputData_dev)

            return  self.outputData



        elif self.FFTMODE=="scipy":
            if data!=None:
                 self.inputData = data
            if self.direction=="FORWARD":
                fft = fftpack.fftn(self.inputData,
                                shape=self.size, axes=self.axes)
            elif self.direction=="BACKWARD":
                fft = fftpack.ifftn(self.inputData,
                                shape=self.size,axes=self.axes)
            return fft



class mpFFT(object):
    '''
    Class to perform FFTs on a large number of problems, using the FFT class, and seperate processes for different problems.
    The input array will be split in the 0 axis onto different processes
    '''
    def __init__(self,inputSize, axes=(-1,),mode="pyfftw",dtype="complex64",
                    direction="FORWARD",fftw_FLAGS=("FFTW_MEASURE",),
                    processes=None):

        if processes==None:
            processes = cpu_count()

        if len(inputSize)<=len(axes):
            raise Exception("inputSize must be larger than transformed axes")

        self.inputData = numpy.zeros(inputSize,dtype=dtype)
        self.outputData = numpy.zeros(inputSize,dtype=dtype)

        self.processes = processes
        inputSize = list(inputSize)
        self.FFTs = []
        for i in range(self.processes):
            procInputSize0 = (self.inputData[i::self.processes].shape[0])
            print(procInputSize0)
            inputSize[0]=procInputSize0
            procInputSize = tuple(inputSize)
            self.FFTs.append(FFT(procInputSize,axes,mode,dtype,direction,
                                fftw_FLAGS,THREADS=1))

    def __call__(self):
        return self.fft()

    def fft(self):

        self.Qs = []
        self.Ps =[]

        for i in range(self.processes):
            data = self.inputData[i::self.processes]

            self.Qs.append(Queue())
            self.Ps.append(Process(target=self.doMpFFT,
                                args=[self.FFTs[i],data,self.Qs[i]]))

            self.Ps[i].start()


        for i in range(self.processes):

            self.outputData[i::self.processes] = self.Qs[i].get()
            self.Ps[i].join()

        return self.outputData

    def doMpFFT(self,fftObj,data,Q):
        fftObj.inputData[:]=data

        fftData = fftObj()

        Q.put(fftData)


def ftShift2d(inputData, outputData=None):
    """
    Helper function to shift and array of 2-D FFT data

    Args:
        inputData(ndarray): array of data to be shifted. Will shift final 2 axes
        outputData(ndarray, optional): array to place data. If not given, will overwrite inputData
    """
    if not outputData:
        outputData = inputData.view()

    shape = inputData.shape

    outputData[..., :shape[-2]*0.5, :shape[-1]*0.5] = inputData[..., :shape[-2]*0.5, :shape[-1]*0.5][...,::-1,::-1]
    outputData[..., shape[-2]*0.5:, :shape[-1]*0.5] = inputData[..., shape[-2]*0.5:, :shape[-1]*0.5][...,::-1,::-1]
    outputData[..., :shape[-2]*0.5, shape[-1]*0.5:] = inputData[..., :shape[-2]*0.5, shape[-1]*0.5:][...,::-1,::-1]
    outputData[..., shape[-2]*0.5:, shape[-1]*0.5:] = inputData[..., shape[-2]*0.5:, shape[-1]*0.5:][...,::-1,::-1]

    return outputData

class Convolve(object):
    
    def __init__(self, shape, mode="pyfftw", fftw_FLAGS=("FFTW_MEASURE",), threads=0):
        #Initialise FFT objects
        self.fFFT = FFT(shape, axes=(0,1), mode=mode,
                        dtype="complex64",direction="FORWARD", 
                        fftw_FLAGS=fftw_FLAGS, THREADS=threads) 
        self.iFFT = FFT(shape, axes=(0,1), mode=mode,
                        dtype="complex64",direction="BACKWARD",
                        fftw_FLAGS=fftw_FLAGS, THREADS=threads)

    def __call__(self, img1, img2):
        #backward FFT arrays
        self.iFFT.inputData[:] = img1
        iImg1 = self.iFFT().copy()  
        self.iFFT.inputData[:] = img2
        iImg2 = self.iFFT()
    
        #Do convolution
        iImg1 *= iImg2

        #do forward FFT
        self.fFFT.inputData[:] = iImg1
        return numpy.fft.ffshift(self.fFFT())
        
