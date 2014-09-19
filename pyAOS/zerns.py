import sys
#sys.path.append("dasp_util")
from .dasp_util import zernikeMod
from .dasp_util import tel
import numpy




def zernikeArray(J,size):

    pup = tel.Pupil(size,size/2.,0).fn
    
    Z = zernikeMod.Zernike(pup,J).zern
   # for i in range(J):

   #     Z[i]/= numpy.sqrt((Z[i]**2).sum())

    return Z

       
        





        

        
