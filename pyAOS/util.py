from multiprocessing import Process,Queue
import numpy


def circle(radius, diameter, centre_offset = (0,0)):
    '''
    creates an array of size (diameter,diameter) of zeros 
    with a circle shape of ones with radius <radius>'''
    
    #Create a mesh of radius values
    coords = numpy.linspace(-diameter/2.,diameter/2., diameter)
    X,Y = numpy.meshgrid(coords, coords)
    X-=centre_offset[0]
    Y-=centre_offset[1]
    r2_mesh = X**2 + Y**2
    
    #Create array of zeros to create circle on
    circle = numpy.zeros((diameter, diameter))
    
    #Find points where radius is smaller than the circle
    radius2 = radius**2
    c_coords = numpy.where(r2_mesh<=radius2)
    
    #Set these points to 1 on the zero array
    circle[c_coords]=1
    
    return circle



def run_mp_func(func, params, nProcs, data=None, iters=None):
    
    procs = []
    queues = []
    params = list(params)
    
    if data!=None:
        iters=data.shape[0]
    
    for p in xrange(nProcs):
        
        queues.append(Queue())
        
        if data!=None:
            procData = data[p::nProcs]
            procs.append(Process(target=mp_func,
                    args=[queues[p], func, params, procData]))
        else:
            procs.append(Process(target=func,
                    args=[queues[p], func, params]))

    for p in xrange(nProcs):
        procs[p].start()
    
    returnData = [0]*iters
    for p in xrange(nProcs):
        returnData[p::nProcs] = queues[p].get()
        
    return returnData
        

def mp_func(Q,func,params,data=None):
    
    argList = params
    if data!=None:
        argList.append(data)
    
    result = func(*argList)
    
    Q.put(result)
    
    