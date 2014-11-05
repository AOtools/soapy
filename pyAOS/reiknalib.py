from reikna.core import Transformation, Annotation, Parameter, Type, Computation
from reikna.fft import FFT as rFFT
from reikna import cluda
import reikna.helpers


import pylab
import numpy
import time
from scipy import interpolate

##########################
#Transformations
##########################
def absSquare_tr(shape):
    
    absSquare = Transformation([
        Parameter("absSq", Annotation(Type("float32", shape=shape), "o")),
        Parameter("complexNum", Annotation(Type("complex64", shape=shape),"i"))
        ],
    """
    ${absSq.store_same}( 
        ${complexNum.load_same}.x * ${complexNum.load_same}.x 
        + ${complexNum.load_same}.y * ${complexNum.load_same}.y 
        );
    """,
    connectors=["complexNum", "absSq"]
    )
    
    return absSquare


###########################
#Computations
###########################    
class Interp1dGPU(Computation):
    
    def __init__(self, inputArr, outputArr, coordArr):
        
        Computation.__init__(self, [
            Parameter('output', Annotation(outputArr, 'o')),
            Parameter('inputArray', Annotation(inputArr, 'i')),
            Parameter('xCoords', Annotation(coordArr, 'i')),
            ])

    def _build_plan(   self, plan_factory, device_params,
                        output, inputArray, xCoords):
        plan = plan_factory()

        template = reikna.helpers.template_from(
        """
        <%def name='interp1d(kernel_decleration, k_output, k_input, k_xCoords)'>
        ${kernel_decleration}
        {
        VIRTUAL_SKIP_THREADS;
        const VSIZE_T idx = virtual_global_id(0);
        const ${k_xCoords.ctype} x = ${k_xCoords.load_idx}(idx);

        const int xInt = (int) x;
        const int xInt1 = (int) xInt+1;
        
        const float xRemainder = (float) x - (float) xInt;
        
        const float grad = (float) ${k_input.load_idx}(xInt1) - (float) ${k_input.load_idx}(xInt);
        
        const ${k_output.ctype} value = (${k_output.ctype}) ${mul}(grad, xRemainder) + ${k_input.load_idx}(xInt);

        ${k_output.store_idx}(idx, value);
        
        printf("i: %d, x: %.4f, xInt: %d, xRem: %.4f, value:%.4f\\n", idx, x, xInt, xRemainder, value);
        }
        </%def>
        """)

        plan.kernel_call(
                template.get_def('interp1d'),
                [output, inputArray, xCoords],
                global_size=xCoords.shape,
                render_kwds={
                    'mul' : cluda.functions.mul(
                                "float32", "float32")
                    }
                )
        return plan

class Interp2dGPU(Computation):
    
    def __init__(self, outputArr, inputArr, coordArr):
        
        Computation.__init__(self, [
            Parameter('output', Annotation(outputArr, 'o')),
            Parameter('inputArray', Annotation(inputArr, 'i')),
            Parameter('xCoords', Annotation(coordArr, 'i')),
            Parameter('yCoords', Annotation(coordArr, 'i'))
            ])
    
    def _build_plan(    self, plan_factory, device_params,
                        output, inputArray, xCoords, yCoords):
        plan = plan_factory()

        template = reikna.helpers.template_from(
        """
        <%def name='interp2d(kernel_decleration, k_output, k_input, k_xCoords, k_yCoords)'>
        ${kernel_decleration}
        {
        VIRTUAL_SKIP_THREADS;
        
        const VSIZE_T i = virtual_global_id(0);
        const VSIZE_T j = virtual_global_id(1);
        
        const ${k_xCoords.ctype} x = ${k_xCoords.load_idx}(i);
        const ${k_yCoords.ctype} y = ${k_yCoords.load_idx}(j);

        const int xInt = (int) x;
        const int yInt = (int) y;

        const int xInt1 = (int) (xInt+1);
        const int yInt1 = (int) (yInt+1);

        const ${k_xCoords.ctype} xRem = (${k_xCoords.ctype}) x - (${k_xCoords.ctype}) xInt;
        const ${k_yCoords.ctype} yRem = (${k_yCoords.ctype}) y - (${k_yCoords.ctype}) yInt;

        const ${k_input.ctype} xGrad = ${sum2}(${k_input.load_idx}(
                            xInt1, yInt), 
                    ${mulConst}(${k_input.load_idx}(xInt, yInt),-1));
        const ${k_input.ctype} yGrad = ${k_input.load_idx}(xInt, yInt1)
                    - ${k_input.load_idx}(xInt, yInt);
                    
        const ${k_output.ctype} value =${sum3}(${mul}(xRem, xGrad), ${mul}(yRem, yGrad), ${k_input.load_idx}(xInt,yInt));

        ${k_output.store_idx}(i, j, value);

        }
        </%def>
        """)

        plan.kernel_call(
                template.get_def('interp2d'),
                [output, inputArray, xCoords, yCoords],
                global_size=(output.shape),
                render_kwds={

                    'mul' : cluda.functions.mul(xCoords.dtype,inputArray.dtype,
                                                out_dtype=inputArray.dtype),
                    'sum3' : cluda.functions.add(
                            inputArray.dtype, inputArray.dtype, 
                            inputArray.dtype, out_dtype=output.dtype),
                    'mulConst':cluda.functions.mul(
                            inputArray.dtype, numpy.float32,
                            out_dtype=inputArray.dtype),
                    'sum2'  : cluda.functions.add(
                            inputArray.dtype, inputArray.dtype,
                            out_dtype=inputArray.dtype),
                    }
                )
        return plan
    
class MultiplyArrays1d(Computation):

    def __init__(self, arr):
        if len(arr.shape) != 1:
            raise ValueError("MultiplyArrays1d only works on 1d data")

        Computation.__init__(self, [
            Parameter('output', Annotation(arr, 'o')),
            Parameter('input1', Annotation(arr, 'i')),
            Parameter('input2', Annotation(arr, 'i')),
           ])


    def _build_plan(    self, plan_factory, device_params, 
                        output,  input1, input2):

        plan = plan_factory()

        template = reikna.helpers.template_from(
                """
                <%def name='testcomp(kernel_declaration, k_output, k_input1, k_input2)'>
                ${kernel_declaration}
                {
                    VIRTUAL_SKIP_THREADS;
                    const VSIZE_T idx = virtual_global_id(0);
                    const ${k_output.ctype} result = 
                        ${mul}( ${k_input1.load_idx}(idx),
                                ${k_input2.load_idx}(idx));
                    ${k_output.store_idx}(idx, result);
                }
                </%def>
                """)

        plan.kernel_call(
                template.get_def('testcomp'),
                [output, input1, input2],
                global_size=output.shape,
                render_kwds={
                        "mul" : cluda.functions.mul(
                                        input1.dtype, input2.dtype
                                        )
                            }
                        )                                                
        return plan    
        
class MultiplyArrays2d(Computation):

    def __init__(self, arr):
        if len(arr.shape) != 2:
            raise ValueError("MultiplyArrays2d only works on 2d data")

        Computation.__init__(self, [
            Parameter('output', Annotation(arr, 'o')),
            Parameter('input1', Annotation(arr, 'i')),
            Parameter('input2', Annotation(arr, 'i')),
           ])

    def _build_plan(    self, plan_factory, device_params, 
                        output,  input1, input2):

        plan = plan_factory()

        template = reikna.helpers.template_from(
                """
                <%def name='mul2d(kernel_declaration, k_output, k_input1, k_input2)'>
                ${kernel_declaration}
                {
                    VIRTUAL_SKIP_THREADS;
                    
                    const VSIZE_T idx = virtual_global_id(0);
                    const VSIZE_T idy = virtual_global_id(1);
                    
                    const ${k_output.ctype} result = 
                        ${mul}( ${k_input1.load_idx}(idx, idy),
                                ${k_input2.load_idx}(idx, idy));
                    ${k_output.store_idx}(idx, idy, result);
                }
                </%def>
                """)

        plan.kernel_call(
                template.get_def('mul2d'),
                [output, input1, input2],
                global_size=output.shape,
                render_kwds={
                        "mul" : cluda.functions.mul(
                                        input1.dtype, input2.dtype
                                        )
                            }
                        )                                                
        return plan
 
        
class MakeSubaps(Computation):

    def __init__(self, subaps, inputArray, subapCoords):

        Computation.__init__(self, [
            Parameter('subaps', Annotation(subaps, 'o')),
            Parameter('inputArray', Annotation(inputArray, 'i')),
            Parameter('subapCoords', Annotation(subapCoords, 'i'))]
            )

    def _build_plan(    self, plan_factory, device_params, 
                        subaps,  inputArray, subapCoords):

        plan = plan_factory()

        template = reikna.helpers.template_from(
        """
        <%def name='makeSubaps(kernel_declaration, k_subaps, k_inputArray, k_subapCoords)'>
        ${kernel_declaration}
        {
            VIRTUAL_SKIP_THREADS;
            const VSIZE_T i = virtual_global_id(0);
            const VSIZE_T j = virtual_global_id(1);
            const VSIZE_T k = virtual_global_id(2);
            
            const VSIZE_T x = (VSIZE_T) ${k_subapCoords.load_idx}(i, 0);
            const VSIZE_T y = (VSIZE_T) ${k_subapCoords.load_idx}(i, 1);
            
            const VSIZE_T xIn = (VSIZE_T) (x + j);
            const VSIZE_T yIn = (VSIZE_T) (y + k);
            
            const ${k_subaps.ctype} value = ${cast}(${k_inputArray.load_idx}(xIn,yIn));
            
            ${k_subaps.store_idx}(i, j, k, value);
        }
        </%def>
        """)

        plan.kernel_call(
                template.get_def('makeSubaps'),
                [subaps, inputArray, subapCoords],
                global_size=subaps.shape,
                render_kwds={'cast' : cluda.functions.cast(
                                        subaps.dtype, inputArray.dtype)
                            }
                )
                                                        
        return plan

class BinImgs(Computation):
    """
    Computation to bin arrays of 2d images into smaller arrays of 2d images.
    The input images must be an integer multiple of the binned images
    """


    def __init__(self, binnedArray, inputArray, binFactor):

        
        Computation.__init__(self, [
            Parameter('binnedArray', Annotation(binnedArray, 'o')),
            Parameter('inputArray', Annotation(inputArray, 'i')),
            Parameter('binFactor', Annotation(binFactor))
            ])
            
        

    def _build_plan(    self, plan_factory, device_params, 
                        binnedArray,  inputArray, binFactor):

        plan = plan_factory()

        template = reikna.helpers.template_from(
        """
        <%def name='binImgs(kernel_declaration, k_binnedArray, k_inputArray, k_binFactor)'>
        ${kernel_declaration}
        {
            VIRTUAL_SKIP_THREADS;
            const VSIZE_T i = virtual_global_id(0);
            const VSIZE_T j = virtual_global_id(1);
            const VSIZE_T k = virtual_global_id(2);
            
            const VSIZE_T binFactor = ${k_binFactor};
            
            const VSIZE_T x1 = j * (VSIZE_T) binFactor;
            const VSIZE_T y1 = k * (VSIZE_T) binFactor;
            
            ${k_binnedArray.ctype} value = 0;
            for(VSIZE_T x = x1; x<(x1+binFactor); x++){
                for(VSIZE_T y = y1; y<(y1+binFactor); y++){
                    value = value + ${k_inputArray.load_idx}(i,x,y);
                }
            }
            
            ${k_binnedArray.store_idx}(i, j, k, ${cast}(value));
        }
        </%def>
        """)

        plan.kernel_call(
                template.get_def('binImgs'),
                [binnedArray, inputArray, binFactor],
                global_size=binnedArray.shape,
                render_kwds={'cast' : cluda.functions.cast(
                                        binnedArray.dtype, inputArray.dtype)
                            }
                )
                                                        
        return plan

    
class PadArrays(Computation):
    """
    Computation to bin arrays of 2d images into smaller arrays of 2d images.
    The input images must be an integer multiple of the binned images
    """


    def __init__(self, paddedArray, inputArray):

        
        Computation.__init__(self, [
            Parameter('paddedArray', Annotation(paddedArray, 'o')),
            Parameter('inputArray', Annotation(inputArray, 'i')),
            ])
            
        

    def _build_plan(    self, plan_factory, device_params, 
                        paddedArray,  inputArray):

        plan = plan_factory()

        template = reikna.helpers.template_from(
        """
        <%def name='padArrays(kernel_declaration, k_paddedArray, k_inputArray)'>
        ${kernel_declaration}
        {
            VIRTUAL_SKIP_THREADS;
            const VSIZE_T i = virtual_global_id(0);
            const VSIZE_T j = virtual_global_id(1);
            const VSIZE_T k = virtual_global_id(2);
            
            ${k_paddedArray.store_idx}(i, j, k, ${cast}(
                    ${k_inputArray.load_idx}(i,j,k)));
        }
        </%def>
        """)

        plan.kernel_call(
                template.get_def('padArrays'),
                [paddedArray, inputArray],
                global_size=inputArray.shape,
                render_kwds={'cast' : cluda.functions.cast(
                                        paddedArray.dtype, inputArray.dtype)
                            }
                )
                                                        
        return plan


class Sum2dto3d(Computation):
    """
    Computation to add a 2d array to the last 2 axes of a 2-dimensional array
    """


    def __init__(self, array3d, array2d ):

        
        Computation.__init__(self, [
            Parameter('array3d', Annotation(array3d, 'io')),
            Parameter('array2d', Annotation(array2d, 'i')),
            ])
            
        

    def _build_plan(    self, plan_factory, device_params, 
                        array3d,  array2d):

        plan = plan_factory()

        template = reikna.helpers.template_from(
        """
        <%def name='sumArrays(kernel_declaration, k_array3d, k_array2d)'>
        ${kernel_declaration}
        {
            VIRTUAL_SKIP_THREADS;
            const VSIZE_T i = virtual_global_id(0);
            const VSIZE_T j = virtual_global_id(1);
            const VSIZE_T k = virtual_global_id(2);
            
            ${k_array3d.ctype} value;



            ${k_array3d.store_idx}(i,j,k, ${sum}(
                            ${k_array3d.load_idx}(i,j,k),
                            ${cast}(${k_array2d.load_idx}(j,k))
                            )
                        );

        }
        </%def>
        """)

        plan.kernel_call(
                template.get_def('sumArrays'),
                [array3d, array2d],
                global_size=array3d.shape,
                render_kwds={'sum' : cluda.functions.add(
                                        array3d.dtype, array2d.dtype,
                                        out_dtype=array3d.dtype),
                            'cast' : cluda.functions.cast(
                                        array3d.dtype, array2d.dtype)
                            }
                )
                                                        
        return plan


class Mul2dto3d(Computation):
    """
    Computation to multiply the last 2 axes of a 3d array to a 2-dimensional array
    """


    def __init__(self, array3d, array2d ):

        
        Computation.__init__(self, [
            Parameter('array3d', Annotation(array3d, 'io')),
            Parameter('array2d', Annotation(array2d, 'i')),
            ])
            
        

    def _build_plan(    self, plan_factory, device_params, 
                        array3d,  array2d):

        plan = plan_factory()

        template = reikna.helpers.template_from(
        """
        <%def name='mulArrays(kernel_declaration, k_array3d, k_array2d)'>
        ${kernel_declaration}
        {
            VIRTUAL_SKIP_THREADS;
            const VSIZE_T i = virtual_global_id(0);
            const VSIZE_T j = virtual_global_id(1);
            const VSIZE_T k = virtual_global_id(2);
            
            ${k_array3d.ctype} value;



            ${k_array3d.store_idx}(i,j,k, ${mul}(
                            ${k_array3d.load_idx}(i,j,k),
                            ${cast}(${k_array2d.load_idx}(j,k))
                            )
                        );

        }
        </%def>
        """)

        plan.kernel_call(
                template.get_def('mulArrays'),
                [array3d, array2d],
                global_size=array3d.shape,
                render_kwds={'mul' : cluda.functions.mul(
                                        array3d.dtype, array2d.dtype,
                                        out_dtype=array3d.dtype),
                            'cast' : cluda.functions.cast(
                                        array3d.dtype, array2d.dtype)
                            }
                )
                                                        
        return plan

class FTShift2d(Computation):
    """
    Computation to preform an fft shift on an array of 2-d arrays (a 3-d array)
    """


    def __init__(self, array):

        
        Computation.__init__(self, [
            Parameter('inputArray', Annotation(array, 'o')),
            Parameter('outputArray', Annotation(array, 'i')),
            ])
            
        

    def _build_plan(    self, plan_factory, device_params, 
                        inputArray,  outputArray):

        plan = plan_factory()

        template = reikna.helpers.template_from(
        """
        <%def name='mulArrays(kernel_declaration, k_input, k_output, shapeX, shapeY)'>
        ${kernel_declaration}
        {
            VIRTUAL_SKIP_THREADS;
            const VSIZE_T i = virtual_global_id(0);
            const VSIZE_T j = virtual_global_id(1);
            const VSIZE_T k = virtual_global_id(2);
            
            ${k_array3d.ctype} value;

            ${k_array3d.store_idx}(i,j,k, ${mul}(
                            ${k_array3d.load_idx}(i,j,k),
                            ${cast}(${k_array2d.load_idx}(j,k))
                            )
                        );
        }
        </%def>
        """)

        plan.kernel_call(
                template.get_def('mulArrays'),
                [array3d, array2d],
                global_size=array3d.shape,
                render_kwds={'mul' : cluda.functions.mul(
                                        array3d.dtype, array2d.dtype,
                                        out_dtype=array3d.dtype),
                            'cast' : cluda.functions.cast(
                                        array3d.dtype, array2d.dtype)
                            }
                )
                                                        
        return plan

##############################
#Combined Calculations
#############################
    
def ftAbs(outputArray, inputArray, axes):
    
    

    
    ft = rFFT(Type(shape=outputArray.shape,dtype=inputArray.dtype), axes)

    absSq_tr = absSquare_tr(outputArray.shape)
    ft.parameter.output.connect( absSq_tr, absSq_tr.complexNum,
                                 out=absSq_tr.absSq)
    
    return ft


#####################
#Test Funcs

def testBin(data, binFactor, thr=None):
    
    if not thr:
        api = cluda.ocl_api()
        thr = api.Thread.create()
        
    data_gpu = thr.to_device(data)
    result_gpu = thr.to_device(numpy.zeros((data.shape[0], data.shape[1]/binFactor, data.shape[2]/binFactor)))
    
    binCalc = BinImgs(result_gpu, data_gpu, binFactor)
    c_binCalc = binCalc.compile(thr)
    
    c_binCalc(result_gpu, data_gpu, binFactor)
    
    return result_gpu.get()
    

def testFT(data):

    api = cluda.ocl_api()
    thr = api.Thread.create()

    data = data.astype("complex64")

    gpu_data = thr.to_device(data)
    gpu_result = thr.array(data.shape, dtype="float32")

    r_ftAbs = ftAbs(data.shape)

    c_ftAbs = r_ftAbs.compile(thr)

    c_ftAbs(gpu_result, gpu_data)

    cpu_result = abs(numpy.fft.fft(data))**2

    pylab.figure()
    pylab.title("CPU Data")
    pylab.subplot(2,1,1)
    pylab.plot(abs(data))
    pylab.subplot(2,1,2)
    pylab.plot(cpu_result)

    pylab.figure()
    pylab.title("GPU Data")
    pylab.subplot(2,1,1)
    pylab.plot(abs(gpu_data.get()))
    pylab.subplot(2,1,2)
    pylab.plot(gpu_result.get())

    print("success: {}".format(numpy.allclose(cpu_result, gpu_result.get())))

    return cpu_result, gpu_result

def benchFT(data, N=10000):
    api = cluda.ocl_api()
    thr = api.Thread.create()

    data = data.astype("complex64")

    gpu_data = thr.to_device(data)
    gpu_result = thr.array(data.shape, dtype="float32")

    r_ftAbs = ftAbs(data.shape)

    c_ftAbs = r_ftAbs.compile(thr)


    print("Benching GPU:")
    gpu_t0 = time.time()
    for i in range(N):
        c_ftAbs(gpu_result, gpu_data)
    gpuTime = (time.time()-gpu_t0)/float(N)

    print("Done GPU, starting CPU:")
    cpu_t0 = time.time()
    for i in range(N):
        cpu_result = abs(numpy.fft.fft(data))**2
    cpuTime = (time.time()-cpu_t0)/float(N)

    print("CPU FFT: {0:.6f}s, GPU FFT: {1: .6f}s".format(cpuTime, gpuTime))
    print("Speedup: {}x".format(cpuTime/gpuTime))


class testComp(Computation):

    def __init__(self, arr, coeff):
        if len(arr.shape) != 1:
            raise ValueError("testComp only works on 1d data")

        Computation.__init__(self, [
            Parameter('output', Annotation(arr, 'o')),
            Parameter('input1', Annotation(arr, 'i')),
            Parameter('input2', Annotation(arr, 'i')),
            Parameter('param', Annotation(coeff))])


    def _build_plan(    self, plan_factory, device_params, 
                        output,  input1, input2, param):

        plan = plan_factory()

        template = reikna.helpers.template_from(
                """
                <%def name='testcomp(kernel_declaration, k_output, k_input1, k_input2, k_param)'>
                ${kernel_declaration}
                {
                    VIRTUAL_SKIP_THREADS;
                    const VSIZE_T idx = virtual_global_id(0);
                    ${k_output.ctype} result = 
                        ${k_input1.load_idx}(idx) +
                        ${mul}(${k_input2.load_idx}(idx), ${k_param});
                    ${k_output.store_idx}(idx, result);
                }
                </%def>
                """)

        plan.kernel_call(
                template.get_def('testcomp'),
                [output, input1, input2, param],
                global_size=output.shape,
                render_kwds={
                        "mul" : cluda.functions.mul(
                                        input2.dtype, param.dtype
                                        )
                            }
                        )
                                                        
        return plan



def test_interp1d(data, newSize, thr=None):
    if not thr:
        api = cluda.ocl_api()
        thr = api.Thread.create()

    output_gpu = thr.to_device(numpy.zeros(newSize).astype(data.dtype))
    data_gpu = thr.to_device(data)

    coords = numpy.linspace(0,data.shape[0]-1,newSize)
    coords[-1] -= 1e-10
    coords_gpu = thr.to_device(coords)

    interp = Interp1dGPU(output_gpu, data_gpu, coords)
    c_interp = interp.compile(thr)

    c_interp(output_gpu, data_gpu, coords_gpu)

    return output_gpu.get()
    


def bench_interp1d(dataSize, newSize, N=1000):

    xdata = numpy.linspace(0,10,dataSize).astype("float32")
    x = numpy.arange(dataSize)
    data = numpy.sin(xdata).astype("float32")


    coords = numpy.linspace(0,data.shape[0]-1, newSize).astype("float32")
    coords[-1] -= 1e-10

    print("Benching CPU...")

    t_cpu=time.time()
    for i in xrange(N):
        I = interpolate.interp1d(x, data, kind="linear", copy=False)
        I(coords)
    t_cpu = time.time()-t_cpu

    print("Done!")

    api = cluda.ocl_api()
    thr = api.Thread.create()

    output_gpu = thr.to_device(numpy.zeros(newSize).astype(data.dtype))
    data_gpu = thr.to_device(data)
    coords_gpu = thr.to_device(coords)

    print(data_gpu.get())

    interp = Interp1dGPU(output_gpu, data_gpu, coords)
    c_interp = interp.compile(thr)

    print("Benching GPU...")
    t_gpu = time.time()
    for i in xrange(N):
        c_interp(output_gpu, data_gpu, coords_gpu)
    t_gpu = time.time()-t_gpu

    print("Done!\n GPU: {0}ms, CPU: {1}ms".format(t_gpu*1000, t_cpu*1000))
    print("Speedup: {:.4f}".format(t_cpu/t_gpu))





def test_interp2d(data, newSize, thr=None):
    if not thr:
        api = cluda.ocl_api()
        thr = api.Thread.create()

    if len(newSize)!=2 or len(data.shape)!=2:
        raise ValueError("2D Arrays only!")

    output_gpu = thr.to_device(numpy.zeros(newSize).astype(data.dtype))
    data_gpu = thr.to_device(data)

    xCoords = numpy.linspace(0, data.shape[0]-1, newSize[0]).astype("float32")
    yCoords = numpy.linspace(0, data.shape[1]-1, newSize[1]).astype("float32")
    xCoords[-1] -= 1e-5
    yCoords[-1] -= 1e-5


    xCoords_gpu = thr.to_device(xCoords)
    yCoords_gpu = thr.to_device(yCoords)


    print("xCoords:{}".format(xCoords))
    print("yCoords:{}".format(yCoords))
    print(data_gpu.get())

    interp = Interp2dGPU(output_gpu, data_gpu, xCoords_gpu)
    c_interp = interp.compile(thr)

    c_interp(output_gpu, data_gpu, xCoords_gpu, yCoords_gpu)
    print("compiled")

    return output_gpu.get()
    
    
def benchInterp2d(dataSize, newSize, thr=None, N=1000):
    
    if not thr:
        api = cluda.ocl_api()
        thr = api.Thread.create()
    
    data = numpy.random.random(dataSize).astype("float32")
    
    dataXCoords = numpy.arange(dataSize[0])
    dataYCoords = numpy.arange(dataSize[1])
    
    xCoords = numpy.linspace(0, dataSize[0]-1, newSize[0]).astype("float32")
    yCoords = numpy.linspace(0, dataSize[1]-1, newSize[1]).astype("float32")
    
    print("Bench CPU...")
    t0cpu = time.time()
    for i in xrange(N):
        cpuInterp2d = interpolate.interp2d(dataXCoords, dataYCoords, data, copy=False)
        cpuInterp2d(xCoords, yCoords)
    tcpu = (time.time()-t0cpu)/N
    
    print("Prepare GPU...")
    xCoords[-1] -= 1e-5
    yCoords[-1] -= 1e-5

    xCoords_gpu = thr.to_device(xCoords)
    yCoords_gpu = thr.to_device(yCoords)
    
    data_gpu = thr.to_device(data)
    result_gpu = thr.to_device(numpy.empty(newSize).astype("float32"))
    
    gpu_interp2d = Interp2dGPU(result_gpu, data_gpu, xCoords_gpu)
    c_gpu_interp2d = gpu_interp2d.compile(thr)
    
    print("Bench GPU...")
    t0gpu = time.time()
    for i in xrange(N):
        c_gpu_interp2d(result_gpu, data_gpu, xCoords_gpu, yCoords_gpu)
    tgpu = (time.time()-t0gpu)/N
        
    print("CPU: {:.4f}ms, GPU: {:.4f}ms  per iteration".format(1000*tcpu, 1000*tgpu))
    print("GPU Speedup: {:.2f}".format(tcpu/tgpu))
    
    
def interp1d_cpu(data, newSize):
    newData = numpy.empty(newSize)
    
    xCoords = numpy.linspace(0, data.shape[0]-1, newSize)
    xCoords[-1] -= 1e-9
    
    for i in range(newSize):
        x = xCoords[i]
        xInt = int(x)
        
        xRem = x-xInt
        xGrad = data[xInt+1] - data[xInt]

        newData[i] = xRem*xGrad + data[xInt]
        
    return newData
    
def interp2d_cpu(data, newSize):

    newData = numpy.empty(newSize)
    xCoords = numpy.linspace(0, data.shape[0]-1, newSize[0])
    yCoords = numpy.linspace(0, data.shape[1]-1, newSize[1])

    xCoords[-1] -= 1e-9
    yCoords[-1] -= 1e-9

    for i in range(newSize[0]):
        for j in range(newSize[1]):

            x = xCoords[i] 
            y = yCoords[j]

            xInt = int(x)
            yInt = int(y)

            xRem = x - xInt
            yRem = y - yInt 

            xGrad = data[xInt+1, yInt] - data[xInt, yInt]
            yGrad = data[xInt, yInt+1] - data[xInt, yInt]

            value = data[xInt, yInt]
            
            ivalue = (xRem*xGrad + yRem*yGrad + value) 
           
            print("x: %.3f, y: %.3f, xInt: %d, yInt: %d, xR: %.3f, yR: %.3fm in:%.2f, out:%.2f"% (x,y,xInt,yInt,xRem,yRem,value,ivalue))

            newData[i, j] = ivalue

    return newData


class dataTestComp(Computation):
    
    def __init__(self, inputArray, outputArray):
        
        Computation.__init__(self, [
            Parameter('output', Annotation(outputArray, 'o')),
            Parameter('inputArray', Annotation(inputArray, 'i')),
            ])


    def _build_plan(   self, plan_factory, device_params,
                        output, inputArray):
        plan = plan_factory()

        template = reikna.helpers.template_from(
        """
        <%def name='dataTest(kernel_decleration, k_output, k_input )'>
        ${kernel_decleration}
        {
        VIRTUAL_SKIP_THREADS;

        const VSIZE_T idx = virtual_global_id(0);
        const VSIZE_T idy = virtual_global_id(1);
    
        ${k_input.ctype} value = ${k_input.load_idx}(idx,idy);

        ##printf("data[%d, %d] = %.2f\\n", idx, idy, value);

        ${k_output.store_idx}(idx, idy, value); 
        }
        </%def>
        """)

        plan.kernel_call(
                template.get_def('dataTest'),
                [output, inputArray],
                global_size=output.shape,
                )
        print("outputShape: {}".format(output.shape))
        return plan


def dataTest(data, thr=None):
    if not thr:
        api = cluda.ocl_api()
        thr = api.Thread.create()

    data_gpu = thr.to_device(data)
    res_gpu = thr.to_device(numpy.zeros(data.shape, data.dtype))

    comp = dataTestComp(data_gpu)
    c_comp = comp.compile(thr)

    c_comp(res_gpu, data_gpu)

    return numpy.allclose(res_gpu.get(), data)






class Interp2dNearestGPU(Computation):
    
    def __init__(self, outputArr, inputArr, coordArr):
        
        Computation.__init__(self, [
            Parameter('output', Annotation(outputArr, 'o')),
            Parameter('inputArray', Annotation(inputArr, 'i')),
            Parameter('xCoords', Annotation(coordArr, 'i')),
            Parameter('yCoords', Annotation(coordArr, 'i'))
            ])


            #xGrad = data[xInt+1, yInt] - data[xInt, yInt]
            #yGrad = data[xInt, yInt+1] - data[xInt, yInt]
    
    def _build_plan(   self, plan_factory, device_params,
                        output, inputArray, xCoords, yCoords):
        plan = plan_factory()

        template = reikna.helpers.template_from(
        """
        <%def name='interp2dnearest(kernel_decleration, k_output, k_input, k_xCoords, k_yCoords)'>
        ${kernel_decleration}
        {
        VIRTUAL_SKIP_THREADS;

        VSIZE_T idx, idy;
        int xInt, yInt;
        float xRem, yRem, xGrad, yGrad;
        ${k_yCoords.ctype} x,y;
        ${k_output.ctype} value, ivalue;

        idx = virtual_global_id(0);
        idy = virtual_global_id(1);
    
        x = ${k_xCoords.load_idx}(idx);
        xInt = (int) x;

        y = ${k_yCoords.load_idx}(idy);
        yInt = (int) y;
        
        value = ${k_input.load_idx}(xInt, yInt);

        ${k_output.store_idx}(idx, idy, value);
        
        ##Print debug info
        ##printf("data[%d, %d] = %.2f,\\t data[%d, %d] = %.2f\\n", xInt, yInt,  value, Int+1, yInt+1, value1);
        ##printf("x: %.10f, y: %.10f, xInt: %d, yInt: %d, xR: %.3f, yR: %.3fm in:%.2f, out:%.2f\\n", x,y,xInt,yInt,xRem,yRem,value,ivalue);
        
        }
        </%def>
        """)

        plan.kernel_call(
                template.get_def('interp2dnearest'),
                [output, inputArray, xCoords, yCoords],
                global_size=(output.shape[0], output.shape[1]),
                render_kwds={
                    'mul' : cluda.functions.mul(
                                "float32", "float32")
                    }
                )
        print("outputShape: {}".format(output.shape))
        return plan
        
        
def test_interp2dNearest(data, newSize, thr=None):
    if not thr:
        api = cluda.ocl_api()
        thr = api.Thread.create()

    if len(newSize)!=2 or len(data.shape)!=2:
        raise ValueError("2D Arrays only!")

    output_gpu = thr.to_device(numpy.zeros(newSize).astype(data.dtype))
    data_gpu = thr.to_device(data)

    xCoords = numpy.linspace(0, data.shape[0]-1, newSize[0]).astype("float32")
    yCoords = numpy.linspace(0, data.shape[1]-1, newSize[1]).astype("float32")
    xCoords[-1] -= 1e-3
    yCoords[-1] -= 1e-3

    xCoords_gpu = thr.to_device(xCoords)
    yCoords_gpu = thr.to_device(yCoords)

    
    print("xCoords:{}".format(xCoords))
    print("yCoords:{}".format(yCoords))
    print(data_gpu.get())

    interp = Interp2dNearestGPU(output_gpu, data_gpu, xCoords_gpu)
    c_interp = interp.compile(thr)

    c_interp(output_gpu, data_gpu, xCoords_gpu, yCoords_gpu)

    return output_gpu.get()
    

class Interp1d_2dGPU(Computation):
    
    def __init__(self, outputArr, inputArr, coordArr):
        
        Computation.__init__(self, [
            Parameter('output', Annotation(outputArr, 'o')),
            Parameter('inputArray', Annotation(inputArr, 'i')),
            Parameter('xCoords', Annotation(coordArr, 'i')),
            ])

    def _build_plan(   self, plan_factory, device_params,
                        output, inputArray, xCoords):
        plan = plan_factory()

        template = reikna.helpers.template_from(
        """
        <%def name='interp1d(kernel_decleration, k_output, k_input, k_xCoords)'>
        ${kernel_decleration}
        {
        VIRTUAL_SKIP_THREADS;
        const VSIZE_T idx = virtual_global_id(0);
        const VSIZE_T idy = virtual_global_id(1);
        
        const ${k_xCoords.ctype} x = ${k_xCoords.load_idx}(idx);
        
        const int xInt = (int) x;

        const float xRemainder = (float) x - (float) xInt;
        
        const float grad = (float) ${k_input.load_idx}(xInt+1, idy) - (float) ${k_input.load_idx}(xInt, idy);
        
        const ${k_output.ctype} value = (${k_output.ctype}) ${mul}(grad, xRemainder) + ${k_input.load_idx}(xInt, idy);

        ${k_output.store_idx}(idx, idy, value);
        
        printf("i: %d, x: %.4f, xInt: %d, xRem: %.4f, value:%.4f\\n", idx, x, xInt, xRemainder, value);
        }
        </%def>
        """)

        plan.kernel_call(
                template.get_def('interp1d'),
                [output, inputArray, xCoords],
                global_size=output.shape,
                render_kwds={
                    'mul' : cluda.functions.mul(
                                "float32", "float32")
                    }
                )
        print("outputShape:{}".format(output.shape))
        print("inputShape:{}".format(inputArray.shape))
        return plan

def test_interp1d_2d(data, newSize, thr=None):
    if not thr:
        api = cluda.ocl_api()
        thr = api.Thread.create()

    output_gpu = thr.to_device(
                numpy.zeros((newSize, data.shape[1])).astype(data.dtype))
    data_gpu = thr.to_device(data)

    coords = numpy.linspace(0,data.shape[0]-1,newSize)
    coords[-1] -= 1e-5
    
    print("coords:{}".format(coords))
    coords_gpu = thr.to_device(coords)

    interp = Interp1d_2dGPU(output_gpu, data_gpu, coords)
    c_interp = interp.compile(thr)

    c_interp(output_gpu, data_gpu, coords_gpu)

    return output_gpu.get()
    

class DataTest2d(Computation):
    
    def __init__(self, outputArr, inputArr):
        
        Computation.__init__(self, [
            Parameter('output', Annotation(outputArr, 'o')),
            Parameter('inputArray', Annotation(inputArr, 'i')),
            ])

    def _build_plan(   self, plan_factory, device_params,
                        output, inputArray):
        plan = plan_factory()

        template = reikna.helpers.template_from(
        """
        <%def name='test2d(kernel_decleration, k_output, k_input)'>
        ${kernel_decleration}
        {
        VIRTUAL_SKIP_THREADS;
        const VSIZE_T idx = virtual_global_id(0);
        const VSIZE_T idy = virtual_global_id(1);
        
        const ${k_output.ctype} value = ${k_input.load_idx}(idx, idy);
        ${k_output.store_idx}(idx, idy, value);
        
        printf("idx: %d, idy: %d, value:%.4f\\n", idx, idy, value);
        }
        </%def>
        """)

        plan.kernel_call(
                template.get_def('test2d'),
                [output, inputArray],
                global_size= output.shape,
                render_kwds={
                    }
                    )
        print(output.shape)
        return plan
        
def testData2d(data, thr=None):
    
    if not thr:
        api = cluda.ocl_api()
        thr = api.Thread.create()
    
    inputData = thr.to_device(data)
    outputData = thr.to_device(numpy.zeros(data.shape).astype("float32"))
    
    dataTest = DataTest2d(outputData, inputData)
    c_dataTest = dataTest.compile(thr)
    
    c_dataTest(outputData, inputData)
    
    return outputData.get()