from soapy import numbalib
import numpy



def test_zoomtoefield():

    input_data = numpy.arange(100).reshape(10,10).astype("float32")

    output_data = numpy.zeros((100, 100), dtype="float32")
    output_efield2 = numpy.zeros((100, 100), dtype="complex64")

    numbalib.wfs.zoom(input_data, output_data)

    output_efield1 = numpy.exp(1j * output_data)

    numbalib.wfs.zoomtoefield(input_data, output_efield2)

    assert numpy.allclose(output_efield1, output_efield2)


if __name__ == "__main__":
    test_zoomtoefield()