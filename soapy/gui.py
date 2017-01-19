#Copyright Durham University and Andrew Reeves
#2014

# This file is part of soapy.

#     soapy is free software: you can redistribute it and/or modify
#     it under the terms of the GNU General Public License as published by
#     the Free Software Foundation, either version 3 of the License, or
#     (at your option) any later version.

#     soapy is distributed in the hope that it will be useful,
#     but WITHOUT ANY WARRANTY; without even the implied warranty of
#     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#     GNU General Public License for more details.

#     You should have received a copy of the GNU General Public License
#     along with soapy.  If not, see <http://www.gnu.org/licenses/>.

"""
The GUI for the Soapy adaptive optics simulation
"""

import sys

try:
    from PyQt5 import QtGui, QtWidgets, QtCore
    PYQT_VERSION = 5
except (ImportError ,RuntimeError):
    from PyQt4 import QtGui, QtCore
    QtWidgets = QtGui
    PYQT_VERSION = 4

# Attempt to import PyQt5, if not try PyQt4

# Do this so uses new Jupyter console if available
try:
    from qtconsole.rich_jupyter_widget import RichJupyterWidget as RichIPythonWidget
    from qtconsole.inprocess import QtInProcessKernelManager
except ImportError:
    from IPython.qt.console.rich_ipython_widget import RichIPythonWidget
    from IPython.qt.inprocess import QtInProcessKernelManager

from IPython.lib import guisupport

if PYQT_VERSION == 5:
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
elif PYQT_VERSION == 4:
    from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas

from matplotlib.figure import Figure

from . import pyqtgraph

# Change pyqtgraph colourmaps to more usual ones
# set default colourmaps available
pyqtgraph.graphicsItems.GradientEditorItem.Gradients = pyqtgraph.pgcollections.OrderedDict([
    ('viridis', {'ticks': [(0.,  ( 68,   1,  84, 255)),
                           (0.2, ( 65,  66, 134, 255)),
                           (0.4, ( 42, 118, 142, 255)),
                           (0.6, ( 32, 165, 133, 255)),
                           (0.8, (112, 206,  86, 255)),
                           (1.0, (241, 229,  28, 255))], 'mode':'rgb'}),
    ('coolwarm', {'ticks': [(0.0, ( 59,  76, 192)),
                            (0.5, (220, 220, 220)),
                            (1.0, (180, 4, 38))], 'mode': 'rgb'}),
    ('grey', {'ticks': [(0.0, (0, 0, 0, 255)),
                        (1.0, (255, 255, 255, 255))], 'mode': 'rgb'}),
    ('magma', {'ticks':[(0., (0, 0, 3, 255)),
                        (0.25, (80, 18, 123, 255)),
                        (0.5, (182,  54, 121, 255)),
                        (0.75, (251, 136,  97, 255)),
                        (1.0, (251, 252, 191))], 'mode':'rgb'})
        ])


if PYQT_VERSION == 5:
    from .aogui_ui5 import Ui_MainWindow
elif PYQT_VERSION == 4:
    from .aogui_ui4 import Ui_MainWindow

from . import logger

import sys
import numpy
import time
import json
import traceback
from functools import partial
#Python2/3  compatibility
try:
    import queue
except ImportError:
    import Queue as queue
try:
    xrange
except NameError:
    xrange = range


from argparse import ArgumentParser
import pylab
import os
try:
    from OpenGL import GL
except ImportError:
    GL = False


guiFile_path = os.path.abspath(os.path.realpath(__file__)+"/..")

#This is the colormap to be used in all pyqtgraph plots
#It can be changed in the GUI using the gradient slider in the top left
#to get the LUT dictionary, use ``gui.gradient.saveState()''
CMAP={'mode': 'rgb',
 'ticks': [ (0., (14, 66, 255, 255)),
            (0.5, (255, 255, 255, 255)),
            (1., (255, 26, 26, 255))]}


# Must overwrite sys.excepthook to aviod crash on exception
def execpthook(etype, value, tb):
    traceback.print_exception(etype, value, tb)

sys.excepthook = execpthook

class GUI(QtWidgets.QMainWindow):
    def __init__(self, sim, useOpenGL=False, verbosity=None):
        QtWidgets.QMainWindow.__init__(self)

        # get current application instance
        self.app = QtCore.QCoreApplication.instance()

        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        self.useOpenGL = useOpenGL

        self.ui.runButton.clicked.connect(self.run)
        self.ui.initButton.clicked.connect(self.init)
        self.ui.iMatButton.clicked.connect(self.iMat)
        self.ui.stopButton.clicked.connect(self.stop)

        self.ui.reloadParamsAction.triggered.connect(self.read)
        self.ui.loadParamsAction.triggered.connect(self.readParamFile)

        #Ensure update is called if sci button pressed
        self.ui.instExpRadio.clicked.connect(self.update)
        self.ui.longExpRadio.clicked.connect(self.update)

        #Initialise Colour chooser
        self.gradient = pyqtgraph.GradientWidget(orientation="bottom")
        self.gradient.sigGradientChanged.connect(self.changeLUT)
        self.ui.verticalLayout.addWidget(self.gradient)
        self.gradient.restoreState(CMAP)


        self.sim = sim

        self.wfsPlots = {}
        self.dmPlots = {}
        self.sciPlots = {}
        self.resPlots = {}

        self.console = IPythonConsole(self.ui.consoleLayout,self.sim,self)

        self.loopRunning=False
        self.makingIMat=False

        #Init Timer to update plots
        self.updateTimer = QtCore.QTimer()
        self.updateTimer.setInterval(100)
        self.updateTimer.timeout.connect(self.update)
        self.ui.updateTimeSpin.valueChanged.connect(self.updateTimeChanged)
        self.updateQueue = queue.Queue(10)
        self.updateLock = QtCore.QMutex()

        #Placeholders for sim threads
        self.initThread = None
        self.iMatThread = None
        self.loopThread = None

        self.resultPlot = None

        #Required for plotting colors
        self.colorList = ["b","g","r","c","m","y","k"]
        self.colorNo = 0

        self.resultPlot = PlotWidget()
        self.ui.plotLayout.addWidget(self.resultPlot)

        #sim.readParams()
        sim.config.calcParams()
        self.config = self.sim.config
        if verbosity is not None:
            self.config.sim.verbosity = verbosity
        self.initPlots()
        self.show()
        self.init()

        self.console.write("Running %s\n"%self.sim.configFile)

    def moveEvent(self, event):
        """
        Overwrite PyQt Move event to force a repaint. (Might) fix a bug on some (my) macs
        """
        self.repaint()
        super(GUI, self).moveEvent(event)

####################################
#Load Param file methods
    def readParamFile(self):

        fname = QtGui.QFileDialog.getOpenFileName(self, 'Open file', '.')

        if PYQT_VERSION == 5:
            fname = fname[0]
        
        fname = str(fname)

        if fname is not "":
            self.sim.readParams(fname)
            self.config = self.sim.config
            self.initPlots()

################################################################
#Plot Methods

    def initPlots(self):

        self.ui.progressBar.setValue(80)
        for layout in [ self.ui.wfsLayout, self.ui.dmLayout,
                        self.ui.residualLayout, self.ui.sciLayout,
                        self.ui.phaseLayout, self.ui.lgsLayout,
                        self.ui.gainLayout]:
            for i in reversed(range(layout.count())):
                layout.itemAt(i).widget().setParent(None)


        self.wfsPlots = {}
        self.lgsPlots = {}
        self.phasePlots = {}
        for wfs in range(self.config.sim.nGS):
            self.wfsPlots[wfs] = self.makeImageItem(
                    self.ui.wfsLayout,
                    self.config.wfss[wfs].nxSubaps*self.config.wfss[wfs].pxlsPerSubap
                    )
            self.phasePlots[wfs] = self.makeImageItem(
                    self.ui.phaseLayout, self.config.sim.simSize)

            if ((self.config.wfss[wfs].lgs is not None) and (self.config.wfss[wfs].lgs.uplink == 1)):
                self.lgsPlots[wfs] = self.makeImageItem(
                        self.ui.lgsLayout, self.config.sim.pupilSize)

        self.dmPlots = {}
        for dm in range(self.config.sim.nDM):
            self.dmPlots[dm] = self.makeImageItem(self.ui.dmLayout,
                                                  self.config.sim.simSize)

        self.sciPlots = {}
        self.resPlots = {}
        for sci in range(self.config.sim.nSci):

            self.sciPlots[sci] = self.makeImageItem(self.ui.sciLayout,
                                                    self.config.scis[sci].pxls)
            self.resPlots[sci] = self.makeImageItem(self.ui.residualLayout,
                                                    self.config.sim.simSize)
        self.sim.guiQueue = self.updateQueue
        self.sim.guiLock = self.updateLock
        self.sim.gui = True
        self.sim.waitingPlot = False

        #Set initial gains
        self.gainSpins = []
        for dm in range(self.config.sim.nDM):
            gainLabel = QtGui.QLabel()
            gainLabel.setText("DM {}:".format(dm))
            self.ui.gainLayout.addWidget(gainLabel)

            self.gainSpins.append(QtGui.QDoubleSpinBox())
            self.ui.gainLayout.addWidget(self.gainSpins[dm])
            self.gainSpins[dm].setValue(self.config.dms[dm].gain)
            self.gainSpins[dm].setSingleStep(0.05)
            self.gainSpins[dm].setMaximum(1.)

            self.gainSpins[dm].valueChanged.connect(
                                                partial(self.gainChanged,dm))

        self.ui.progressBar.setValue(100)
        self.statsThread = StatsThread(self.sim)

    def update(self):

        #tell sim that gui wants a plot
        self.sim.waitingPlot = True
        #empty queue so only latest update is present
        plotDict = None
        self.updateLock.lock()
        try:
            while not self.updateQueue.empty():
                plotDict = self.updateQueue.get_nowait()
        except:
            self.updateLock.unlock()
            traceback.print_exc()
        self.updateLock.unlock()

        if plotDict:

            # Get the min and max plot scaling
            scaleValues = self.getPlotScaling(plotDict)

            for wfs in range(self.config.sim.nGS):
                if numpy.any(plotDict["wfsFocalPlane"][wfs])!=None:
                    wfsFP = plotDict['wfsFocalPlane'][wfs]
                    self.wfsPlots[wfs].setImage(wfsFP, lut=self.LUT)
                    # self.wfsPlots[wfs].getViewBox().setRange(
                    #         QtCore.QRectF(0, 0, wfsFP.shape[0],
                    #         wfsFP.shape[1])
                    #         )

                if numpy.any(plotDict["wfsPhase"][wfs])!=None:
                    wfsPhase = plotDict["wfsPhase"][wfs]
                    self.phasePlots[wfs].setImage(
                            wfsPhase, lut=self.LUT, levels=scaleValues)
                    self.phasePlots[wfs].getViewBox().setRange(
                            QtCore.QRectF(0, 0, wfsPhase.shape[0], wfsPhase.shape[1]))

                if numpy.any(plotDict["lgsPsf"][wfs])!=None:
                    self.lgsPlots[wfs].setImage(
                        plotDict["lgsPsf"][wfs], lut=self.LUT)


            if numpy.any(plotDict["ttShape"])!=None:
                self.ttPlot.setImage(plotDict["ttShape"], lut=self.LUT)

            for dm in range(self.config.sim.nDM):
                if numpy.any(plotDict["dmShape"][dm]) !=None:
                    dmShape = plotDict["dmShape"][dm]
                    self.dmPlots[dm].setImage(plotDict["dmShape"][dm],
                                            lut=self.LUT, levels=scaleValues)

            for sci in range(self.config.sim.nSci):
                if numpy.any(plotDict["sciImg"][sci])!=None:
                    if self.ui.instExpRadio.isChecked():
                        self.sciPlots[sci].setImage(
                                plotDict["instSciImg"][sci], lut=self.LUT)
                    elif self.ui.longExpRadio.isChecked():
                        self.sciPlots[sci].setImage(
                                plotDict["sciImg"][sci], lut=self.LUT)

                if numpy.any(plotDict["residual"][sci])!=None:
                    residual = plotDict["residual"][sci]

                    self.resPlots[sci].setImage(
                            residual, lut=self.LUT, levels=scaleValues)

            if self.loopRunning:
                self.updateStrehls()

            self.app.processEvents()

    def getPlotScaling(self, plotDict):
        """
        Loops through all phase plots to find the required min and max values for plot scaling
        """
        plotMins = []
        plotMaxs = []
        for wfs in range(self.config.sim.nGS):
            if numpy.any(plotDict["wfsPhase"])!=None:
                plotMins.append(plotDict["wfsPhase"][wfs].min())
                plotMaxs.append(plotDict["wfsPhase"][wfs].max())

        for dm in range(self.config.sim.nDM):
            if numpy.any(plotDict["dmShape"][dm])!=None:
                plotMins.append(plotDict["dmShape"][dm].min())
                plotMaxs.append(plotDict["dmShape"][dm].max())

        for sci in range(self.config.sim.nSci):
            if numpy.any(plotDict["residual"][sci])!=None:
                plotMins.append(plotDict["residual"][sci].min())
                plotMaxs.append(plotDict["residual"][sci].max())

        # Now get the min and max of mins and maxs
        plotMin = min(plotMins)
        plotMax = max(plotMaxs)

        return plotMin, plotMax


    def makeImageItem(self, layout, size):
        gv = pyqtgraph.GraphicsView()
        if self.useOpenGL and GL:
            gv.useOpenGL()
        layout.addWidget(gv)
        vb = pyqtgraph.ViewBox()
        vb.setAspectLocked(True)
        vb.enableAutoRange(axis=pyqtgraph.ViewBox.XYAxes, enable=True)

        gv.setCentralItem(vb)
        img = pyqtgraph.ImageItem(border="w")
        vb.addItem(img)
        vb.setRange(QtCore.QRectF(0, 0, size, size))
        return img

    def plotPupilOverlap(self):

        if self.resultPlot:
            self.resultPlot.setParent(None)
        scrnNo = self.sim.config.atmos.scrnNo
        self.resultPlot = OverlapWidget(scrnNo)
        self.ui.plotLayout.addWidget(self.resultPlot)

        for i in range(scrnNo):

            self.resultPlot.canvas.axes[i].imshow(
                    numpy.zeros((   self.config.sim.pupilSize*2,
                                    self.config.sim.pupilSize*2)),
                                        origin="lower")
            for wfs in range(self.config.sim.nGS):
                if self.sim.config.wfss[wfs].GSHeight>self.sim.config.atmos.scrnHeights[i] or self.sim.config.wfss[wfs].GSHeight==0:
                    cent = (self.sim.wfss[wfs].los.getMetaPupilPos(
                            self.sim.config.atmos.scrnHeights[i])
                            *self.sim.config.sim.pxlScale
                            +self.config.sim.pupilSize)

                    if self.sim.wfss[wfs].radii!=None:
                        radius = self.sim.wfss[wfs].radii[i]

                    else:
                        radius = self.config.sim.pupilSize/2.

                    if self.sim.config.wfss[wfs].GSHeight!=0:
                        colour="r"
                    else:
                        colour="g"

                    circ = pylab.Circle(cent,radius=radius,alpha=0.2, fc=colour)
                    self.resultPlot.canvas.axes[i].add_patch(circ)
                    self.resultPlot.canvas.axes[i].set_yticks([])
                    self.resultPlot.canvas.axes[i].set_xticks([])

            for sci in range(self.config.sim.nSci):
                cent = self.sim.sciCams[sci].los.getMetaPupilPos(
                        self.sim.config.atmos.scrnHeights[i])
                cent*=self.sim.config.sim.pxlScale
                cent+=self.config.sim.pupilSize

                radius = self.config.sim.pupilSize/2.

                circ = pylab.Circle(cent, radius=radius, alpha=0.2, fc="y")
                self.resultPlot.canvas.axes[i].add_patch(circ)
                self.resultPlot.canvas.axes[i].set_yticks([])
                self.resultPlot.canvas.axes[i].set_xticks([])


    def initStrehlPlot(self):
        #init plot
        if self.resultPlot:
            self.resultPlot.setParent(None)
        self.resultPlot = PlotWidget()
        self.ui.plotLayout.addWidget(self.resultPlot)

        self.strehlAxes = self.resultPlot.canvas.ax
        self.strehlAxes.set_xlabel("Iterations",fontsize="xx-small")
        self.strehlAxes.set_ylabel("Strehl Ratio",fontsize="xx-small")
        self.strehlAxes.set_ylim(0, 1.)
        self.strehlAxes.tick_params(axis='both', which='major', labelsize="xx-small")
        self.strehlAxes.tick_params(axis='both', which='minor', labelsize="xx-small")
        self.strehlPlts=[]

        self.colorNo+=1
        if self.colorNo==len(self.colorList):
            self.colorNo=0

    def updateStrehls(self):

        instStrehls = []
        longStrehls = []

        for i in range(self.config.sim.nSci):
            instStrehls.append(100*self.sim.sciCams[i].instStrehl)
            longStrehls.append(100*self.sim.sciCams[i].longExpStrehl)

        self.ui.instStrehl.setText( "Instantaneous Strehl: "
           +self.config.sim.nSci*"%.1f%%  "%tuple(instStrehls))
        self.ui.longStrehl.setText("Long Exposure Strehl: "
           +self.config.sim.nSci*"%.1f%%  "% tuple(longStrehls))

        for plt in self.strehlPlts:
            for line in plt:
                line.remove()
            del plt

        self.strehlPlts=[]
        for sci in xrange(self.config.sim.nSci):
            self.strehlPlts.append(self.strehlAxes.plot(self.sim.instStrehl[sci],
                    linestyle=":", color=self.colorList[(self.colorNo+sci) % len(self.colorList)]))
            self.strehlPlts.append(self.strehlAxes.plot(self.sim.longStrehl[sci],
                 color=self.colorList[(self.colorNo+sci) % len(self.colorList)]))
        self.resultPlot.canvas.draw()

    def updateStats(self, itersPerSec, timeRemaining):

        self.ui.itersPerSecLabel.setText(
                                "Iterations Per Second: %.2f"%(itersPerSec))
        self.ui.timeRemaining.setText( "Time Remaining: %.2fs"%(timeRemaining) )

########################################################
#Sim Call Backs
    def read(self):
        self.sim.readParams()

    def init(self):
        self.ui.progressLabel.setText("Initialising...")
        self.ui.progressBar.setValue(2)

        self.iThread = InitThread(self)
        self.iThread.updateProgressSignal.connect(self.progressUpdate)
        self.iThread.finished.connect(self.initPlots)
        self.iThread.start()
        self.config = self.sim.config

    def iMat(self):

        if self.iMatThread!=None:
            running = self.iMatThread.isRunning()
        else:
            running = False

        if running == False:

            self.plotPupilOverlap()
            print("making IMat")
            self.ui.progressLabel.setText("Generating DM Shapes...")
            self.ui.progressBar.setValue(10)
            self.updateTimer.start()
            self.iMatThread = IMatThread(self)
            self.iMatThread.updateProgressSignal.connect(self.progressUpdate)
            self.iMatThread.start()


    def run(self):

        self.initStrehlPlot()

        self.startTime = time.time()
        self.ui.progressLabel.setText("Running AO Loop")
        self.ui.progressBar.setValue(0)
        self.loopThread = LoopThread(self)
        self.loopThread.updateProgressSignal.connect(self.progressUpdate)
        self.statsThread.updateStatsSignal.connect(self.updateStats)
        self.loopThread.start()

        self.updateTimer.start()
        self.statsThread.start()

    def stop(self):
        self.sim.go=False
        try:
            self.loopThread.quit()
        except AttributeError:
            pass

        try:
            self.iMatThread.quit()
        except AttributeError:
            pass

        try:
            self.statsThread.quit()
        except AttributeError:
            pass

        self.updateTimer.stop()
#####################################################


###########################################
#Misc GUI Callbacks

    def changeLUT(self):
        self.LUT = self.gradient.getLookupTable(256)

    def gainChanged(self, dm):
        self.config.dms[dm].gain = self.gainSpins[dm].value()

    def updateTimeChanged(self):

        try:
            self.updateTime = int(numpy.round(1000./float(self.ui.updateTimeSpin.value())))
            self.updateTimer.setInterval(self.updateTime)
        except ZeroDivisionError:
            pass

    def progressUpdate(self, message, i="", maxIter=""):

        if i!="" and maxIter!="":
            percent = int(round(100*(float(i)/float(maxIter))))
            self.ui.progressBar.setValue(percent)
            self.ui.progressLabel.setText(
                    "{0}: Iteration {1} of {2}".format(message, i, maxIter))

        else:
            if i!="":
                message+=" {}".format(i)
            self.ui.progressLabel.setText(message)


###############################################

#Tidy up before closing the gui
    #
    # def closeEvent(self, event):
    #     del(self.app)


###########################################


class StatsThread(QtCore.QThread):
    updateStatsSignal = QtCore.pyqtSignal(float,float)
    def __init__(self, sim):
        QtCore.QThread.__init__(self)

        self.sim = sim

    def run(self):
        self.startTime = time.time()

        while self.sim.iters+1 < self.sim.config.sim.nIters and self.sim.go:
            time.sleep(0.4)
            iTime = time.time()
            try:
                #Calculate and print running stats
                itersPerSec = self.sim.iters / (iTime - self.startTime)
                timeRemaining = (self.sim.config.sim.nIters-self.sim.iters)/itersPerSec
                self.updateStatsSignal.emit(itersPerSec, timeRemaining)
            except ZeroDivisionError:
                pass


class InitThread(QtCore.QThread):
    updateProgressSignal = QtCore.pyqtSignal(str,str,str)
    def __init__(self,guiObj):
        QtCore.QThread.__init__(self)
        self.guiObj = guiObj
        self.sim = guiObj.sim

    def run(self):
        logger.setStatusFunc(self.progressUpdate)
        if self.sim.go:
            self.guiObj.stop()

        self.sim.aoinit()

    def progressUpdate(self, message, i="", maxIter=""):
        self.updateProgressSignal.emit(str(message), str(i), str(maxIter))


class IMatThread(QtCore.QThread):
    updateProgressSignal = QtCore.pyqtSignal(str,str,str)

    def __init__(self,guiObj):
        self.sim = guiObj.sim
        self.guiObj = guiObj
        QtCore.QThread.__init__(self)

    def run(self):
        print("initing..")
        self.guiObj.makingIMat=True
        logger.setStatusFunc(self.progressUpdate)
        try:
            self.sim.makeIMat(forceNew=self.guiObj.ui.newCMat.isChecked(),
                                    progressCallback=self.progressUpdate)
            self.guiObj.makingIMat=False
            self.guiObj.stop()
        except:
            self.guiObj.makingIMat=False
            self.guiObj.stop()
            traceback.print_exc()

    def progressUpdate(self, message, i="", maxIter=""):
        i = str(i)
        maxIter = str(maxIter)
        message = str(message)
        self.updateProgressSignal.emit(message, i, maxIter)


class LoopThread(QtCore.QThread):
    updateProgressSignal = QtCore.pyqtSignal(str,str,str)

    def __init__(self, guiObj):

        QtCore.QThread.__init__(self)
        #multiprocessing.Process.__init__(self)
        self.guiObj=guiObj

        self.sim = guiObj.sim

    def run(self):
        logger.setStatusFunc(self.progressUpdate)
        try:
            self.guiObj.loopRunning=True
            self.sim.aoloop()
            self.guiObj.loopRunning=False
            self.guiObj.stop()
        except:
            self.sim.go = False
            self.guiObj.loopRunning = False
            self.guiObj.stop()
            traceback.print_exc()


    def progressUpdate(self, message, i="", maxIter=""):

        self.updateProgressSignal.emit(str(message), str(i), str(maxIter))


class IPythonConsole:
    def __init__(self, layout, sim, gui):
        # Create an in-process kernel
        # >>> print_process_id()
        # will print the same process ID as the main process
        self.kernel_manager = QtInProcessKernelManager()
        self.kernel_manager.start_kernel()
        self.kernel = self.kernel_manager.kernel
        self.kernel.gui = 'qt4'

        self.kernel.shell.write("Welcome to AO Sim!")

        config = sim.config
        #Pass some useful objects to the user
        usefulObjects = {    "sim" : sim,
                            "gui" : gui,
                            "config" : config,
                            "simConfig" : sim.config.sim,
                            "telConfig" : sim.config.tel,
                            "atmosConfig" : sim.config.atmos}

        for i in range(sim.config.sim.nGS):
            usefulObjects["wfs{}Config".format(i)] = sim.config.wfss[i]
        for i in range(sim.config.sim.nDM):
            usefulObjects["dm{}Config".format(i)] = sim.config.dms[i]
        for i in range(sim.config.sim.nSci):
            usefulObjects["sci{}Config".format(i)] = sim.config.scis[i]

        self.kernel.shell.push(usefulObjects)
        #kernel.shell.push({'foo': 43, 'print_process_id': print_process_id})

        self.kernel_client = self.kernel_manager.client()
        self.kernel_client.start_channels()


        control = RichIPythonWidget()
        control.kernel_manager = self.kernel_manager
        control.kernel_client = self.kernel_client
        control.exit_requested.connect(self.stop)
        layout.addWidget(control)

        self.kernel.shell.ex("")
        #control.show()

        #self.kernel.show
    def stop(self):
        self.kernel_client.stop_channels()
        self.kernel_manager.shutdown_kernel()

    def write(self,message):
        self.kernel.shell.write(message)
        self.kernel.shell.ex("")

class OverlapCanvas(FigureCanvas):
    def __init__(self, nAxes):
        self.fig = Figure(facecolor="white", frameon=False)
        self.axes=[]
        for i in range(nAxes):
            self.axes.append(self.fig.add_subplot(2, numpy.ceil(nAxes/2.),i+1))

        FigureCanvas.__init__(self, self.fig)
        FigureCanvas.setSizePolicy(self, QtGui.QSizePolicy.Expanding,QtGui.QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)

class OverlapWidget(QtGui.QWidget):

    def __init__(self, nAxes, parent = None):
        QtGui.QWidget.__init__(self, parent)
        self.canvas = OverlapCanvas(nAxes)
        self.vbl = QtGui.QVBoxLayout()
        self.vbl.addWidget(self.canvas)
        self.setLayout(self.vbl)


class PlotCanvas(FigureCanvas):

    def __init__(self):
        self.fig = Figure(facecolor="white", frameon=False)
        self.ax = self.fig.add_subplot(111)

        FigureCanvas.__init__(self, self.fig)
        FigureCanvas.setSizePolicy(self, QtGui.QSizePolicy.Expanding,QtGui.QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)


class PlotWidget(QtGui.QWidget):

    def __init__(self, parent = None):
        QtGui.QWidget.__init__(self, parent)
        self.canvas = PlotCanvas()
        self.vbl = QtGui.QVBoxLayout()
        self.vbl.addWidget(self.canvas)
        self.setLayout(self.vbl)


def start_gui(simulation, useOpenGL=False, verbosity=1):
    app = QtWidgets.QApplication([])

    gui = GUI(simulation, useOpenGL=useOpenGL, verbosity=verbosity)

    app.exec_()
    del(gui.initThread)
    del(gui.iMatThread)
    del(gui.loopThread)
    del(gui.console)
    del(gui)
    # sys.exit()

if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("configFile",nargs="?",action="store")
    parser.add_argument("-gl",action="store_true")
    args = parser.parse_args()

    if args.configFile != None:
        confFile = args.configFile
    else:
        confFile = "conf/testConf.py"


    G = GUI(confFile,useOpenGL=args.gl)

