#! /usr/bin/env python

import os
os.environ["QT_API"]="pyqt"
from IPython.qt.console.rich_ipython_widget import RichIPythonWidget
from IPython.qt.inprocess import QtInProcessKernelManager
from IPython.lib import guisupport

from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
 

from PyQt4 import QtGui,QtCore
import pyqtgraph
from .AOGUIui import Ui_MainWindow
from . import logger

import sys
import numpy
import time
import json
import traceback
from functools import partial
#Python2/3 queue compatibility
try:
    import queue
except ImportError:
    import Queue as queue
    
from argparse import ArgumentParser
import pylab
import os
try:
    from OpenGL import GL
except ImportError:
    pass


guiFile_path = os.path.abspath(os.path.realpath(__file__)+"/..")

#This is the colormap to be used in all pyqtgraph plots
#It can be changed in the GUI using the gradient slider in the top left
#to get the LUT dictionary, use ``gui.gradient.saveState()''
CMAP={'mode': 'rgb',
 'ticks': [(0.31533696306403636, (27, 222, 222, 255)),
  (0.8933823529411765, (199, 50, 13, 255)),
  (0.733865287190744, (209, 191, 55, 255)),
  (0.007515187841599431, (15, 6, 143, 255)),
  (0.4424019607843137, (138, 240, 109, 255)),
  (0.9959871589085072, (255, 5, 5, 255)),
  (0.5460955536634461, (252, 255, 32, 255))]}

class GUI(QtGui.QMainWindow):
    def __init__(self,sim,useOpenGL=False):
        self.app = QtGui.QApplication([])
        QtGui.QMainWindow.__init__(self)

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

        sim.readParams()
        self.config = self.sim.config
        self.initPlots()
        self.show()
        self.init()
    
        self.console.write("Running %s\n"%self.sim.configFile)
        sys.exit(self.app.exec_())


####################################
#Load Param file methods
    def readParamFile(self):

        fname = QtGui.QFileDialog.getOpenFileName(self, 'Open file', 
                '/home')
    
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
                    self.config.wfs[wfs].subaps*self.config.wfs[wfs].pxlsPerSubap
                    )
            self.phasePlots[wfs] = self.makeImageItem(self.ui.phaseLayout,self.config.sim.pupilSize)
                                                      
            if self.config.lgs[wfs].lgsUplink == 1:
                self.lgsPlots[wfs] = self.makeImageItem(
                        self.ui.lgsLayout, self.config.sim.pupilSize)


        if self.config.sim.tipTilt:
            self.ttPlot = self.makeImageItem(self.ui.dmLayout,
                                            self.config.sim.pupilSize)
            
        self.dmPlots = {}
        for dm in range(self.config.sim.nDM):
            self.dmPlots[dm] = self.makeImageItem(self.ui.dmLayout,
                                                  self.config.sim.pupilSize)

        self.sciPlots = {}
        self.resPlots = {}
        for sci in range(self.config.sim.nSci):

            self.sciPlots[sci] = self.makeImageItem(self.ui.sciLayout,
                                                    self.config.sci[sci].pxls)
            self.resPlots[sci] = self.makeImageItem(self.ui.residualLayout,
                                                    self.config.sim.pupilSize)
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
            self.gainSpins[dm].setValue(self.config.dm[dm].gain)
            self.gainSpins[dm].setSingleStep(0.05)
            self.gainSpins[dm].setMaximum(1.)

            self.gainSpins[dm].valueChanged.connect(
                                                partial(self.gainChanged,dm))

        self.ui.progressBar.setValue( 100)
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
            for wfs in range(self.config.sim.nGS):
                if numpy.any(plotDict["wfsFocalPlane"][wfs])!=None:
                    self.wfsPlots[wfs].setImage(
                        plotDict["wfsFocalPlane"][wfs], lut=self.LUT)
                        
                if numpy.any(plotDict["wfsPhase"][wfs])!=None:
                    self.phasePlots[wfs].setImage(
                        plotDict["wfsPhase"][wfs], lut=self.LUT)
                        
                if numpy.any(plotDict["lgsPsf"][wfs])!=None:
                    self.lgsPlots[wfs].setImage(
                        plotDict["lgsPsf"][wfs], lut=self.LUT)
                    
        
            if numpy.any(plotDict["ttShape"])!=None:
                self.ttPlot.setImage(plotDict["ttShape"], lut=self.LUT)
            
            for dm in range(self.config.sim.nDM):
                if numpy.any(plotDict["dmShape"][dm]) !=None:
                    self.dmPlots[dm].setImage(plotDict["dmShape"][dm],
                                            lut=self.LUT)
           
            for sci in range(self.config.sim.nSci):
                if numpy.any(plotDict["sciImg"][sci])!=None:
                    if self.ui.instExpRadio.isChecked():
                        self.sciPlots[sci].setImage(
                                plotDict["instSciImg"][sci], lut=self.LUT)
                    elif self.ui.longExpRadio.isChecked():
                        self.sciPlots[sci].setImage(
                                plotDict["sciImg"][sci], lut=self.LUT)
                    
                if numpy.any(plotDict["residual"][sci])!=None:
                    self.resPlots[sci].setImage(
                                plotDict["residual"][sci], lut=self.LUT)
            
            if self.loopRunning:
                self.updateStrehls()
            
            self.app.processEvents()

    def makeImageItem(self,layout,size):
        gv = pyqtgraph.GraphicsView()
        if self.useOpenGL and GL:
            gv.useOpenGL()
        layout.addWidget(gv)
        vb = pyqtgraph.ViewBox()
        vb.setAspectLocked(True)
        vb.enableAutoRange(axis = pyqtgraph.ViewBox.XYAxes,enable=True)

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

            self.resultPlot.canvas.axes[i].imshow(numpy.zeros((   self.config.sim.pupilSize*2,
                                                self.config.sim.pupilSize*2)),
                                        origin="lower")
            for wfs in range(self.config.sim.nGS):
                if self.sim.config.wfs[wfs].GSHeight>self.sim.config.atmos.scrnHeights[i] or self.sim.config.wfs[wfs].GSHeight==0:
                    cent = self.sim.wfss[wfs].getMetaPupilPos(
    self.sim.config.atmos.scrnHeights[i])*self.sim.config.sim.pxlScale+self.config.sim.pupilSize

                    if self.sim.wfss[wfs].radii!=None:
                        radius = self.sim.wfss[wfs].radii[i]
                    
                    else:
                        radius = self.config.sim.pupilSize/2.
                
                    if self.sim.config.wfs[wfs].GSHeight!=0:
                        colour="r"
                    else:
                        colour="g"
                    
                    circ = pylab.Circle(cent,radius=radius,alpha=0.4,fc=colour)
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
        self.strehlAxes.set_ylabel("Strehl Ratio (%)",fontsize="xx-small")
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
        if self.config.sim.nSci>0:
            self.strehlPlts.append(self.strehlAxes.plot(self.sim.instStrehl[0],
                    linestyle=":", color=self.colorList[self.colorNo]))
            self.strehlPlts.append(self.strehlAxes.plot(self.sim.longStrehl[0],
                 color=self.colorList[self.colorNo]))
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
        self.config.dm[dm].gain = self.gainSpins[dm].value()

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
    
    def __init__(self,guiObj):

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
    def __init__(self,layout,sim,gui):
        # Create an in-process kernel
        # >>> print_process_id()
        # will print the same process ID as the main process
        self.kernel_manager = QtInProcessKernelManager()
        self.kernel_manager.start_kernel()
        self.kernel = self.kernel_manager.kernel
        self.kernel.gui = 'qt4'

        self.kernel.shell.write("Welcome to AO Sim!")

        self.kernel.shell.push({"sim":sim, "gui":gui})
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





