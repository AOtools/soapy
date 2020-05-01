from PyQt5 import QtWidgets

from qtconsole.rich_jupyter_widget import RichJupyterWidget as RichIPythonWidget
from qtconsole.inprocess import QtInProcessKernelManager

class JupyterConsoleWidget(QtWidgets.QWidget):
    def __init__(self):
        # Create an in-process kernel
        self.kernel_manager = QtInProcessKernelManager()
        self.kernel_manager.start_kernel()

        self.kernel = self.kernel_manager.kernel

        self.kernel_client = self.kernel_manager.client()
        self.kernel_client.start_channels()

        control = RichIPythonWidget()
        control.kernel_manager = self.kernel_manager
        control.kernel_client = self.kernel_client
        control.exit_requested.connect(self.stop)
    
        self.setCentralWidget(control)

    def stop(self):
        self.kernel_client.stop_channels()
        self.kernel_manager.shutdown_kernel()

    def write(self,message):
        self.kernel.shell.write(message)
        self.kernel.shell.ex("")



if __name__ == "__main__":

    jcw = JupyterConsoleWidget()

    jcw.show()