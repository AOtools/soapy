#! /usr/bin/env python

#Copyright Durham University and Andrew Reeves
#2015

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

import os
import sys
if sys.platform == "win32":
    # hack to stop the interpretter craching on ctrl-c events when scipy imported.
    # described in more detail here:
    # http://stackoverflow.com/questions/15457786/ctrl-c-crashes-python-after-importing-scipy-stats

    import imp
    import ctypes
    import _thread as thread
    import win32api

    # Load the DLL manually to ensure its handler gets
    # set before our handler.
    basepath = os.path.dirname(sys.executable)
    ctypes.CDLL(os.path.join(basepath, "Library", "bin", 'libmmd.dll'))
    ctypes.CDLL(os.path.join(basepath, "Library", "bin", 'libifcoremd.dll'))

    # Now set our handler for CTRL_C_EVENT. Other control event
    # types will chain to the next handler.
    def handler(dwCtrlType, hook_sigint=thread.interrupt_main):
        if dwCtrlType == 0: # CTRL_C_EVENT
            hook_sigint()
            return 1 # don't chain to the next handler
        return 0 # chain to the next handler

    win32api.SetConsoleCtrlHandler(handler, 1)

# Useful things to have in top namespace
from .simulation import Sim, make_mask
from .confParse import loadSoapyConfig
from .atmosphere import makePhaseScreens

# Compatability with older API
from . import wfs as WFS

#Try to import GUI, if not then its ok
try:
    from . import gui
except:
    pass


from ._version import get_versions
__version__ = get_versions()['version']
del get_versions
