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

# Useful things to have in top namespace
from .simulation import Sim, make_mask
from .confParse import loadSoapyConfig
from .atmosphere import makePhaseScreens

# Compatability with older API
from . import wfs as WFS
from . import scienceinstrument as SCI

#Try to import GUI, if not then its ok
# Don't do this as it slows down importing for script or CLI use
# try:
#     from . import gui
# except:
#     pass


from . import _version
__version__ = _version.get_versions()['version']
