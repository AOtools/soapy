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

from ._version import get_versions
__version__ = get_versions()['version']
del get_versions

from .simulation import Sim

# Compatability with older API
from . import wfs as WFS

#Try to import GUI, if not then its ok
try:
    from . import gui
except ImportError:
    pass
