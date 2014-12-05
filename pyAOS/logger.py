#Copyright Durham University and Andrew Reeves
#2014

# This file is part of pyAOS.

#     pyAOS is free software: you can redistribute it and/or modify
#     it under the terms of the GNU General Public License as published by
#     the Free Software Foundation, either version 3 of the License, or
#     (at your option) any later version.

#     pyAOS is distributed in the hope that it will be useful,
#     but WITHOUT ANY WARRANTY; without even the implied warranty of
#     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#     GNU General Public License for more details.

#     You should have received a copy of the GNU General Public License
#     along with pyAOS.  If not, see <http://www.gnu.org/licenses/>.
"""
A module to provide a common logging interface for all simulation code.

Contains a ``Logger`` object, which can either, print information, save to file
or both. The verbosity can also be adjusted between 0 and 3, where all is logged when verbosity is 3, debugging and warning information is logged when verbosity is 2, warnings logged when verbosity is 1 and nothing is logged when verbosity is 0. 
"""
import inspect
import sys

LOGGING_LEVEL = 1
LOGGING_FILE = None
STATUS_FUNC = None

def setLoggingLevel(level):
	global LOGGING_LEVEL
	LOGGING_LEVEL = level

def setLoggingFile(logFile):
	global LOGGING_FILE
	LOGGING_FILE = logFile

def setStatusFunc(func):
	global STATUS_FUNC
	STATUS_FUNC = func


def statusMessage(i, maxIter, message):
	if not STATUS_FUNC:
		sys.stdout.flush()
		sys.stdout.write("\r{0} of {1}: {2}".format(i+1,maxIter, message))

	else:
		STATUS_FUNC(message, i, maxIter)

	if i+1==maxIter:
		if not STATUS_FUNC:
			sys.stdout.flush()
			sys.stdout.write("\n")

def _printMessage(message, level=3):
	"""
	Internal function to add debug info to message and print

	Args:
		message(str): The message to log
	"""

	if LOGGING_LEVEL>=level:
		if LOGGING_LEVEL>2 or level==1:
			curframe = inspect.currentframe()
			calframe = inspect.getouterframes(curframe, 2)
			message = calframe[2][1].split("/")[-1]+" - "+calframe[2][3] + ": " + message

		if LOGGING_FILE:
			with open(LOGGING_FILE, "a") as File:
				File.write(message+"\n")

		if STATUS_FUNC:
			STATUS_FUNC(message)

		print(message)

def print_(message):
	"""
	Always logs message, regardless of verbosity level

	Args:
		message(str): The message to log
	"""

	_printMessage(message)


def info(message):
	"""
	Logs message if verbosity is 2 or higher. Useful for information which is not vital, but good to know.

	Args:
		message (string): The message to log
	"""
	
	_printMessage(message, 2)

def debug(message):
	"""
	Logs messages if debug level is 3. Intended for very detailed debugging information.

	Args:
		message (string): The message to log
	"""
	_printMessage(message, 3)

def warning(message):
	"""
	Logs messages if debug level is 1 or over. Intended for warnings

	Args:
		message (string): The message to log
	"""
	_printMessage(message,1)

