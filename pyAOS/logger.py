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

LOGGING_LEVEL = 3
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

def _printMessage(message):
	"""
	Internal function to add debug info to message and print

	Args:
		message(str): The message to log
	"""

	curframe = inspect.currentframe()
	calframe = inspect.getouterframes(curframe, 2)

	logMessage = calframe[2][1].split("/")[-1]+" - "+calframe[2][3] + ": " + message
	if LOGGING_FILE:
		with open(LOGGING_FILE, "w") as File:
			File.write(logMessage+"\n")
	print(logMessage)

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
	
	if LOGGING_LEVEL==2:
			print(message)
	elif LOGGING_LEVEL>=3:
			_printMessage(message)
	else:
		pass


def debug(message):
	"""
	Logs messages if debug level is 3. Intended for very detailed debugging information.

	Args:
		message (string): The message to log
	"""
	if LOGGING_LEVEL>=3:
		_printMessage("Debug: {0}".format(message))
	
	else:
		pass

def warning(message):
	"""
	Logs messages if debug level is 1 or over. Intended for warnings

	Args:
		message (string): The message to log
	"""
	if LOGGING_LEVEL>=1:
		_printMessage("WARNING: {0}".format(message))		
	else:
		pass



# class Logger(object):
# 	"""
# 	The object providing a common interface for all logging.

# 	Args:
# 		filename (string): Filename of the file to log to. If ``None``, then logging isn't saved to file.
# 		verbosity (int): controls amount of output. Can be set to 0,1,2,3
# 	"""
# 	def __init__(self):

# 		self.loggingFile = LOGGING_FILE
# 		self.verbosity = LOGGING_LEVEL


# 	def _setVerbosity(self, v):
# 		global LOGGING_LEVEL

# 		if v<0:
# 			self._verbosity = 0
# 		elif v>3:
# 			self._verbosity = 3
# 		else:
# 			self._verbosity = int(v)	

# 		LOGGING_LEVEL = self.verbosity
# 		self.debug("Logging level set to: %d"%self._verbosity)

# 	def _getVerbosity(self):
		
# 		return self._verbosity

# 	def _setLoggingFile(self, f):
# 		global LOGGING_FILE

# 		self._loggingFile = f
# 		LOGGING_FILE = self.loggingFile

# 	def _getLoggingFile(self):
# 		return self._loggingFile


# 	def _printMessage(self, message):
# 		"""
# 		Internal function to add debug info to message and print

# 		Args:
# 			message(str): The message to log
# 		"""

# 		curframe = inspect.currentframe()
# 		calframe = inspect.getouterframes(curframe, 2)

# 		logMessage = calframe[2][1].split("/")[-1]+" - "+calframe[2][3] + ": " + message
# 		if self.loggingFile:
# 			with open(self.filename, "w") as File:
# 				File.write(logMessage+"\n")
# 		print(logMessage)

# 	def print_(self, message):
# 		"""
# 		Always logs message, regardless of verbosity level

# 		Args:
# 			message(str): The message to log
# 		"""

# 		self._printMessage(message)


# 	def info(self, message):
# 		"""
# 		Logs message if verbosity is 2 or higher. Useful for information which is not vital, but good to know.

# 		Args:
# 			message (string): The message to log
# 		"""
		
# 		if self.verbosity>=2:
# 			#self._printMessage("Info: {0}".format(message))
# 			print(message)
# 		else:
# 			pass


# 	def debug(self, message):
# 		"""
# 		Logs messages if debug level is 3. Intended for very detailed debugging information.

# 		Args:
# 			message (string): The message to log
# 		"""
# 		if self.verbosity>=3:
# 			self._printMessage("Debug: {0}".format(message))
		
# 		else:
# 			pass

# 	def warning(self, message):
# 		"""
# 		Logs messages if debug level is 1 or over. Intended for warnings

# 		Args:
# 			message (string): The message to log
# 		"""
# 		if self.verbosity>=1:
# 			self._printMessage("WARNING: {0}".format(message))		
# 		else:
# 			pass
		
# 	verbosity = property(_getVerbosity, _setVerbosity, doc="The verbosity level of the logger. If 0, nothing is logged, if 1 debug is logged, if 2 debug and info is logged")	
# 	loggingFile = property(_getLoggingFile, _setLoggingFile)
	
	
	
