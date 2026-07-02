
import platform
from setuptools import setup
import versioneer

scripts = ["bin/soapy"]
if platform.system() == "Windows":
    scripts.append("bin/soapy.bat")

setup(
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    scripts=scripts,
)
