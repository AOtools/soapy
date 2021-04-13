
import platform
from setuptools import setup, find_packages
import versioneer


versioneer.VCS = 'git'
versioneer.versionfile_source = 'soapy/_version.py'
versioneer.versionfile_build = 'soapy/_version.py'
versioneer.tag_prefix = '' # tags are like 1.2.0
versioneer.parentdir_prefix = 'soapy-' # dirname like 'myproject-1.2.0'


scripts = ["bin/soapy"]

if (platform.system() == "Windows"):
    scripts.append("bin/soapy.bat")

  
setup(
    name='soapy',
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    author='Andrew P. Reeves',
    author_email='andrewpaulreeves@gmail.com',
    packages=find_packages(),
    scripts=scripts,
    description='A tomographic astronomical adaptive optics simulation with realistic laser guide star propagation.',
    long_description=open('README.md').read(),
    install_requires=[
        "numpy >= 1.7.0",
        "scipy >= 0.15.0",
        "astropy >= 1.0",
        "aotools >= 1.0",
        "pyfftw >= 0.12.0",
        "pyyaml >= 5.1.1",
        "numba >= 0.40"
      ],
    classifiers=[
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
    ],
)
