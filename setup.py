from distutils.core import setup
import versioneer


versioneer.VCS = 'git'
versioneer.versionfile_source = 'pyAOS/_version.py'
versioneer.versionfile_build = 'pyAOS/_version.py'
versioneer.tag_prefix = '' # tags are like 1.2.0
versioneer.parentdir_prefix = 'pyAOS-' # dirname like 'myproject-1.2.0'
  
  
setup(
    name='pyAOS',
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    author='Andrew Reeves',
    author_email='a.p.reeves@durham.ac.uk',
    packages=['pyAOS', 'pyAOS'],
    scripts=['bin/pyAOS'],
    description='A tomographic astronomical adaptive optics simulation with realistic laser guide star propagation.',
    long_description=open('README.md').read(),
    # install_requires=[
    #     "numpy >= 1.8.0",
    #     "scipy >= 0.14.0",
    #     "pyfits >= 3.3",
    # ],
)
