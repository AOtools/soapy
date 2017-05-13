from setuptools import setup, find_packages
import versioneer


versioneer.VCS = 'git'
versioneer.versionfile_source = 'soapy/_version.py'
versioneer.versionfile_build = 'soapy/_version.py'
versioneer.tag_prefix = '' # tags are like 1.2.0
versioneer.parentdir_prefix = 'soapy-' # dirname like 'myproject-1.2.0'
  
  
setup(
    name='soapy',
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    author='Andrew Reeves',
    author_email='a.p.reeves@durham.ac.uk',
    packages=find_packages(),
    scripts=['bin/soapy'],
    description='A tomographic astronomical adaptive optics simulation with realistic laser guide star propagation.',
    long_description=open('README.md').read(),
    install_requires=[
        "numpy >= 1.7.0",
        "scipy >= 0.15.0",
        "astropy >= 1.0",
        # "aotools >= 0.3.0"
      ],
)
