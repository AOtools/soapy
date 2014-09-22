from distutils.core import setup

setup(
    name='pyAOS',
    version='0.1.0',
    author='Andrew Reeves',
    author_email='a.p.reeves@durham.ac.uk',
    packages=['pyAOS', 'pyAOS'],
    scripts=['bin/pyAOS'],
    description='A tomographic astronomical adaptive optics simulation with realistic laser guide star propagation.',
    long_description=open('README.md').read(),
    install_requires=[
        "numpy >= 1.8.0",
        "scipy >= 0.14.0",
    ],
)
