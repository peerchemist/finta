from setuptools import setup

classifiers = [
  'Development Status :: 3 - Alpha',
  'Intended Audience :: Financial and Insurance Industry',
  'Programming Language :: Python',
  'Operating System :: OS Independent',
  'Natural Language :: English',
  'License :: OSI Approved :: GNU Lesser General Public License v3 or later (LGPLv3+)',
]

setup(name='finta',
      version='0.2.8',
      description=' Common financial technical indicators implemented in Pandas.',
      url='https://github.com/peerchemist/finta',
      author='Peerchemist',
      author_email='peerchemist@protonmail.ch',
      license='LGPLv3+',
      packages=['finta'],
      install_requires=['pandas']
      )
