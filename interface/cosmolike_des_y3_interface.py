"""Loader stub for the compiled CosmoLike interface of des_y3.

The des_y3 likelihoods call CosmoLike through cosmolike_des_y3_interface.so,
an extension module: a compiled shared library that Python imports like a
module (the likelihoods import it as ci). scripts/compile_des_y3.sh builds
it with interface/MakefileCosmolike from interface.cpp and the CosmoLike C
sources. This file is the loader stub setuptools writes for such a library;
it is not the interface itself.

Python's import system tries extension modules (.so) before source files
(.py) in the same folder. With the compiled .so next to this file, `import
cosmolike_des_y3_interface` therefore loads the .so directly and this file
never runs. It runs only when the folder holds no .so, and then its attempt
to load the .so beside it fails: compile the project first.
"""

def __bootstrap__():
   """Replace this stub module by the compiled .so of the same name.

   pkg_resources.resource_filename returns the path of
   cosmolike_des_y3_interface.so in the folder of this module, and
   imp.load_dynamic loads that shared library as the extension module of
   the same name, which replaces this stub in sys.modules (the table of
   loaded modules), so the import returns the compiled module. The global
   statement and the del remove the helper names from the module.
   """
   global __bootstrap__, __loader__, __file__
   import sys, pkg_resources, imp
   __file__ = pkg_resources.resource_filename(__name__,'cosmolike_des_y3_interface.so')
   __loader__ = None; del __bootstrap__, __loader__
   imp.load_dynamic(__name__,__file__)
__bootstrap__()
