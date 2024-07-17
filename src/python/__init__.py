# import drjit first is important on Windows platform. Otherwise, the following error will occur:
# ImportError: DLL load failed while importing binding: The specified module could not be found.
import drjit
from wos_ext.wos_ext import *