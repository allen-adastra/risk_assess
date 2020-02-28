import ctypes
from ctypes import cdll

class AMS(ctypes.Structure):
    _fields_ = [('E_x', ctypes.c_double),
                ('E_y', ctypes.c_double),
                ('E_xy', ctypes.c_double),
                ('E2_x', ctypes.c_double),
                ('E2_y', ctypes.c_double),
                ('E_xvs', ctypes.c_double),
                ('E_xvc', ctypes.c_double),
                ('E_yvs', ctypes.c_double),
                ('E_yvc', ctypes.c_double),
                ('E_xs', ctypes.c_double),
                ('E_xc', ctypes.c_double),
                ('E_ys', ctypes.c_double),
                ('E_yc', ctypes.c_double),
                ('E_v', ctypes.c_double),
                ('E2_v', ctypes.c_double)]

class ExogMoments(ctypes.Structure):
    _fields_ = [('E_wv', ctypes.c_double),
                ('E2_wv', ctypes.c_double),
                ('E_c', ctypes.c_double),
                ('E2_c', ctypes.c_double),
                ('E_s', ctypes.c_double),
                ('E2_s', ctypes.c_double),
                ('E_cs', ctypes.c_double),
                ('E_cw', ctypes.c_double),
                ('E_sw', ctypes.c_double)]

def wrap_function(lib, funcname, restype, argtypes):
    """Simplify wrapping ctypes functions"""
    func = lib.__getattr__(funcname)
    func.restype = restype
    func.argtypes = argtypes
    return func

lib = cdll.LoadLibrary('/home/allen/plan_verification_rss_2020/risk_assess/build/libtest_cpp_moment_dynamics.so')
test_fun = wrap_function(lib, 'propagate_moments', AMS, [AMS, ExogMoments])
foo1 = AMS()
foo2 = ExogMoments()
import time
tstart = time.time()
for i in range(int(1e3)):
    bar = test_fun(foo1, foo2)
t_tot = time.time() - tstart
print(t_tot)