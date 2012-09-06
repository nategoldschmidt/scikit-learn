#!/usr/bin/env sh

PYTHON_INCLUDE='/usr/include/python2.7'
NUMPY_INCLUDE='/usr/lib/python2.7/site-packages/numpy/core/include'

cython _LCA.pyx
gcc -O3 -c -fPIC -I ${PYTHON_INCLUDE} -I ${NUMPY_INCLUDE} _LCA.c
gcc -shared _LCA.o -o _LCA.so
