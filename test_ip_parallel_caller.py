#!/usr/bin/env python

"""
Test ip_parallel.py. We run test_ip_parallel.ipy on the IPython 
interpreter.
"""

import subprocess

test_file = '/home/owainevans/myunit/test_ip_parallel.ipy'
out = subprocess.check_output(['ipython',test_file])
if 'error' in out.lower() or 'assertion' in out.lower():
    assert False, 'Error running %s in IPython' % test_file

