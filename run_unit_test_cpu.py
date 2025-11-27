#!/usr/bin/env python
# Force JAX to use CPU before any imports
import os
os.environ['JAX_PLATFORMS'] = 'cpu'
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'

# Now run the unit test
exec(open('unit_test.py').read())
