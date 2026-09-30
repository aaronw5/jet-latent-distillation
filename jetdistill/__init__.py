import os, sys

if sys.platform == 'darwin':
    # macOS: numpy's BLAS (Accelerate) runs on GCD thread pools, which do not survive fork(); step 1's forked workers
    # then crash (SIGSEGV in dispatch_apply) and multiprocessing.Pool waits forever for their neurons. One BLAS thread
    # per process avoids GCD (the parallelism comes from the worker processes). Must be set before numpy is imported.
    os.environ.setdefault('VECLIB_MAXIMUM_THREADS', '1')
