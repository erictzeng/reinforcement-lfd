import os
import glob
import subprocess

if __name__ == '__main__':
    currdir = os.path.dirname(os.path.realpath(__file__))
    logdir = os.path.join(currdir, 'logs')
    if not os.path.exists(logdir):
        os.mkdir(logdir)
    outdir = os.path.join(currdir, 'out')
    if not os.path.exists(outdir):
        os.mkdir(outdir)
    os.chdir(currdir)
    subprocesses = []
    for fname in glob.glob('splits/*.h5'):
        fname = os.path.join(currdir, fname)
        _, nameonly = os.path.split(fname)
        logname = os.path.join(logdir, nameonly + '.txt')
        errname = os.path.join(logdir, nameonly + '.err')
        outname = os.path.join(outdir, nameonly)
        subprocesses.append(subprocess.Popen('python rope_qlearn.py --sc_features build-constraints {} {}'.format(fname, outname),
                            stdout=open(logname, 'w'),
                            stderr=open(errname, 'w'),
                            shell=True))
    exit_codes = [p.wait() for p in subprocesses]

