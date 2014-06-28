import h5py
import sys
import os.path

def delete(n, fname):
    with h5py.File(fname) as f:
        N = len(f.keys())
        for i in range(n):
            N -= 1
            del f[str(N)]
        return N

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print 'Helper script to delete collected data points.'
        print
        print 'python undo.py DATAFILE.h5'
        print
        print 'That will give you a prompt from which you can delete things if you desire.'
        exit(1)
    fname = sys.argv[1]
    if not os.path.exists(fname):
        print '{} not found, aborting'
        exit(1)
    print 'Found {}.'.format(fname)
    print 'Type a number N to delete the N most recent collected points, or q to quit.'
    while True:
        x = raw_input('> ')
        if x == 'q':
            break
        try:
            n = int(x)
        except ValueError:
            print "Not understood: " + x
            continue
        remaining = delete(n, fname)
        print 'Deleted {} entries from {}. {} remaining.'.format(n, fname, remaining)
