import run_job as job
import h5py
import argparse
import os.path
from pprint import pprint

def build_conf(weightfile, num_instances=1, extra_copies=None, extra_flags=None):
    if extra_flags is None:
        extra_flags = []
    if extra_copies is None:
        extra_copies = []
    name = os.path.splitext(os.path.basename(weightfile))[0]
    conf = {}
    conf['keyfile'] = '~/.ssh/eric-east-nv.pem'
    conf['commit'] = 'HEAD'
    conf['output'] = 'jobs/eval/{0}/{0}.h5'.format(name, name)
    conf['workdir'] = 'jobs/eval/{}'.format(name)
    conf['num_examples'] = 100
    conf['start'] = 0
    conf['end'] = conf['num_examples']
    conf['num_instances'] = num_instances
    conf['jobs_per_instance'] = 7
    cmd = 'python reinforcement-lfd/do_task_eval.py --i_start={{start}} --i_end={{end}} reinforcement-lfd/data/misc/actions.h5 reinforcement-lfd/data/misc/holdout_set.h5 {} --resultfile=out/{{id}}-{{num}}.h5'
    conf['command'] = cmd.format(os.path.basename(weightfile))
    if extra_flags:
        conf['command'] += ' ' + ' '.join(extra_flags)
    instance_info = {}
    conf['instance_info'] = instance_info
    instance_info['image_id'] = 'ami-11ead678'
    instance_info['key_name'] = 'eric-east-nv'
    instance_info['security_groups'] = ['reinforcement-lfd']
    instance_info['instance_type'] = 'c3.2xlarge'
    conf['files_to_copy'] = [weightfile] + extra_copies
    return conf

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('weightfile')
    parser.add_argument('extraflags', nargs=argparse.REMAINDER)
    parser.add_argument('--num_instances', type=int, default=1)
    parser.add_argument('--copy', default=[], action='append')
    args = parser.parse_args()
    all_found = True
    if not os.path.exists(args.weightfile):
        print 'Weightfile {} not found!'.format(args.weightfile)
        all_found = False
    for to_copy in args.copy:
        if not os.path.exists(to_copy):
            print 'File to copy {} not found!'.format(to_copy)
            all_found = False
    if not all_found:
        print 'Files not found! Terminating.'
        exit(1)
    conf = build_conf(args.weightfile,
                      num_instances=args.num_instances,
                      extra_copies=args.copy,
                      extra_flags=args.extraflags)
    print 'Running with the following configuration:'
    print
    pprint(conf)
    print
    print 'Is this okay?',
    if job.yesno():
        print 'Running job.'
        job.run(conf)
    else:
        print 'Cancelled.'
