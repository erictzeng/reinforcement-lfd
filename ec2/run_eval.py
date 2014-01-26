import run_job as job
import argparse
import os.path

def build_conf(weightfile, extra_flags=None):
    if extra_flags is None:
        extra_flags = []
    name = os.path.splitext(os.path.basename(weightfile))[0]
    conf = {}
    conf['keyfile'] = '~/.ssh/eric-east-nv.pem'
    conf['commit'] = 'HEAD'
    conf['output'] = 'jobs/eval/{0}/{0}.h5'.format(name, name)
    conf['workdir'] = 'jobs/eval/{}'.format(name)
    conf['num_examples'] = 100
    conf['start'] = 0
    conf['end'] = 100
    conf['num_instances'] = 1
    conf['jobs_per_instance'] = 7
    cmd = 'python reinforcement-lfd/do_task_eval.py --i_start={{start}} --i_end={{end}} reinforcement-lfd/data/misc/actions.h5 reinforcement-lfd/data/misc/holdout_set.h5 {} --resultfile=out/{{id}}-{{num}}.h5'
    conf['command'] = cmd.format(os.path.basename(weightfile))
    if extra_flags:
        conf['command'] += ' ' + ' '.join(extra_flags)
    instance_info = {}
    conf['instance_info'] = instance_info
    instance_info['image_id'] = 'ami-21f7ca48'
    instance_info['key_name'] = 'eric-east-nv'
    instance_info['security_groups'] = ['reinforcement-lfd']
    instance_info['instance_type'] = 'c3.2xlarge'
    conf['files_to_copy'] = [weightfile]
    return conf

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('weightfile')
    parser.add_argument('extraflags', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if not os.path.exists(args.weightfile):
        print 'Weightfile {} not found!'.format(args.weightfile)
        exit(1)
    conf = build_conf(args.weightfile, extra_flags=args.extraflags)
    job.run(conf)
