import boto.ec2 as ec2
import h5py
import paramiko

import argparse
import glob
from math import ceil
import os
import time
import socket
from threading import Thread
import yaml

def read_jobfile(path):
    with open(path, 'r') as jobfile:
        result = yaml.safe_load(jobfile)
    if 'start' not in result:
        result['start'] = 0
    if 'end' not in result:
        result['end'] = result['num_examples']
    if 'files_to_copy' not in result:
        result['files_to_copy'] = []
    return result

def launch_instances(connection, conf):
    info = dict(conf['instance_info'])
    num_instances = conf['num_instances']
    info['min_count'] = num_instances
    info['max_count'] = num_instances
    reservation = connection.run_instances(**info)
    return reservation

def attempt_connection(instance, keyfile):
    client = paramiko.SSHClient()
    client.load_system_host_keys()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    try:
        client.connect(instance.public_dns_name,
                       username='ubuntu',
                       key_filename=os.path.expanduser(keyfile))
    except socket.error, e:
        print e
        return None
    return client

def collect_results(client, conf):
    outdir = os.path.join(conf['workdir'], 'out')
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    sftp = client.open_sftp()
    fnames = sftp.listdir('out')
    for fname in fnames:
        sftp.get(os.path.join('out', fname),
                 os.path.join(outdir, fname))

def read_channel(channel):
    channel.readlines()

def copy_files(client, conf):
    sftp = client.open_sftp()
    for file_info in conf['files_to_copy']:
        sftp.put(file_info, os.path.basename(file_info))

def start_job(instance, client, conf, start, end):
    streams = []
    copy_files(client, conf)
    jobs_remaining = conf['jobs_per_instance']
    while jobs_remaining > 0:
        step = int(ceil(float(end - start) / jobs_remaining))
        info = {}
        info['id'] = instance.id
        info['num'] = conf['jobs_per_instance'] - jobs_remaining
        info['start'] = start
        info['end'] = start + step
        print 'Command: {}'.format(conf['command'].format(**info))
        gitcmd = 'cd reinforcement-lfd; git stash; git pull; git checkout {}'.format(conf['commit'])
        _, stdout, stderr = client.exec_command(gitcmd)
        stdout.readlines()
        _, stdout, stderr = client.exec_command(conf['command'].format(**info))
        t = Thread(target=read_channel, args=(stdout,))
        t.daemon = True
        t.start()
        write_log(conf, '{id}-{num} {start} {end}'.format(**info))
        streams.append((stdout, stderr))
        start += step
        jobs_remaining -= 1
    return streams

def write_log(conf, s):
    with open(os.path.join(conf['workdir'], 'log.txt'), 'a') as logfile:
        logfile.write(s + '\n')

def read_log_names(conf):
    with open(os.path.join(conf['workdir'], 'log.txt'), 'r') as logfile:
        for line in logfile:
            yield line.strip().split()[0]

def combine_results(conf):
    outfile = h5py.File(conf['output'], 'w')
    i = 0
    for key in read_log_names(conf):
        fname = os.path.join(conf['workdir'], 'out', key + '.h5')
        j = 0
        f = h5py.File(fname, 'r')
        while True:
            try:
                outfile.copy(f[str(j)], str(i))
            except KeyError:
                break
            i += 1
            j += 1

def done(streams):
    if all(stdout.channel.exit_status_ready() for stdout, stderr in streams):
        print [stdout.channel.recv_exit_status() for stdout, stderr in streams]
        print stderr.readlines()
        return True
    return False

def main(args):
    conn = ec2.connect_to_region('us-east-1')
    conf = read_jobfile(args.jobfile)
    if not os.path.exists(conf['workdir']):
        os.makedirs(conf['workdir'])
    reservation = launch_instances(conn, conf)
    instances = reservation.instances
    print 'Started {} instances.'.format(len(instances))
    statuses = {instance.id: 'pending' for instance in instances}
    clients = {}
    streams = {}
    num_examples = conf['end']
    curr_ind = conf['start']
    num_unstarted = conf['num_instances']
    while not all(status == 'done' for status in statuses.values()):
        time.sleep(5)
        for instance in instances:
            instance.update()
            if statuses[instance.id] == 'pending':
                print 'Attempting connection to {}.'.format(instance.id)
                client = attempt_connection(instance, conf['keyfile'])
                if client:
                    print 'Connected.'
                    clients[instance.id] = client
                    step = int(ceil(float(num_examples - curr_ind) / num_unstarted))
                    streams[instance.id] = start_job(instance,
                                                      client,
                                                      conf,
                                                      curr_ind,
                                                      curr_ind+step)
                    print 'Job started on {}.'.format(instance.id)
                    curr_ind += step
                    num_unstarted -= 1
                    statuses[instance.id] = 'running'
            elif statuses[instance.id] == 'running':
                if done(streams[instance.id]):
                    print 'Instance {} done.'.format(instance.id)
                    client = clients[instance.id]
                    collect_results(client, conf)
                    print 'Results collected from {}.'.format(instance.id)
                    client.close()
                    instance.terminate()
                    statuses[instance.id] = 'done'
    print 'All jobs finished. Combining results...'
    combine_results(conf)
    print 'Done.'



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('jobfile')
    args = parser.parse_args()
    main(args)
