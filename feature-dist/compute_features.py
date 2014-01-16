from numpy import array_split
import h5py
import paramiko
import yaml

import errno
import getpass
import glob
import os
import sys
import tarfile
import tempfile

def read_conf(confpath):
    with open(confpath, 'r') as conffile:
         conf = yaml.safe_load(conffile)
    return conf

def split_data(datafname, outfolder, n):
    if not os.path.exists(outfolder):
        os.makedirs(outfolder)
    with h5py.File(datafname, 'r') as datafile:
        num_examples = len(datafile.keys())
        grouped_indices = array_split(range(num_examples), n)
        for outfile_i, indices in enumerate(grouped_indices):
            outfname = os.path.join(outfolder, '{}.h5'.format(outfile_i))
            copy_indices(datafile, indices, outfname)

def copy_indices(datafile, indices, outfname):
    num_skipped = 0
    with h5py.File(outfname, 'w') as outfile:
        for i, index in enumerate(indices):
            if datafile[str(index)]['action'][()].startswith('endstate'):
                num_skipped += 1
                continue
            outfile.copy(datafile[str(index)], str(i))
    print "Skipped {} endstates.".format(num_skipped)

def pack_payload(conf):
    info = conf['payload']
    path = info['path']
    fnames = [os.path.join(path, fname) for fname in os.listdir(path)]
    tartemp = tempfile.NamedTemporaryFile(suffix='.tar')
    with tarfile.open(mode='w', fileobj=tartemp) as tar:
        for fname in fnames:
            tar.add(fname, arcname=os.path.relpath(fname, path))
        for fileinfo in info['additional-files']:
            tar.add(fileinfo['path'], arcname=fileinfo['archive-name'])
    assert os.path.exists(tartemp.name)
    return tartemp

def add_to_end(final, new, i):
    for index in new.iterkeys():
        final.copy(new[index], str(i))
        i += 1
    return i

def rexists(sftp, path):
    try:
        sftp.stat(path)
    except IOError, e:
        if e.errno == errno.ENOENT:
            return False
        raise
    return True

def check_remote_overwrites(conf, password=None):
    problems = []
    for server in conf['servers']:
        client = paramiko.SSHClient()
        client.load_system_host_keys()
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        client.connect(server['host'], password=password)
        sftp = client.open_sftp()
        exists = rexists(sftp, server['path'])
        sftp.close()
        if exists:
            problems.append((server['host'], server['path']))
    return problems

def distribute_jobs(conf, password=None):
    servers = conf['servers']
    problems = check_remote_overwrites(conf, password)
    if problems:
        for host, path in problems:
            print 'ERROR: {} on {} already exists!'.format(path, host)
        print 'Terminating.'
        exit(1)
    num_jobs = sum(server['cores'] for server in servers)
    split_data(conf['datafile'], conf['splitsdir'], num_jobs)
    split_i = 0
    stdouts = []
    print 'Preparing payload...'
    payloadtar = pack_payload(conf)
    for server in servers:
        client = paramiko.SSHClient()
        client.load_system_host_keys()
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        client.connect(server['host'], password=password)
        _, stdout, _ = client.exec_command('mkdir -p {}'.format(server['path']))
        stdout.readlines()
        sftp = client.open_sftp()
        print 'Copying scripts to {}...'.format(server['host'])
        remote_path = os.path.join(server['path'], os.path.basename(payloadtar.name))
        sftp.put(payloadtar.name, remote_path)
        _, stdout, _ = client.exec_command('tar -xf {} -C {}'.format(remote_path, server['path']))
        stdout.readlines()
        print 'Copying data files to {}...'.format(server['host'])
        _, stdout, _ = client.exec_command('mkdir -p {}'.format(os.path.join(server['path'], 'splits')))
        stdout.readlines()
        for i in range(split_i, split_i + server['cores']):
            sftp.put(os.path.join(conf['splitsdir'], '{}.h5'.format(i)),
                     os.path.join(server['path'], 'splits', '{}.h5'.format(i)))
        split_i += server['cores']
        stdin, stdout, stderr = client.exec_command("python {}".format(os.path.join(server['path'], 'driver.py')))
        stdouts.append(stdout)
    print "Waiting for servers to finish..."
    [stdout.readlines() for stdout in stdouts]
    print "Done. Collecting results..."
    collect_results(conf, password=password)

def collect_results(conf, password=None):
    if not os.path.exists(conf['outfolder']):
        os.makedirs(conf['outfolder'])
    for server in conf['servers']:
        print '- Collecting results from {}'.format(server['host'])
        client = paramiko.SSHClient()
        client.load_system_host_keys()
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        client.connect(server['host'], password=password)
        sftp = client.open_sftp()
        fnames = sftp.listdir(os.path.join(server['path'], 'out'))
        for fname in fnames:
            sftp.get(os.path.join(server['path'], 'out', fname),
                     os.path.join(conf['outfolder'], fname))
    outfiles = glob.glob(os.path.join(conf['outfolder'], '*.h5'))
    final_outfile = h5py.File(conf['outfile'], 'w')
    i = 0
    for outfile in outfiles:
        i += add_to_end(final_outfile, h5py.File(outfile, 'r'), i)
    print "Final results in {}.".format(conf['outfile'])


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print 'Distribute constraint computation across multiple machines.'
        print
        print 'python compute_features.py CONF'
        print
        print 'CONF should be a .yml file -- see servers.yml for an example.'
        exit(1)
    ymlfile = sys.argv[1]
    conf = read_conf(ymlfile)
    pw = getpass.getpass()
    distribute_jobs(conf, password=pw)

