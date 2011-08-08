#testbranch script (formerly in fabfile.py)

import sys
import os
import shutil
import subprocess
import atexit
import time
import datetime
import getpass
import fnmatch
import socket
from fabric.api import run, env, local, put, cd, get, settings, prompt, \
                       hide, show
from fabric.state import connections
from socket import gethostname
import ConfigParser

from openmdao.devtools.utils import get_git_branch, repo_top, remote_tmpdir, \
                                    push_and_run, rm_remote_tree, make_git_archive,\
                                    fabric_cleanup, remote_listdir, remote_mkdir,\
                                    ssh_test
from openmdao.devtools.remote_cfg import CfgOptionParser, process_options, \
                                         run_host_processes

from openmdao.devtools.tst_ec2 import run_on_ec2_image

import paramiko.util

def test_on_remote_host(remotedir=None, fname=None, 
                        pyversion='python', keep=False, 
                        branch=None, testargs=(), hostname=''):
    if remotedir is None:
        raise RuntimeError("test_on_remote_host: missing arg 'remotedir'")
    if fname is None:
        raise RuntimeError("test_on_remote_host: missing arg 'fname'")
    
    remote_mkdir(remotedir)
    
    locbldfile = os.path.join(os.path.dirname(__file__), 'locbuild.py')
    loctstfile = os.path.join(os.path.dirname(__file__), 'loctst.py')
    
    if fname.endswith('.py'):
        build_type = 'release'
    else:
        build_type = 'dev'
        
    if os.path.isfile(fname):
        remotefname = os.path.join(remotedir, os.path.basename(fname))
        print 'putting %s on remote host' % fname
        put(fname, remotefname) # copy file to remote host
        remoteargs = ['-f', os.path.basename(fname)]
    else:
        remoteargs = ['-f', fname]
        
    remoteargs.append('--pyversion=%s' % pyversion)
    if branch:
        remoteargs.append('--branch=%s' % branch)
        
    expectedfiles = set(['locbuild.py','build.out'])
    dirfiles = set(remote_listdir(remotedir))
    
    print 'building...'
    with settings(warn_only=True):
        result = push_and_run(locbldfile, 
                              remotepath=os.path.join(remotedir,
                                                      os.path.basename(locbldfile)),
                              args=remoteargs)
    print result
    # retrieve build output file
    get(os.path.join(remotedir, 'build.out'), 'build.out')
    
    if result.return_code != 0:
        raise RuntimeError("problem with remote build (return code = %s)" % 
                           result.return_code)
    
    print 'build successful\ntesting...'
    newfiles = set(remote_listdir(remotedir)) - dirfiles - expectedfiles
    
    if build_type == 'dev':
        if len(newfiles) != 1:
            raise RuntimeError("expected a single new file in %s after building but got %s" %
                               (remotedir, list(newfiles)))
        
        builddir = newfiles.pop()
        envdir = os.path.join(builddir, 'devenv')
    else:
        matches = fnmatch.filter(newfiles, 'openmdao-?.*')
        if len(matches) > 1:
            raise RuntimeError("can't uniquely determine openmdao env directory from %s" % matches)
        elif len(matches) == 0:
            raise RuntimeError("can't find an openmdao environment directory to test in")
        envdir = matches[0]

    remoteargs = ['-d', envdir]
    remoteargs.append('--pyversion=%s' % pyversion)
    if keep:
        remoteargs.append('--keep')
    if len(testargs) > 0:
        remoteargs.append('--')
        remoteargs.extend(testargs)
        
    result = push_and_run(loctstfile, 
                          remotepath=os.path.join(remotedir,
                                                  os.path.basename(loctstfile)),
                          args=remoteargs)
    print result
        
    if remotedir is not None and (result.return_code==0 or not keep):
        rm_remote_tree(remotedir)
        
    return result.return_code
        
def main(argv=None):
    socket.setdefaulttimeout(30)
    t1 = time.time()
    
    if argv is None:
        argv = sys.argv[1:]
        
    parser = CfgOptionParser()
    parser.add_option("-k","--keep", action="store_true", dest='keep',
                      help="if there are test/build failures, don't delete "
                           "the temporary build directory "
                           "or terminate the remote instance if testing on EC2.")
    parser.add_option("-f","--file", action="store", type='string', dest='fname',
                      help="pathname of a tarfile or URL of a git repo")
    parser.add_option("-b","--branch", action="store", type='string', 
                      dest='branch',
                      help="if file_url is a git repo, supply branch name here")

    (options, args) = parser.parse_args(sys.argv[1:])
    
    config, conn, image_hosts = process_options(options)
    
    startdir = os.getcwd()
    
    if options.fname is None: # assume we're testing the current repo
        print 'creating tar file of current branch: ',
        options.fname = os.path.join(os.getcwd(), 'testbranch.tar')
        ziptarname = options.fname+'.gz'
        if os.path.isfile(ziptarname): # clean up the old tar file
            os.remove(ziptarname)
        make_git_archive(options.fname)
        subprocess.check_call(['gzip', options.fname])
        options.fname = os.path.abspath(options.fname+'.gz')
        print options.fname
        
    fname = options.fname
    
    if fname.endswith('.tar.gz') or fname.endswith('.tar'):
        if not os.path.isfile(fname):
            print "can't find tar file '%s'" % fname
            sys.exit(-1)
    elif fname.endswith('.git'):
        pass
    else:
        parser.print_help()
        print "\nfilename must end in '.tar.gz', '.tar', or '.git'"
        sys.exit(retcode)
        
    funct_kwargs = { 'keep': options.keep,
                     'branch': options.branch,
                     'testargs': args,
                     'fname': fname,
                     'remotedir': options.remotedir,
                     }
    run_host_processes(config, conn, image_hosts, options, 
                       funct=test_on_remote_host, funct_kwargs=funct_kwargs)
    
    #processes = []
    
    #try:
        #for host in options.hosts:
            #shell = config.get(host, 'shell')
            #if host in image_hosts:
                #runner = run_on_ec2_image
            #else:
                #runner = run_on_host
            #proc_args = [host, config, conn, test_on_remote_host,
                         #options.outdir, fname, shell, options.remotedir]
            #p = Process(target=runner,
                        #name=host,
                        #args=proc_args,
                        #kwargs={ 'keep': options.keep,
                                 #'branch': options.branch,
                                 #'testargs': args,
                                 #'hostname': host,
                                 #})
            #processes.append(p)
            #print "starting build/test process for %s" % p.name
            #p.start()
        
        #while len(processes) > 0:
            #time.sleep(1)
            #for p in processes:
                #if p.exitcode is not None:
                    #processes.remove(p)
                    #if len(processes) > 0:
                        #remaining = '(%d hosts remaining)' % len(processes)
                    #else:
                        #remaining = ''
                    #print '%s finished. exit code=%d %s\n' % (p.name, 
                                                              #p.exitcode, 
                                                              #remaining)
                    #break
            
    #finally:
        #os.chdir(startdir)
        
        #t2 = time.time()
        #secs = t2-t1
        
        #hours = int(secs)/3600
        #mins = int(secs-hours*3600.0)/60
        #secs = secs-(hours*3600.)-(mins*60.)
        
        #print '\nElapsed time:',
        #if hours > 0:
            #print ' %d hours' % hours,
        #if mins > 0:
            #print ' %d minutes' % mins,
        #print ' %5.2f seconds\n\n' % secs


if __name__ == '__main__': #pragma: no cover
    atexit.register(fabric_cleanup, True)
    paramiko.util.log_to_file('paramiko.log')
    main()
