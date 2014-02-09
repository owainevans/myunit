import os,time
from os import environ

def my_fork():
    environ['i']='0'
    print "i environmental variable set to: %s" % environ['i']
    n=5
    k=0
    child_pid = os.fork()
    if child_pid == 0:
        print "Child Process: PID# %s" % os.getpid()
        print "Child i environmental variable == %s" % environ['i']
        k += 1
        time.sleep(.2)

        for i in range(n):
            time.sleep(.8)
            k += 2
            print "Child Process: PID# %s" % os.getpid()
            print k
    else:
        print "Parent Process: PID# %s" % os.getpid()
        print "Parent i environmental variable == %s" % environ['i']

    for i in range(n):
        time.sleep(.8)
        k += 2
        print "Parent Process: PID# %s" % os.getpid()
        print k


if __name__ == "__main__":
    my_fork()
