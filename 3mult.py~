#!/usr/bin/env python
import multiprocessing,os,time,sys
import numpy as np
beta = np.random.beta
randint = np.random.randint
def flip(p=.5): return np.random.binomial(1,p)
def repeat(f,n): return [] if n==0 else ( [f(),] + repeat(f,n-1) )
def uni(lst): return lst[np.random.randint(len(lst))]

         
def evolve(s,runs,out_q,fav):
    np.random.seed(int(round(time.time())))
    wl = wl2 + [fav]*20
    print 'Starting:', multiprocessing.current_process().name 

    while True:
        try:
            s = out_q.get(timeout=10)
        except:
            s='Last in the barrel'
            print s
            return None
        if flip(.8): s = ins(s,uni(wl))
        if flip(.8): s = uni(fs)(s)
        print 'name: ',multiprocessing.current_process().name,'\n',s+'\n'

    print 'Exiting :', multiprocessing.current_process().name

def worker(out_q,mynum):
    print 'Starting :', multiprocessing.current_process().name
    global n
    for i in range(10):
        sorted([x**2 for x in range(10**4) ])
        n += mynum
    shirker()
    print n
    print 'Exiting :', multiprocessing.current_process().name


def shirker():
    global n
    n += 22222
    
n=5

def ite(inq,outq,start):
    d=[start]
    it = lambda n: n+3
    name = multiprocessing.current_process().name

    while True:
        t = int(inq.get())
        for i in range(t):
            d.append( it(d[-1]) )
        time.sleep(1)
        print (name,d)
        outq.put( (name, start, d) )
    

if __name__ == '__main__':
    nprocs=2
    inq = multiprocessing.Queue()
    outq = multiprocessing.Queue()
    procs = []; starts=[0,1]; ts=[3,3]
    job_count = len(ts)
    
    for i in range(nprocs):
        p = multiprocessing.Process(target=ite, args=(inq,outq,starts[i]) )
        procs.append(p)
        
        inq.put( ts[i] )
        p.start()

    
    while int(outq.qsize()) != job_count:
        time.sleep(0.5)
        
    ts=[5,5]
    for i in range(nprocs): inq.put( ts[i] )
    job_count += len(ts)
    
    while int(outq.qsize()) != job_count:
        time.sleep(1)

    #for p in procs: p.terminate()




