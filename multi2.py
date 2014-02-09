#!/usr/bin/env python
import multiprocessing,os,time,sys
import numpy as np
beta = np.random.beta
randint = np.random.randint
def flip(p=.5): return np.random.binomial(1,p)
def repeat(f,n): return [] if n==0 else ( [f(),] + repeat(f,n-1) )
def uni(lst): return lst[np.random.randint(len(lst))]


def daemon():
    print 'Starting:', multiprocessing.current_process().name
    time.sleep(20)
    print 'Exiting :', multiprocessing.current_process().name

def non_daemon():
    print 'Starting:', multiprocessing.current_process().name
    print 'Exiting :', multiprocessing.current_process().name

def divisors(n,quout):
    print 'Starting:', multiprocessing.current_process().name
    ds= [i for i in range(2,n-1) if n%i==0] 
    print 'Exiting :', multiprocessing.current_process().name
    quout.put(ds)
    return ds

def ins(s,w):
    ls = s.split()
    ls.insert(randint(len(ls)),w)
    return ' '.join(ls)


nouns=['time','space','being','spirit','life','energy','structure','existence','matter','mind']
verbs=['speaks','is','leans','laughs','defies','intervenes','instigates','dissolves','transubstantiates']

wl2 = ['only','again','exactly','fabulously','(!)','...','hithero',
     'exasperatingly','--absurdly--']
code='''def divisors(n,quout):
    print 'Starting:', multiprocessing.current_process().name
    ds= [i for i in range(2,n-1) if n%i==0] 
    print 'Exiting :', multiprocessing.current_process().name
    quout.put(ds)'''
sts = ['The s was not created by any human being. I call it natural, inhuman, ahuman, evolved by the very same physics ',
       'The Assyrian came down like a wolf on the fold. His cohorts they were gleaming in ocher and gold. The swords and their shields were like',
       'It was the best of times; It was the breakfasting at noon and the dining at midnight. The cock and balls stories and bullfighting in the rafters.','Beastly, these place-holders scattered pell-mell in my text.','My years, my dreams, my failure.',
       'Hansel and grettelina went to villa','Houston Texas as the future of the world','Dreams of another day','Dogville, superbad']
fs = [lambda s: s.replace(' I ',' Bobby '),lambda s:s.swapcase(),
          lambda s: ' '.join(sorted(s.split(),key=len)),
          lambda s: s.replace('.','?'),
          lambda s: s.replace('the','basket'),
          lambda s: s+'--or so they say',
          lambda s: 'Demonstrably, '+s,
          lambda s: s.replace('a','o'),
          lambda s: 'John utters: ' + s,
          lambda s: s.replace(' ','-')] + [ lambda s: s.replace('the',uni(wl2)) ]*8

         
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

def maker(out_q,repeats):
    print 'Starting :', multiprocessing.current_process().name
    for i in range(repeats):
         s=uni(nouns).upper() + ' ' + uni(verbs) + '.'
         print multiprocessing.current_process().name+' created: ',s,'\n'
         out_q.put(s)
         time.sleep(0.5)
    print 'Exiting :', multiprocessing.current_process().name

    
if __name__ == '__main__':
    runs=8; fav = ['katze','Ganges']
    repeats = 20
    nprocs = 1
    procs = []; gprocs = []
    out_q = multiprocessing.Queue()

    for i in range(nprocs):
        g = multiprocessing.Process(target=maker,args=(out_q,repeats) )
        p = multiprocessing.Process(target=evolve,args=(sts[i],runs, out_q,fav[i]) )
        g.demon = True;   p.demon = True
        gprocs.append(g)
        procs.append(p)
        g.start()
        time.sleep(3) # wait for g.start to have put things on the queue
        p.start()

    for p in procs:
        p.join()

    for g in gprocs:
        g.join()




