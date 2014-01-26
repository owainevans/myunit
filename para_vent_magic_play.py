from IPython.core.magic import (Magics,register_cell_magic, magics_class, line_magic,cell_magic, line_cell_magic)

import numpy as np  
from my_unit import *
import os,time,multiprocessing
from venture.shortcuts import *
ipy_ripl = make_church_prime_ripl()


############# Functions for multiprocess inference        


#FIXME need to work out how long to sleep    
def make_ripl():
    'sleep before making'
    time.sleep(1)
    return make_church_prime_ripl()


def build_class(py_lines):
    'takes in lines of ripl.assume(...) code and builds class for model'
    
    class MyModelClass(VentureUnit):
        def makeAssumes(self):
            [eval(py_line) for py_line in py_lines if py_line[5]=='a']

        def makeObserves(self):
            [eval(py_line) for py_line in py_lines if py_line[5]=='o']


            # Example unit input from LDA
            # D = self.parameters['documents']
            # N = self.parameters['words_per_document']
        
            # for doc in xrange(D):
            #     for pos in xrange(N):
            #         self.observe("(get-word %d %d)" % (doc, pos), 0)

    return MyModelClass

    
def worker(out_q,ModelClass,ripl,params,no_sweeps,infer_msg,plot_msg):
    print 'Starting:', multiprocessing.current_process().name
    
    unit_model = ModelClass(ripl,params)
    if infer_msg=='runFromConditional':
        hist = unit_model.runFromConditional(sweeps=no_sweeps,runs=1)
        out_q.put(hist)
    print 'Exiting:', multiprocessing.current_process().name 



def multi_ripls(no_ripls,ModelClass,no_sweeps_list,infer_msg,plot_msg):

    # Parent code that runs the worker processes
    # do i need 'name'='main'?. we just run the script. worry that 
    # interactive will make it not work. look up again what interactive
    # entails for multiprocess. if we ran it, workers would have access
    # to a copy of the name space of their encloding env? key thing is 
    # we wouldn't want the workers to generate new child workers, which
    # we would get if we ran whole module. but here the code that generates
    # the workers is not in the global env of script. (we don't want the 
    # workers to run the ipython loading code, however, and so need to 
    # watch out for that ... but their having multi_ripls in namespace
    # or the ipython magics, shouldn't itself be a problemo. 

    assert(len(no_sweeps_list)==no_ripls)
    assert(isinstance(infer_msg,str))
    test_model = ModelClass(make_ripl(),{})
    print 'assumes:',test_model.assumes,'\n','observes:',test_model.observes
    assert(test_model.assumes != [])
    
    nprocs= no_ripls; procs = []
    out_q = multiprocessing.Queue() # could add max-size
    hists = []

    ripls = [make_ripl() for i in range(no_ripls)] 

    for i in range(nprocs):
        mytarget = worker
        myargs=(out_q, ModelClass, ripls[i],{},
                no_sweeps_list[i],infer_msg,plot_msg)

        p = multiprocessing.Process(target=mytarget,args=myargs)
        
        procs.append(p)
        p.start()

    for p in procs:
        p.join()

    while not(out_q.empty()):
        hists.append(out_q.get())
                                        
    return hists

                                              
def remove_white(s):
    t=s.replace('  ',' ')
    return s if s==t else remove_white(t)

## FIXME, all we need is tags and so this could be much shorter                                        
def cell_to_venture(s,terse=0):
    """Converts vchurch to python self.v.assume (OBSERVE is broken)"""
    s = str(s)
    s = s[:s.rfind(']')]
    ls = s.split(']')
    ls = [remove_white(line.replace('\n','')) for line in ls]

    # venture lines in python form, and dict of components of lines
    v_ls = []
    v_ls_d = {}

    for count,line in enumerate(ls):
        if terse==1: line = '[ASSUME '+line[1:]

        lparen = line.find('[')        
        tag = line[ lparen+1: ].split()[0].lower()

        if tag=='assume':
            var=line[1:].split()[1]
            exp = ' '.join( line[1:].split()[2:] )
            v_ls.append( "self.v.assume('%s', '%s')" % ( var, exp ) )
            v_ls_d[count] = (tag,var,exp)

        elif tag=='observe':
            var=line[1:].split()[1]
            exp = ' '.join( line[1:].split()[2:] )
            v_ls.append( "self.v.observe('%s', '%s')" % ( var, exp ) )
            v_ls_d[count] = (tag,var,exp)
        elif tag=='predict':
            exp = ' '.join( line[1:].split()[1:] )
            v_ls.append( "self.v.predict('%s')" % exp  )
            v_ls_d[count] = (tag,exp)
        elif tag=='infer':
            num = line[1:].split()[1]
            v_ls.append( "self.v.infer(%i)" % int(num) )
            v_ls_d[count] = (tag,num)
        elif tag=='clear':
            v_ls.append( "self.v.clear()" )
            v_ls_d[count] = (tag,'')   # comes with empty string for simplcity
        else:
            assert 0==1,"Did not recognize directive"

    #make tag upper
    for key in v_ls_d.keys():
        old = v_ls_d[key]
        v_ls_d[key] = (old[0].upper(),) + old[1:]

    return v_ls,v_ls_d


        
@magics_class
class ParaMagics(Magics):

    def __init__(self, shell):
        super(ParaMagics, self).__init__(shell)

    @line_cell_magic
    def vl2(self, line, cell=None):
        def format_parts(parts):
            'format the input string for pretty printing'
            return '[%s]' % ' '.join(parts)
        
        ## LINE MAGIC
        if cell is None:
            vouts = ipy_ripl.execute_instruction(str(line), params=None)

            py_lines,py_parts = cell_to_venture(line)
            
            for key in py_parts:
                print format_parts(py_parts[key])
                 
            if 'value' in vouts: print vouts['value'].get('value',None)

            return vouts
            
        ## CELL MAGIC    
        else:
            vouts = ipy_ripl.execute_program( str(cell), params=None )

            py_lines,py_parts = cell_to_venture(cell)
                              
            for count,v_line in enumerate(vouts):
                print format_parts(py_parts[count])
                if 'value' in v_line: print v_line['value'].get('value',None)

            return vouts
    

    @cell_magic
    def p(self, line, cell):
        'need to fix OBS issue'

        # Convert Venchurch to python directives
        py_lines,py_parts = cell_to_venture(cell)
        py_lines = [ py_line.replace('self.v.','self.') for py_line in py_lines]

        # Use line input to determine no_ripls
        try: no_ripls = int(line)
        except: no_ripls = 1
        print 'using %i explanations' % no_ripls

        # call multiprocess inference and plotting on the ripls
        no_sweeps_list = [50] + ( [60] * (no_ripls-1) )
        infer_msg = 'runFromConditional'                                        
        plot_msg = None                                
        hists = multi_ripls(no_ripls,build_class(py_lines),no_sweeps_list,infer_msg,plot_msg)
        for h in hists:
            try: h.label = hists[0].label
            except: print 'Error calling hist.lable'
        return hists
                            


    
    
    
    
    ## for ipythonNB, remove function defintion and uncomment following two lines
def load_ipython_extension(ip):
    """Load the extension in IPython."""
    ip.register_magics(ParaMagics)
    print 'loaded ParaMagics'
try:
    ip = get_ipython()
    load_ipython_extension(ip)
except:
    print 'failed to load'
    #     ip.register_magics(VentureMagics)
#     ip_register_success = 1

# except:
#     print 'ip=get_ipython() didnt run'   

# if found_venture_ripl==1: print 'VentureMagics is active: see %vl? for docs'
