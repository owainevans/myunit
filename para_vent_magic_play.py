from IPython.core import display
import numpy as np  
from my_unit import *
import os,time,multiprocessing

from IPython.core.magic import (Magics,register_cell_magic, magics_class, line_magic,cell_magic, line_cell_magic)

found_venture_ripl = 0

try: 
    from venture.shortcuts import *
    ipy_ripl = make_church_prime_ripl()
    found_venture_ripl = 1
except:
    try:
        import venture.engine as v2
        print 'found veng'; found_venture_ripl = 1
    except:
        print 'failed to make venture ripl'
        

@magics_class
class VentureMagics(Magics):

    def __init__(self, shell):
        super(VentureMagics, self).__init__(shell)

    @line_cell_magic
    def vl2(self, line, cell=None):
        '''VentureMagics creates a RIPL on IPython startupt called ipy_ripl.
        You can use the RIPL via Python:
           ipy_ripl.assume('coin','(beta 1 1)')
        
        You can also use the RIPL via magic commands. Use %vl for single
        lines:
           %vl [ASSUME coin (beta 1 1)]

        This magic can take Python expansions:
           %vl [ASSUME coin {np.random.beta(1,1)} ]

        Use the cell magic %%vl for multi-line input:
           %%vl
           [ASSUME coin (beta 1 1)]
           [ASSUME x (flip coin)]'''

        def format_parts(parts):
            'format the input string for pretty printing'
            return '[%s]' % ' '.join(parts)
        
        
        ## LINE MAGIC
        if cell is None:
            vouts = ipy_ripl.execute_instruction(str(line), params=None)

            py_lines,py_parts = self.cell_to_venture(line)
            
            for key in py_parts:
                print format_parts(py_parts[key])
                 
            if 'value' in vouts: print vouts['value'].get('value',None)

            return vouts
            
        ## CELL MAGIC    
        else:
            vouts = ipy_ripl.execute_program( str(cell), params=None )

            py_lines,py_parts = self.cell_to_venture(cell)
                              
            for count,v_line in enumerate(vouts):
                print format_parts(py_parts[count])
                if 'value' in v_line: print v_line['value'].get('value',None)

            return vouts
    


    #### FIXME need to work out how long to sleep    
    def makeRipl(self):
        'sleep before making'
        time.sleep(1)
        return make_church_prime_ripl()

    def worker(self,out_q,ModelClass,ripl,params={},infer_msg,plot_msg):
        print 'Starting:', multiprocessing.current_process().name
        
        unit_model = ModelClass(ripl,params)
        if infer_msg=='runFromConditional':
            hist = unit_model.runFromConditional(sweeps=no_sweeps,runs=1)
            out_q.put(hist)
        print 'Exiting:', multiprocessing.current_process().name 


    @cell_magic
    def p(self, line, cell):
        'need to fix OBS issue'
        
        py_lines,py_parts = self.cell_to_venture(cell)

        py_lines = [ py_line.replace('self.v.','self.') for py_line in py_lines]

        class MyModelClass(VentureUnit):
            def makeAssumes(self):
                [eval(py_line) for py_line in py_lines if py_line[5]=='a']

            def makeObserves(self):
                [eval(py_line) for py_line in py_lines if py_line[5]=='o']

        try: no_ripls = int(line)
        except: no_ripls = 1

        print 'using %i explanations' % no_ripls

        
        if 








        return models



        
    
    
                                                                                                                                                                                                                                                            
    def remove_white(self,s):
        t=s.replace('  ',' ')
        return s if s==t else self.remove_white(t)
        
        
    def cell_to_venture(self,s,terse=0):
        """Converts vchurch to python self.v.assume (OBSERVE is broken)"""
        s = str(s)
        s = s[:s.rfind(']')]
        ls = s.split(']')
        ls = [ self.remove_white(line.replace('\n','')) for line in ls]

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
    
    
    
    
    ## for ipythonNB, remove function defintion and uncomment following two lines
def load_ipython_extension(ip):
    """Load the extension in IPython."""
    ip.register_magics(VentureMagics)
    print 'loaded VentureMagics'
# try:
#     ip = get_ipython()
#     ip.register_magics(VentureMagics)
#     ip_register_success = 1

# except:
#     print 'ip=get_ipython() didnt run'   

# if found_venture_ripl==1: print 'VentureMagics is active: see %vl? for docs'
