# ASANA

## GET SEED GET SEED!!!
# ALSO: seed fail for lite in ventureunit
# ValueError                                Traceback (most recent call last)
# <ipython-input-247-3f79dce4f0c0> in <module>()
# ----> 1 hi,r=ana.runFromConditional(10,runs=2)

# /home/owainevans/myunit/unit/analytics.py in runFromConditional(self, sweeps, name, data, **kwargs)
#     502                history.History with nameToSeries dictionary of runs*Series
#     503                for each recorded expression.
# --> 504            ripl :: ripl
#     505                Pointer to self.ripl, i.e. a ripl with same backend as given to
#     506                constructor, mutated by assumes,observes (with values given by

# /home/owainevans/myunit/unit/analytics.py in _runRepeatedly(self, f, tag, runs, verbose, profile, **kwargs)
#     395 
#     396         return iterations
# --> 397 
#     398     # TODO: run in parallel?
#     399     def _runRepeatedly(self, f, tag, runs=3, verbose=False, profile=False, **kwargs):

# /home/owainevans/myunit/unit/analytics.py in runFromConditionalOnce(self, data, **kwargs)
#     516             # but not if user calls. Does this lead to any problems?
#     517             data = [(exp,datum) for (exp,_),datum in zip(self.observes,data)]
# --> 518         else:
#     519             data = self.observes
#     520         history.addData(data)

# /usr/local/lib/python2.7/dist-packages/venture/ripl/ripl.pyc in clear(self)
#     248     def clear(self):
#     249         s = self._cur_parser().get_instruction_string('clear')
# --> 250         self.execute_instruction(s,{})
#     251         return None
#     252 

# /usr/local/lib/python2.7/dist-packages/venture/ripl/ripl.pyc in execute_instruction(self, instruction_string, params)
#      69         try:
#      70             # execute instruction, and handle possible exception
# ---> 71             ret_value = self.sivm.execute_instruction(parsed_instruction)
#      72         except VentureException as e:
#      73             # all exceptions raised by the Sivm get augmented with a

# /usr/local/lib/python2.7/dist-packages/venture/sivm/venture_sivm.pyc in execute_instruction(self, instruction)
#      61                 f = getattr(self,'_do_'+instruction_type)
#      62                 return f(instruction)
# ---> 63             return self._call_core_sivm_instruction(instruction)
#      64 
#      65     ###############################

# /usr/local/lib/python2.7/dist-packages/venture/sivm/venture_sivm.pyc in _call_core_sivm_instruction(self, instruction)
#     121             desugared_src_location['expression_index'] = new_index
#     122         try:
# --> 123             response = self.core_sivm.execute_instruction(desugared_instruction)
#     124         except VentureException as e:
#     125             if e.exception == "evaluation":

# /usr/local/lib/python2.7/dist-packages/venture/sivm/core_sivm.pyc in execute_instruction(self, instruction)
#      42         utils.validate_instruction(instruction,self._implemented_instructions)
#      43         f = getattr(self,'_do_'+instruction['instruction'])
# ---> 44         return f(instruction)
#      45 
#      46     ###############################

# /usr/local/lib/python2.7/dist-packages/venture/sivm/core_sivm.pyc in _do_clear(self, _)
#     140     def _do_clear(self,_):
#     141         utils.require_state(self.state,'default')
# --> 142         self.engine.clear()
#     143         self.observe_dict = {}
#     144         return {}

# /usr/local/lib/python2.7/dist-packages/venture/engine/engine.pyc in clear(self)
#     101     # method often.
#     102     import random
# --> 103     self.set_seed(random.randint(1,2**32-1))
#     104 
#     105   # Blow away the trace and rebuild one from the directives.  The goal

# /usr/local/lib/python2.7/dist-packages/venture/engine/engine.pyc in set_seed(self, seed)
#     176 
#     177   def set_seed(self, seed):
# --> 178     self.trace.set_seed(seed)
#     179 
#     180   def continuous_inference_status(self):

# /usr/local/lib/python2.7/dist-packages/venture/lite/trace.pyc in set_seed(self, seed)
#     371   def set_seed(self, seed):
#     372       random.seed(seed)
# --> 373       numpy.random.seed(seed)
#     374 
#     375   def getGlobalLogScore(self):

# /usr/lib/python2.7/dist-packages/numpy/random/mtrand.so in mtrand.RandomState.seed (numpy/random/mtrand/mtrand.c:4941)()

# ValueError: object of too small depth for desired array



# [assume x (binomial 1 .00001)]
# [observe (poisson x) 1]
# [infer 1]
# --core dump in puma

# v.assume('dir_mult','(make_dir_mult (array 1 1))')
# == 'unknown'
# -- should be something like (see sym_dir_mult value)
# {'simplex': [1.0,1.0], 'counts': [0, 0],  'type': 'dir_mult'}


# v.assume('sym_dir','(make_sym_dir_mult 1 2)')
# v.observe('(sym_dir)','atom<0>')
# v.list_directives()[-1]
# {'directive_id': 17,
#  'expression': ['s'],
#  'instruction': 'observe',
#  'value': 0.0}

# v.assume('atom_or_num','(lambda ()(if (flip theta) atom<0> 0))')
# v.observe('(atom_or_num)','0')
# v.infer(1)
# RuntimeError: Cannot constrain a deterministic value.
# --not sure if there's a way to observe an atom noisily. want sp
#   where input and output are atoms. 

# --can't recover fact that value we conditioned on was atom
