In [4]: %%v
   ...: [assume ar_eq (lambda (a b n) (if (< n 0) true
                              (and (= (lookup a n) (lookup b n) )
                                   (ar_eq a b (- n 1)) ) ) ) ]
[predict (ar_eq (array 1) (array 1) 0 ) ]
[predict (not (ar_eq (array 1 2) (array 3 2) 1 )) ]
[predict (not (ar_eq (array 1) (array 2) 0 )) ]
   ...:    ...:    ...:    ...:    ...:    ...: 
[directive_id: 1].  assume ar_eq   =  sp 
[directive_id: 2].  predict (ar_eq ... )   =  True 
[directive_id: 3].  predict (not ... )   =  True 
[directive_id: 4].  predict (not ... )   =  True 
Out[4]: 
(['sp', True, True, True],
 [{'directive_id': 1, 'value': {'type': 'sp', 'value': 'sp'}},
  {'directive_id': 2, 'value': {'type': 'boolean', 'value': True}},
  {'directive_id': 3, 'value': {'type': 'boolean', 'value': True}},
  {'directive_id': 4, 'value': {'type': 'boolean', 'value': True}}])

In [5]: ipy_ripl.clear()

Process Python segmentation fault (core dumped)

--------------------------------
really big definition results in crash



---------------------
Cleanup mismatch applyPSP
8 4
python: backend/cxx/src/trace.cxx:108: Trace::~Trace(): Assertion `false' failed.

Process Python aborted (core dumped)




---------------
In [5]: v.predict('(range (power 10 6))')
Segmentation fault (core dumped)
