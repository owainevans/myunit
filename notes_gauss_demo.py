0. For the raw data for gaussians, how about we use the height data from here:

    http://www.math.hope.edu/swanson/data/children_heights.txt

    Initially we can use 18 year old boys only, but to add a case where we need a crp mixture, we can try putting the girls in too, or perhaps all of the ages.

1. Can we use a broad scale that spans 1 order of magnitude for the initial prior, on both mean and variance? That shouldn't make plots uninterpretable, and is somewhat intuitive --- we know roughly how tall people are, but maybe not much else.

*2. Can we have a single mripl with 100 samples set in the very beginning, and just reuse it throughout, with magics, clearing in between each problem? I think that'll be more intuitive for users --- less switching in and out of python.

*3. Can we visualize the mu/sigma pairs in scatter and heatmap form, too? So they can see that there's no dependence between the variables.

4. Can the first version not reify the x_i, but instead work with (normal mu sigma) directly? The indirection may be confusing, and actually corresponds to an interesting modeling choice. If all we have is a fixed population, then all we've got is a single Gaussian. Also this'll serve to clarify the meaning of observe and predict.

5. I'd refactor the population sampler ( in[31] with the mr_map ) to extract the samples:

   def generate_synthetic_population(ripl):
        return [ripl.sample("(normal mu sigma)") for i in range(200)]

   (mr_map that procedure, and then plot all the populations on the same axes, so you see the variability visually, just with a single hist:

6. Can we add datapoints one at a time? So after we visualize the prior and the populations it licenses, we can see how rapidly Bayes fixes the scale to be roughly right --- it should only take ~3 datapoints. This is worth titrating a bit to get it to look "really right" numerically.

7. Then we can show inference from the full population.

8. Then we should have a couple instances where we break it:

    - Fix the variance to be way high and repeat the diagnostics
    - Fix the variance to be way low
    - Fix the mean to be very wrong, and see how it has to infer the variance to be high
    - Add in the girls, and say OK, how could we deal with this? We'll show how later.

9. For the constant function, can we remove the model for x, collapse y_x into the function y, and call noise "amount_of_noise"? Also what's the purpose of n?

(I really like going from constant to linear as just making a small modification.)

10. Can we have them write the code to probe x along a range, and make a plot of the predictive? That's teaching them something valuable. We can supply (but show) the code snippet for plotting polynomials themselves, overlaid with the raw data.

11. I would try to show posteriors for a few unambiguous datasets --- constant at 3, little noise; slope 1, little noise; slope -1, medium noise --- and then also show an ambiguous dataset where noise and positive slope are confusable. Titrating the priors so that really works --- and is evident in a plot where the raw data and the curves are overlaid --- would again teach them quite a lot.

*12. I think we should suppress the code for generating the synthetic datasets we use, and instead just provide them as named variables. 

summary plan:
1. implement venchurch cell magic
%%venture mripl_name
[assume x 1]

2. tricky coin. use snapshot. infer on is_tricky and weight of coin. uniform prior on coin bias.

3. prior: use gaussian on mu centered on zero and with big variance. also high variance distribution on sigma (e.g. gamma(.5 .1) or uniform_contious(.01 100)).

4. add data points one at a time. plot mu and sigma, plot posterior on x. then do inference on whole data set.

5. allow user to put in different prior on variance, or to just fix the variance to a bad constant value. if variance very high, then a few points won't change distribution on mu much or much shift predictive. variance too low, we'll do poorly on inference of variance, but rest should be ok. if mean way off and prior confident, then we'll explain data by high variance.

6. re: 10 above about looking at prediction in range. we can do snapshot( exp_list='(y_x  x_value)'), which will be a hist of P(y / X=x_value). we can then compare 'in-sample' vs. 'out-of-sample' x's and see that at least in the quadratic case, we'll have more confidence prediction close to sample.

i guess we could do y conditional on x being in a given range by saying observe('(< (x index) 3','), sample('(y index)'), but i'm not sure that's what you mean. 

7. do model selection with const/linear/quadratic. 


<code>#%%mr_map v plot_population_dist st 2
def plot_population_dist(ripl):
    xs=[ripl.sample('(x %i)'%index) for index in range(200)]
    mu=ripl.sample('mu'); sigma = ripl.sample('sigma')
    fig,ax = plt.subplots(figsize=(5,2))
    ax.hist(xs,bins=20)
    ax.set_title('Population distribution on heights (mu=%f, sigma=%f)' %(mu,sigma))
    return None #'Sample heights: ', xs[:5]</code>



def sample_flips(v):
    return v.sample('is_tricky'), [v.sample('(flip theta)') for i in range(8) ]

def print_worlds(v):
    ls=mr_map_f(v,sample_flips,limit=4)['out']
    for a,b in ls:
        print 'is_tricky=',a,' Flips:', b
        
print_worlds(v)



## FIX PLOTTING LIMITS
optional argument for snapshot
posterior -> posterior_samples

snapshot(take previous snapshot as optional argument, feed back vals and then get 

gather populations.

single procedures: either sample a bunch of populations or integrate over them. 

        v.sample('(x 0)')   from prior
is set of draws from different population

v.sample('(list (x 0)  ... (x n))') from prior
returns a list of lists, one for each mripl, which are a list of sample populations we can plot as overlaid histograms

flatten(v.sample('(list ..

snapshot( 

##CANT DO INFER

rough plan:
   sample predictive is just add more ripls. but better to take extra samples. could have something like v.snapshot(exp_list=['x'],population=True,grouped=False)
or something. note that imp thing is to have small number of populations to get a sense of what they look like, and to be sure to use all ripls when you want to get a sense of the predictive. (difference between mapping across and collecting). 

so one way is just with mr_map, but with plots collected by getting data back
plot(mr_map_f)

vs. snapshot(predictive(exp))






####
notes on what we need for cambridge:

schedule: i'm leaving AM wednesday. have thu-sunday to work on this full-time. will be 3.5 hrs travel away from cam. could arrive whenever. 

systems: yura, dlovell, alexey? terminal and os x install.


lite vs. puma:
1. lite lacks 'nil' for '(list)'
2. lite bernoulli outputs bool and puma outputs int. both should output int (as flip already outputs bool). 
3. give lite and puma ripls doc_strings so you can find out straightforwardly what kind of ripl you have
4. lite 'if' can't take numbers or lists. puma 'if' can take numbers (and '0' is false) but can't take lists. should be consistent. 
    lite 'or/and/not' can't take numbers but puma can. (my sense is both should take numbers). 
5. lite 'size' outputs python int, puma outputs float.
6. lite outputs 'venture.lite.valueSPRef' for sp, puma 'unknown'.
7. lite outputs a numpy array for (dirichlet (array 1 1)), puma 
8. (is_atom (binomial 10 .5)) is false for lite, true for puma (but categorical and crp do have values that are atoms in lite).
9. lite exceptions are very long. is there helpful information there or should they just skip to the assert?
10. puma can test arrays for equality with '=', lite can't. in neither puma nor lite can you say 'observe (array (flip)) (array true)', but in puma you can get around this by saying 'observe (= (array (flip)) (array true)) true', but not in lite.  
11. how do variable-length arguments work for functions?
12. FORCE doesn't work in lite (at least on a cursory look)




notes vikash: bug in the paper re: dynamic scopes. q: general set of tools for dealing with sets of ripls. two main
situations, mripls, with identical model, which can be used for estimating posterior and where you want to do
snapshot and form hists. then model comparison (or backend comparison, parametric variation) where you don't wanna collapse data into one hist, but you do want to map functions across all the models to amke comparisons. 

even without parallel, we'd like to create mripls for convenience of avoiding loop constructs. if we don't have to deal with mripls, we can simplify things. maybe some way of using inheritance to do this. big thing is that we couldn't copy the state of a chain before due to serialization. but if all mripls are local, it should be easier to copy them. then we can do things like take a chain we've been running and let that be the seed for a bunch of mripls. with a local mripl, it's easy to map functions across them (without worrying about namespace issues). and we can map across them and get figure objects that we can then interactively edit. 

plan: make a local mripl constructor, which sets the seeds as in the remote mripl. (so we can use it for debugging). code will be very simple:
self.assume(x,y) return [r.assume(x,y) for r in mripls]. 

so the mripl is an object that just stores the ripls, the seeds for the ripls, and has a bunch of shadow methods. 

convenience functions snapshot, plot_past_vals, plot_populations all go through a call to self.sample or self.predict. and so if we redefine these to work in the local_mripl case, they'll work just as well. 

venture wishlist:
1. exceptions (also want to avoid whatever kills engines)
2. let, quote syntactic sugar
lite needs to have poisson, thing with flip. map_list
3. poisson, mvn and wishart, student t should have mean, variance args,
    pitman yor?, uniform_discrete for lite
4. map,list2array, list2dict, apply
5. standard library on master:
    repeat, length(list), append, fold, reverse
    sum(lst/array), prod(lst/array)

6. v.observe('(list (x) (y))','(list 1 2)')
 

multiripl:

concrete short-term things:
ability to swith from lite to puma (having to redo inference). lite is the reference implementation. puma

1. check again for lack of interactive graphics in matplotlib
2. for topic models and IRM, we'll want visualization of discrete variables. examples: number of clusters for IRM, number of topics for topic models. need general facility for 1D and 2D discrete variables. also: what about plotting simplex points? and the 2d plot for cluster membership we had in matlab. 
3. sample_populations: we can always take a thunk and do mr_map('(repeat n thunk)', limit=some_limit). for 1D cts variable we can then plot on same axis, but in general we can just produce a set of plots. so we need to modify this, with an important special case for discrete and cts 1D variables that can be put on the same plot.
4. plot_old_vals: similarly, we need to consider the more general case where you can just plot in subplot side-by-side with a shared x and y and good labeling. 
  titles and legend cannot always easily be added later: you might have lots of plots and not be easily able to keep track of which is which. (and the mripl plot function might want to alter order somehow). this makes modularity trickier. in terms of look of plots, there seems to be some way of fixing general settings, coz seaborn did that. 
5. can you get plot to do individual plots and then subplot the figs you get back? otherwise, we'd have to use plot to do the whole subplot. maybe that's ok, if we can customize the plot with a special title etc. before it comes back, but this may be a problem (maybe plot gets to plot it inline?). 

6. saving files, saving ripls?


longer-term issues:

--can it be refactored somehow to abstract all the shared structure? maybe some kind of inheritance from a normal ripl?

--add something for easily loading parametrically varying models over a set of ripls (either an mripl, set of mripls, or set of single ripls). want to be able to keep track of what the starting params were.

--one would like to be able to map a set of observes over a set of ripls/set of mripls. [note the importance here of the identical methods syntax for ripls and mripls]. a general way for this to work is that you have a thunk '(draw sample)', which is always generating exchangeable data. then you can say:

asssume draw_data (lambda () (let ( [x (x_d)]] ) (list x (y_x x))))
batch_observe(vs,data):
    [v.observe('(draw_data)','%s' % str(datum)) for datum in data]

but note that this requires us to always use thunk for generating data.
this seems reasonable for most of the modeling we might do, and the 
tutorials could be build around this assumption. we could have an 
optional argument where you specify the two arguments to observe.


--we want to do bulk infer, with arbtirary inference specificiation

--other standard features: show changing posterior as data comes in, show changing state of chain as no_samples goes up (probes). these are things that can only apply to an mripl (where you have immediate access to a snapshot of the posterior). if we apply this across multiple models, then we are doing a kind of hybrid thing (our meta-model is weird python-venture mix and we don't do inference in it, we just visualize it). 

distinguish: snapshot is something that doesn't make sense for model comparison. we want to snapshot only for an mripl. 
function like observe_infer, which maps the same observe/infer operation across all models. (the plotting tools we have, e.g. plot_conditional, work in the same way).

function like: compare_logscore. we take the mean logscores of models and compare them. we could also select the best scoring ripl for each model and compare the likelihood each ripl gives to the data. or compare the KL-divergence between this ripl and the data. these involve computing on each model and then computing some statistics of the output from each model. 
   --can have a more general mr_map_vs(vs,f,store,limit), which takes set of ripls or mripls, and either does mr_map, or just applies f to a single ripl. 







gen model for dangerous equation:

as student_score mem la i (sum (student_default i) (school_size (school i)))))
observe (mean stud_score_1 stud_score_2 ...) 45)

or:
assume school_score(i) sum(students 3 4 5)
assume school_size(i) ?round a log_normal
then want to look at the correlation between 
