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
