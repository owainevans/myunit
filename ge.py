from venture.venturemagics.ip_parallel import *
from venture.venturemagics.reg_demo_utils import *

## to add: PP plots
## compare to single chain geweke style

def pp_plot(s1,s2,label1=None,label2=None):
    fig,ax = plt.subplots(figsize=(3,3))
    ax.scatter(sorted(s1),sorted(s2))
    if label1:
        ax.set_xlabel(label1); ax.set_ylabel(label2)
    ax.set_title('PP Plot')


def geweke_sample(r,model,observes,step_size,total_samples,query_exps):
    exp_vals = {exp:[] for exp in query_exps}

    r.execute_program(model)
    obs_did_start = len(r.list_directives())+1 #did start at 1
    
    for sample in range(total_samples):
        data=[r.sample(obs) for i,obs in enumerate(observes)]
        [r.observe(obs,data[i]) for i,obs in enumerate(observes)]
        r.infer(step_size)
        [lst.append(r.sample(exp)) for exp,lst in exp_vals.items()]
        [r.forget(obs_did_start+i) for i,obs in enumerate(observes)]
    return exp_vals


def condition_prior_sample(r,model,no_datasets,gen_data_exp,size_data,no_transitions,query_exps):

    exp_vals = {exp:[] for exp in query_exps}

    for dataset in range(no_datasets):
        r.clear()
        r.execute_program(model)
        
        for point in range(size_data):
            r.observe(gen_data_exp,r.sample(gen_data_exp))
        r.infer(no_transitions)
        [lst.append(r.sample(exp)) for exp,lst in exp_vals.items()]
        
    return exp_vals



def geweke(r,model,observes,step_size,total_samples,query_exps):
    exp_vals = {exp:[] for exp in query_exps}

    r.execute_program(model)
    obs_did_start = len(r.list_directives())+1 #dids start at 1
    
    for sample in range(total_samples):
        data=[r.predict(obs) for i,obs in enumerate(observes)]
        [r.forget(obs_did_start+i) for i,obs in enumerate(observes)]
        [r.observe(obs,data[i]) for i,obs in enumerate(observes)]
        r.infer(step_size)
        [lst.append(r.sample(exp)) for exp,lst in exp_vals.items()]
        [r.forget(obs_did_start+i) for i,obs in enumerate(observes)]
    return exp_vals


def condition_prior(r,model,no_datasets,observes,no_transitions,
                         query_exps):
    
    exp_vals = {exp:[] for exp in query_exps}

    for dataset in range(no_datasets):
        r.clear()
        r.execute_program(model)
        obs_did_start = len(r.list_directives())+1 #dids start at 1
        #data=[r.predict(obs,label='test_'+str(i)+obs) for i,obs in enumerate(observes)]
        data=[r.predict(obs) for i,obs in enumerate(observes)]
        [r.forget(obs_did_start+i) for i,obs in enumerate(observes)]
        [r.observe(obs,data[i]) for i,obs in enumerate(observes)]
        
        r.infer(no_transitions)
        [lst.append(r.sample(exp)) for exp,lst in exp_vals.items()]
        
    return exp_vals



def mr_condition_prior(no_ripls, no_datasets, model,
                       observes, query_exps, no_transitions,**mrkwargs):
    v=MRipl(no_ripls,**mrkwargs)
    all_out = mr_map_proc(v,no_ripls,condition_prior,
                          model,no_datasets,observes,
                          no_transitions,query_exps)
 
    store_exp_vals={exp:[] for exp in query_exps}        
    [store_exp_vals[exp].extend(ripl_out[exp]) for exp in query_exps for ripl_out in all_out]
    return store_exp_vals

def mr_forward_sample(no_samples,model,query_exps,**mrkwargs):
    mr = MRipl(no_prior_samples,**mrkwargs)
    mr.execute_program(model)
    return {exp:mr.sample(exp) for exp in query_exps}

def compare_stats(list_dicts,list_labels=None):
    if not list_labels:
        list_labels = ['Dict %i'%i for i in range(len(list_dicts))]
    print 'Stat, exp, ', ' '.join(list_labels)

    for exp in list_dicts[0].keys():
        lists = [d[exp] for d in list_dicts]
        print '%s means:',np.mean(lists,axis=1)
        print '%s stds:',np.std(lists,axis=1)
        print '----'


# Params    
no_ripls=2; no_datasets=3
size_data=5
model=simple_quadratic_model #x_model_t+quad_fourier_model
gen_data_exp = '(y_x (x_d))'
observes = [gen_data_exp] * size_data
query_exps = ['w0','w1','w2','noise']
no_transitions=100
backend='puma'
local_mode=True; no_local_ripls=no_ripls

# from inference
store_exp_vals_inf = mr_condition_prior(no_ripls,no_datasets,model,
                                        observes,query_exps,no_transitions,
                                        
)
# from prior
no_samples = no_ripls * no_datasets
store_exp_vals_prior = mr_forward_sample(no_samples,model,query_exps)

# from geweke
total_samples = no_ripls * no_datasets
step_size=10
#store_exp_vals_geweke = geweke(mk_p_ripl(),model,observes,
#                               step_size,total_samples,query_exps)

assert len(store_exp_val_inf['w0'])==len(store_exp_vals_prior['w0'])
#assert len(store_exp_val_geweke['w0'])==len(store_exp_vals_prior['w0'])

compare_stats([store_exp_vals_inf,store_exp_vals_prior],['inf','prior'])


















def con_prior(r,no_datasets,gen_data_exp,size_data,no_transitions,vars):
    for i in no_datasets:
        data=[]
        for data_point in range(size_data):
            data.append(r.predict(gen_data_exp,label=str(datapoint)))

        for data_point in data:
            r.observe(gen_data_exp,label=str(datapoint))
            








# # simple version
# for i in no_datasets:
#     datasets = [r.predict(gen_data) for ripl in no_ripls for i in size_data]

#     mr_map_array(v,lambda r,exp,val:r.observe(exp,val),dataset[
