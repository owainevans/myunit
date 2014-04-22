from venture.venturemagics.ip_parallel import *
from venture.venturemagics.reg_demo_utils import *

def condition_prior_pred(r,model,no_datasets,observes,no_transitions,
                         query_exps):
    # assumes 
    exp_vals = {exp:[] for exp in query_exps}

    for dataset in range(no_datasets):
        r.clear()
        r.execute_program(model)
        
        data=[r.predict(obs,label=str(i)+obs) for i,obs in enumerate(observes)]
        [r.forget(str(i)+obs) for i,obs in enumerate(observes)]
        [r.observe(obs,data[i]) for i,obs in enumerate(observes)]
        
        r.infer(no_transitions)
        [lst.append(r.sample(exp)) for exp,lst in exp_vals.items()]
        
    return exp_vals


def condition_prior(r,model,no_datasets,gen_data_exp,size_data,no_transitions,query_exps):

    exp_vals = {exp:[] for exp in query_exps}

    for dataset in range(no_datasets):
        r.clear()
        r.execute_program(model)
        
        for point in range(size_data):
            r.observe(gen_data_exp,r.sample(gen_data_exp))
        r.infer(no_transitions)
        [lst.append(r.sample(exp)) for exp,lst in exp_vals.items()]
        
    return exp_vals

pred=1
no_ripls=2; no_datasets=5

v=MRipl(no_ripls)
model=simple_quadratic_model #x_model_t+quad_fourier_model

gen_data_exp = '(y_x (x_d))'; size_data= 5
if pred:
    observes = [gen_data_exp] * size_data
    
query_exps = ['w0','w1','w2','noise']
no_transitions=300

if not(pred):
    all_out = mr_map_proc(v,2,condition_prior,model,
                  no_datasets, gen_data_exp, size_data,
                          no_transitions, query_exps)
else:
    all_out = mr_map_proc(v,2,condition_prior_pred,
                          model,no_datasets,observes,
                          no_transitions,query_exps)
                  
store_exp_vals={exp:ripl_out[exp] for exp in query_exps for ripl_out in all_out}


# from prior
no_prior_samples = no_ripls * no_datasets
mr = MRipl(no_prior_samples)
mr.execute_program(model)
store_exp_vals_prior = {exp:mr.sample(exp) for exp in query_exps}


for exp,vals in store_exp_vals.items():
    prior_vals=store_exp_vals_prior[exp]
    print exp
    print 'inf mean std:', np.mean(vals),np.std(vals)
    print 'prior mean std:', np.mean(prior_vals),np.std(prior_vals)








def condition_prior(r,no_datasets,gen_data_exp,size_data,no_transitions,vars):
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
