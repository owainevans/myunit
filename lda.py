from venture.venturemagics.ip_parallel import *
from venture.venturemagics.reg_demo_utils import *
import time
def multi(dist): return np.argmax(np.random.multinomial(1,dist))
def normalize(lst):
    s=float(sum(lst))
    return [el/s for el in lst] if s>0 else 'negative sum'
ro = lambda ar: np.round(ar,2)


def mk_lda(clda=True,no_topics=5,size_vocab=20,
           alpha_t_prior='(gamma 1 1)',alpha_w_prior='(gamma 1 1)'):
    lda='''
    [assume topics %i]
    [assume vocab %i]
    [assume alpha_t %s ]
    [assume alpha_w %s ]
    [assume doc (mem (lambda (doc_ind) (make_sym_dir_mult alpha_t topics) ) )]
    [assume topic (mem (lambda (topic_ind) (make_sym_dir_mult alpha_w vocab) ) )]
    [assume word (mem (lambda (doc_ind word_ind) ( (topic ((doc doc_ind)) ) )  ) ) ]
    ''' % (no_topics,size_vocab,alpha_t_prior,alpha_w_prior)

    ulda='''
    [assume topics %i]
    [assume vocab %i]
    [assume alpha_t %s ]
    [assume alpha_w %s ]
    [assume doc (mem (lambda (doc_ind) (symmetric_dirichlet alpha_t topics)))]
    [assume topic (mem (lambda (topic_ind) (symmetric_dirichlet alpha_w vocab) ) )]
    [assume z (mem (lambda (doc_ind word_ind) (categorical (doc doc_ind)) ) ) ]
    [assume word (mem (lambda (doc_ind word_ind)
      (categorical (topic (z doc_ind word_ind) ) ) ) ) ] 
    ''' % (no_topics,size_vocab,alpha_t_prior,alpha_w_prior)
    return lda if clda else ulda

def test_lda():
    def gen_data(lite=0,clda=1):
        vs=[mk_c() for i in range(2)] if not(lite) else [mk_l() for i in range(2)]
        no_topics=2;size_vocab=20; nt=no_topics; sv=size_vocab
        vs[0].execute_program(mk_lda(clda=clda,no_topics=nt,size_vocab=sv,
                                     alpha_w_prior='.5') )
        vs[1].execute_program(mk_lda(clda=clda, no_topics=nt,size_vocab=sv,
                                     alpha_w_prior='.001') )
        no_docs = 5; words_per_doc = 300; corpus={}
        for i,v in enumerate(vs):
            corpus[i]=[]
            for doc in range(no_docs):
                p=[v.predict('(word %i %i)' % (doc,ind) ) for ind in range(words_per_doc)]
                corpus[i].append(p)
        
        assert 3>np.mean(np.abs((np.mean(corpus[0],axis=1) - np.mean(range(sv))))) 
        assert sum(np.std(corpus[1],axis=1)) < sum(np.std(corpus[0],axis=1))
        
        if not(clda):
            uni = np.array( vs[0].sample('(topic 0)') )
            concen = np.array( vs[1].sample('(topic 0)') )
            
            assert .1 > np.mean( np.abs( uni - 1./size_vocab) )
            assert any(.85 < np.abs( concen - 1./size_vocab) )
             
            print [v.sample('(topic 0)') for v in vs]
            print [v.sample('(doc 0)') for v in vs]
    
    gen_data(lite=0,clda=1)
    gen_data(lite=0,clda=0)
    gen_data(lite=1,clda=1)
    return None

def no_x(n,x): return str(int(n)).count(str(int(x)))

def gen_zdocs(half_doc_length=20,no_docs=5,size_vocab=50,no_topics=2,easy=0):
    N=half_doc_length
    topics=[]
    for k in range(no_topics):
        l=[1.+np.mod(i+k,size_vocab)**4 for i in range(1,size_vocab+1)]
        topics.append( normalize(np.array(l)**(-1)) )
        
    #topics[1]= normalize([1./abs(((size_vocab+1)-i)) for i in range(1,size_vocab+1)])
    docs=[]; docs_topics=[]
    for doc in range(no_docs):
        s,t =  np.random.randint(0,no_topics,2)
        words = [multi(topics[s]) for rep in range(N)] + [multi(topics[t]) for rep in range(N)]
        docs.append( words )
        docs_topics.append( (s,t) )
    
    if easy: 
        topics=map(normalize,( [10,10] + [1]*(size_vocab-2), [1]*(size_vocab-2) + [10,10] ) ) 
        docs = [ [0,1]*N, [size_vocab,size_vocab-1]*N ] * no_docs
        docs_topics = []
    
    out = {'topics':topics,'docs':docs,'docs_topics':docs_topics}
    print 'docs sample of doc0: ',out['docs'][0][:10]
    print 'docs sample of doc1: ',out['docs'][1][:10]
    return out

def plot_disc(xs,clist=[],title=None):
    #if not xs: xs=np.arange(len(clist[0]))
    fig, ax = plt.subplots(len(clist),1,figsize=(15,3*len(clist)))
    width=0.4
    for i,c in enumerate(clist):
        xs=range(len(c))
        ax[i].bar(xs,c, width,label=str(i))
        ax[i].set_xticks(xs)
        ax[i].set_xticklabels( map(str,xs) )
        ax[i].legend()
        if title: ax[i].set_title(title)
    fig.tight_layout()
    return fig

def plot_docs(dlst,no_bins=20):
    fig,ax = plt.subplots(1,2,figsize=(14,4))
    xr=range(min(if_lst_flatten(dlst)),max(if_lst_flatten(dlst)))
    counts = [np.histogram(d,bins=xr,density=True) for d in dlst]
    [ax[0].hist(d,bins=no_bins,alpha=0.3,label=str(i)) for i,d in enumerate(dlst)]
    [ax[1].bar(xr,counts[i],width,label=str(i)) for i,d in enumerate(dlst)]
    [ax[i].legend() for i in [0,1]]
    return fig

def inf_til(v,n=2000,t=120):
    st=time.time(); el = 0
    while el<t:
        v.infer(n); print display_logscores(v)
        el = time.time() - st
        
        
    ## INFERENCE
def lda_inf():
    no_topics=4;size_vocab=20; nt=no_topics; sv=size_vocab
    no_docs = 6; half_doc_length=10
    clda=False
    v=mk_p_ripl()
    v.execute_program(mk_lda(clda=False,no_topics=nt,size_vocab=sv,
                         alpha_t_prior='(gamma 1 1)',alpha_w_prior='(gamma 1 1)'))
    
    out = gen_zdocs(half_doc_length=half_doc_length,
                    no_docs=no_docs,size_vocab=sv,no_topics=nt,easy=0)
    topics = out['topics']; docs =  out['docs']; docs_topics = out['docs_topics']
    
    # [v.observe('(word %i %i)'%(i,j),'1.') for i in range(5) for j in range(5)]
    
    for doc_ind,doc in enumerate(docs):
        for word_ind in range(len(doc)):
            v.observe('(word %i %i)'%(doc_ind,word_ind),'atom<%i>'%doc[word_ind])
    
    inf_til(v,n=2000,t=30)

    inf_topics= [v.sample('(topic %i)'%i) for i in range(no_topics)]
    inf_docs = [v.sample('(doc %i)'%i) for i in range(no_docs)]
    print 'true topics: ', ro(topics)
    print 'inf_topics: ', ro(inf_topics)

    print 'true docs ', ro(docs)
    print 'inf_docs', ro(inf_docs)

    doc_wordhists = [np.histogram(doc,bins=range(size_vocab),density=True)[0] for doc in docs]
    inf_wordhists=[]
    for i in range(no_docs):
        doc_hist = v.sample('(doc %i)'%i)
        word_hist = np.sum([ doc_hist[topic_ind] * np.array(inf_topics[topic_ind]) for topic_ind in range(no_topics)],
                           axis=0)
        assert .05 > abs(1-sum(word_hist))
        inf_wordhists.append(word_hist)
    print 'doc_wordhist: ', ro(doc_wordhists)
    print 'inf_wordhist: ', ro(inf_wordhists)
    

    
    # now sample words from topics, and convert to density
    # inf_topics = []; repeats=400
    # fig,ax = plt.subplots(1,2)
    # for topic_ind in range(no_topics):
    #     t_new = [v.predict('( (topic %i) )'%topic_ind) for i in range(repeats) ]
    #     ax[topic_ind].hist(t_new)
    #     t_probs = np.histogram(t_new,bins=range(size_vocab),density=True)[0]
    #     inf_topics.append(t_probs)
    
    # doc0_new = [v.predict('(word 0 %i)'%ind) for ind in range(1000,1100) ]
    
    # plot_disc(0,topics,title='True topics')
    # plot_disc(0,inf_topics,title='Inf topics')
    
    # fig,ax = plt.subplots(1,2)
    # ax[0].hist(docs[0],c='m'); ax.set_title('true doc 0')
    # ax[1].hist(docs[1],c='y'); ax.set_title('inf doc 0')
    
    
    return v, out, inf_topics

    # doc0_new = []; topic_gen=[]
    # for v in vs:
    #     d_new= [v.predict('(word 0 %i)'%i) for i in range(20,30)]
    #     d_probs = np.histogram(d_new,bins=range(size_vocab),density=True)
    #     doc0_new.append( {'hashv':hash(v),
    #                       'doc0_draws':d_new, 'doc0_probs':d_probs} )

    #     topic_gen.append({'hashv':hash(v)})
    #     for topic_ind in range(no_topics):
    #         t_new = [v.predict('( (topic %i) )'%topic_ind) for i in range(30) ]
    #         t_probs = np.histogram(t_new,bins=range(size_vocab),density=True)
    #         topic_gen[-1].update({'topic%i_draws'%topic_ind :t_new,
    #                               'topic%i__probs'%topic_ind :t_probs} )
    # #print 'doc0 data:',docs[0],'doc0_new: ', doc0_new
    # #print 'topic0 data:',topics[0],'topics0_new: ', topic0

    # new_docs=[]
    # for v in vs:
    #     new_docs.append( [v.predict('(word 3 %i)'%i) for i in range(30)] )    
    # #print new_docs



# Starting a real-data run would be fine but I would be very surprised if it worked.

# I think priority 1 should be getting ventureunit to run acceptably from the notebook (or getting a good way to browse its output via terminal, for example by using the built-in browser).

# VentureUnit is precisely designed to make this kind of exploration easy, and to record enough that you can distinguish signal from noise. Do you know the workflow? Samples from the prior, Geweke, running conditioned from the prior (titrating amount of data and computation, or revising model, until "it works"), then running conditioned in data is probably what you want.

# Stick with the uncollapsed model for now; the collapsed model will reject continuous parameter changes a lot in the very sparse case. So you might have success w uncollapsed only by generating data from a sparse model but doing inference using a model that has a very broad prior.

# One useful diagnostic would be to plot the predictive doc/word frequencies and compare them to the observed ones. That should decrease for the synthetic stuff.

# For complexity, I'd start with:

# - Small #T (with very sparse doc-topic; only a few topics per document)
# - Small #W (with very sparse topic-word; only a few words per topic)
# - Small N, D --- enough so that you 

# Then I'd use plot_asymptotics to get a sense of runtime scaling and be able to estimate the time for larger groups of runs.

# Then I'd increase N and D, then T, then W staying synthetic, keeping things sparse.

# If that all works, then I might try generating data from noisier topics (perhaps doing it manually, outside of VentureUnit), with about 100 vocab, 10 topics, 100 docs, 30 words per doc, and retaining sparse doc-topic params.

# If all that works, then I'd try psych review.

# I'd consider using pgibbs for the hyperparams (and/or a broader hyper-prior, and/or slice if you're doing this in puma); this will make a big difference.

# Using enumerative_gibbs for the symmetric dirichlet discretes might also be good.

# I'd email Vlad, Alexey and Selsam about this as you go along.






# err = [3]*len(zipf1)
# fig, ax = plt.subplots()
# width=0.35
# ax.bar(np.arange(len(zipf1)), doc1_counts, width, color='r', yerr=err,label='doc1')
# ax.bar(np.arange(len(zipfsv))+width, doc2_counts, width, color='y', yerr=err,label='doc2')
# ax.set_title('Scores by group and gender')
# ax.set_xticks(np.arange(len(zipf1))+width)
# ax.set_xticklabels( map(str,range(len(doc1_counts))) )
# ax.set_ylim(0,max(doc1_counts)+10)
# ax.legend()
