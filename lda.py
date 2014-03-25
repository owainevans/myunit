from venture.venturemagics.ip_parallel import *
from reg_demo_utils import *
import time
def multi(dist): return np.argmax(np.random.multinomial(1,dist))
def normalize(lst):
    s=float(sum(lst))
    return [el/s for el in lst] if s>0 else 'negative sum'
assert multi([.999, .001])==0



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
        # .001 is high concentration
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


def gen_zdocs(doc_length=20,no_docs=5,size_vocab=50,no_topics=2):
    N=doc_length
    topics=[]
    for k in range(no_topics):
        l=[1.+np.mod(i+k,size_vocab) for i in range(1,size_vocab+1)]
        topics.append( normalize(np.array(l)**(-1)) )
        
    topics[1]= normalize([1./abs(((size_vocab+1)-i)) for i in range(1,size_vocab+1)])
    docs=[]; docs_topics=[]
    for doc in range(no_docs):
        s,t =  np.random.randint(0,no_topics,2)
        words = [multi(topics[s]) for rep in range(doc_length)] + [multi(topics[t]) for rep in range(doc_length)]
        docs.append( words )
        docs_topics.append( (s,t) )

    return {'topics':topics,'docs':docs,'docs_topics':docs_topics}


# SIMPLE TOPICS
# topics = [ [.5]*2 + [0.001]*8, [0.001]*8 + [.5]*2 ]
# s=30; docs = [ [0]*s +[1]*s, [9]*s + [8]*s, [9]*s + [8]*s ]

def plot_disc(xs,clist=[],title=None):
    if not xs: xs=np.arange(len(counts))

    fig, ax = plt.subplots(len(clist),1,figsize=(15,3*len(clist)))
    width=0.4
    for i,c in enumerate(clist):
        ax[i].bar(xs,c, width,label=str(i))
        ax[i].set_xticks(xs)
        ax[i].set_xticklabels( map(str,xs) )
        ax[i].legend()
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

    ## INFERENCE
def lda_inf():
    no_topics=2;size_vocab=50; nt=no_topics; sv=size_vocab
    
    v=mk_c()
    v.execute_program(mk_lda(alpha_t_prior='(gamma 1 1)',
                                 no_topics=nt,size_vocab=sv))
    
    out = gen_zdocs(doc_length=20,no_docs=5,size_vocab=sv,no_topics=2)
    topics = out['topics']
    docs =  out['docs']
    docs_topics = out['docs_topics']
    

    for doc_ind,doc in enumerate(docs):
        for word_ind in range(len(doc)):
            v.observe('(word %i %i)'%(doc_ind,word_ind),'%i'%doc[word_ind])
    print logscores(v)
    v.infer(5000)
    print logscores(v)

    # now sample words from topics, and convert to density
    inf_topics = []; repeats=200
    for topic_ind in range(no_topics):
        t_new = [v.sample('( (topic %i) )'%topic_ind) for i in range(repeats) ]
        t_probs = np.histogram(t_new,bins=range(size_vocab),density=True)
        inf_topics.append(t_probs)
    
    plot_disc(range(sv),topics,title='True topics')
    plot_disc(range(sv),inf_topics,title='Inf topics')
    

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
    return None
