#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os, sys
#, re
from itertools import izip
from math import sqrt, exp, log
import shutil
import tempfile
import subprocess

import time
from optparse import OptionParser
import shlex
from signal import signal, SIGPIPE, SIG_DFL

from threading  import Thread

os.system("taskset -p 0xFFFFFFFF %d >/dev/stderr" % os.getpid())


class named_pipe():
    def __init__(self):
#        print >>sys.stderr, "Creating pipe"
        self.dirname = tempfile.mkdtemp()
        
        try:
            self.path = os.path.join(self.dirname, 'named_pipe')
            os.mkfifo(self.path)
        except OSError:
            raise "Could not create FIFO"
    def kill_pipe(self):
#        os.remove(self.path)
        shutil.rmtree(self.dirname)
#    @property
#    def path(self):
#        return self.path


def run_local(cmd, dep=[], stdin=None, 
                                    out=sys.stdout,
                                    err=sys.stderr,
                       no_exec=False, 
                       wait=True):
    print >>sys.stderr,"Executing:", cmd
#    print >>sys.stderr,"Splitted:", shlex.split(cmd)
    if no_exec:        
        return None
    #wait for all dependencies
    for p in dep:
        p.wait()
    #Run the cmd
    p=subprocess.Popen(shlex.split(cmd),
                       stdin=stdin, 
                         stdout=out,
                         stderr=err
                         , preexec_fn = lambda: signal(SIGPIPE, SIG_DFL)
                         ,close_fds=True
                         )
    if wait:
        p.wait()
        return p
    
    return p
                    
                    
def cat(files, result):
    subprocess.Popen(['cat']+files,
                         stdout=result,
                         stderr=subprocess.STDOUT
                         ).communicate()[0]
def multi_lcount(filenames):
#    print>>sys.stderr, "Counting file: '%s'"%(filename)
    procs=[subprocess.Popen(['wc', '-l', filename],
                         stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT
                         ) for filename in filenames]
    return [int(p.communicate()[0].partition(b' ')[0]) for p in procs]

def lcount(filename):
#    print>>sys.stderr, "Counting file: '%s'"%(filename)
    out = subprocess.Popen(['wc', '-l', filename],
                         stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT
                         ).communicate()[0]
#    print >>sys.stdout, "WC OUT=", out
    return int(out.partition(b' ')[0])        
    
def open_infile(f):
    return open(f,"rb") if f !="-" else sys.stdin
    
def open_outfile(f):
    return open(f,"wb") if f !="-" else sys.stdout


    
description= "This script scores sentences by repeated sampling"
usage = "Usage: %prog [options] corpus"
parser = OptionParser(usage)
parser.set_description(description)
#
#parser.add_option("-m", "--make-vocab", action="store_true",
#                      dest="make_vocab",
#                      help="Whether to create a vocabulary from the provided text file",
#                      default=False)
# 
#parser.add_option("-s", "--save-scores-for-sub-corpora", action="store_true",
#                              dest="save_sub",
#                              help="Save the scores for each file if multiple out-of-domain files are specififed (default: False)",
#                              default=False)       
##                              
#parser.add_option("-s", "--stop-word-list", action="store",type="string",
#                              dest="stop_words_file",
#                              help="file containing stop words one per line",
#                              default='')      

parser.add_option("-s", "--sample-size", action="store",type="int",
                          dest="ssize",
                          help="Number of sentences to sample in each evaluation. (default: 1000)",
                          default=1000) 
                          
#parser.add_option("-v", "--word-vectors", action="store",type="string",
#                              dest="w2v_vectors",
#                              help="The word2vec binary vectors file name (default: None)",
#                              default=None)                  
#                 
#     
#parser.add_option("-e", "--extension-table", action="store",type="string",
#                              dest="ttable",
#                              help="The translation table used to extend the LMs (default: None)",
#                              default=None)                  
     
parser.add_option("-o", "--output", action="store",type="string",
                              dest="output",
                              help="Output file (default: stdout)",
                              default="-")       

parser.add_option("-d", "--output-dir", action="store",type="string",
                              dest="outdir",
                              help="Directory where all files will be saved (default: .)",
                              default=".")       
                          
parser.add_option("-L", "--log-dir", action="store",type="string",
                              dest="logdir",
                              help="The output directory for log files (default: .)",
                              default=".")       
                              
parser.add_option("-n", "--ngram-order", action="store",type="int",
                              dest="order",
                              help="The maximum ngram order for LMs (default: 4)",
                              default=4)       
                              
                               
parser.add_option("-i", "--niters", action="store",type="int",
                              dest="niters",
                              help="Number of iterations (default: 5 )",
                              default=5)       
                      
parser.add_option("-m", "--max-procs", action="store",type="int",
                              dest="max_procs",
                              help="The maximum number of processes which can be started in parallel (default: 1)",
                              default=1)       
                      
#parser.add_option("-p", "--percentage-lines", action="store",type="string",
#                              dest="percent",
#                              help="The percentage of line to select (comma-separated list if many). If this is given {-N| --nlines} will be ignored (default: 10% of the input)",
#                              default=None)     

parser.add_option("-S", "--slurm-args", action="store",type="string",
                              dest="slurm_args",
                              help="The Slurm arguments which will be used when starting the dictionary pruner (default: -N 2 -c 2)",
                              default="-N 2 -c 2 -J PRUNE  ") #"-x i13hpc[17-18] ")     
   
#parser.add_option("-P", "--pruner-extra-args", action="store",type="string",
#                              dest="pdprune_args",
#                              help="The extra arguments to be passed to pdprune binary (default: -p -w /tmp,/export/data1/data)",
#                              default=" -p -w /export/data1/data ")                                        
                              
#parser.add_option("-B", "--pruner-nbest-size", action="store",type="int",
#                              dest="pnbest",
#                              help="The n-best list size used to prune the lexicon (default: 100)",
#                              default=100)       

#parser.add_option("-b", "--normalizer-nbest-size", action="store",type="int",
#                              dest="nnbest",
#                              help="The n-best list size used to prune the lexicon before normalization (default: 10)",
#                              default=10)       
                            
#parser.add_option("", "--extend-intersection-only", action="store_true",
#                              dest="intersect_only",
#                              help="Extend the intesection of the indomain and out of domani vocabularies instead of all the vocabulary",
#                              default=False)                                     
                              
#parser.add_option("-R", "--random-ood-selection", action="store_true",
#                              dest="rand_ood_selection",
#                              help="Select the out-of-domain sample randomly rather than perplexity-guided",
#                              default=False)                   
                              
#parser.add_option("-M", "--extend-at-most", action="store",type="int",
#                              dest="max_ext",
#                              help="The maximal number of vocabulary words to extend (default: illimitted)",
#                              default=None)       
                              
(options, args) = parser.parse_args()
if len(args)<1:
    parser.error("Insufficient number of arguments")
#if options.ttable is not None and options.w2v_vectors is not None:
#    parser.error("Only one of bilingual/monolingual extension table can be used")
def mk_title(title):
    amph_str="*"*(len(title)+4*2)
    print >>sys.stderr, "\n\n%s\n**  %s  **\n%s\n"%(amph_str, title, amph_str)
SELECT_OOD="/project/mt/user/mmediani/simsel/select-OOD.R.sh"
#SELECT_OOD="/project/mt/user/mmediani/simsel/select-OOD.sh"
SELECT_RAND_SAMPLE="/project/mt/user/mmediani/wngram/rand-sample.sh"
#SELECT_VOC="python /project/mt/user/mmediani/simsel/mk-indom-ood-voc.py -f -o %s" #%(options.prefix)
#SELECT_VOC="python /project/mt/user/mmediani/simsel/mk-indom-ood-voc.py -o %s -s %s" #%(options.prefix)
#NG_COUNT="python /project/mt/user/mmediani/simsel/mk-ngrams.py"
NG_COUNT="/project/mt/user/mmediani/tools/srilm/bin/i686-m64/ngram-count -text %s -order %d -write -"
#COMPUTE_LM="mpirun python /project/mt/user/mmediani/simsel/compute-kn-lm.mpi.py -i -k -r -a -s kn -x xi_kn"
#COMPUTE_LM="/project/mt/user/mmediani/tools/mpi/openmpi/bin/mpirun python /project/mt/user/mmediani/simsel/compute-kn-lm.only-unigrams.mpi.py -i -k -r -a -s kn -x xi_kn"
if True:
    COMPUTE_LM="/project/mt/user/mmediani/tools/srilm/bin/i686-m64/ngram-count -wbdiscount -interpolate -gt3min 0 -gt4min 0"
else:
    COMPUTE_LM="mpirun -n 1 python /project/mt/user/mmediani/ker-smooth/train-lm.pen-p.new.mpi.py  -i -k -s kn -a -x xi_exp -u good-turing -p 2:small-noisy.nllr.2,3:small-noisy.nllr.3,4:small-noisy.nllr.4 -S 0.1 "
#SCORER="python /project/mt/user/mmediani/simsel/score.py"
SCORER="python /project/mt/user/mmediani/ker-smooth/xent.py"
#SELECTOR="/project/mt/user/mmediani/simsel/global-select-files.sh"
SAVE_SELECT="python /project/mt/user/mmediani/simsel/select-lines-mono.py"
#OOD_VOC="/project/mt/user/mmediani/simsel/mk-ood-voc.sh"
#UNION_VOC="/project/mt/user/mmediani/simsel/union-voc.sh"
#PRUNER="/project/mt/user/mmediani/pdict-prune/pdprune"
#MONO_ASSOCS="python /project/mt/user/mmediani/simsel/mk-mono-assocs4voc.py"
#NORMALIZER="python /project/mt/user/mmediani/pdict-prune/normalize-pruned.py"
#MPI_WRAPPER="/project/mt/user/mmediani/ompiwrap/ompiwrapper.py -w"


#SORT="/project/mt/user/mmediani/simsel/global-score-sorter.sh "

NGC="/project/mt/user/mmediani/tools/srilm/bin/i686-m64/ngram-count"
PRINT_ONLY=False # True #
RUN=run_local
#options.prefix=os.path.join(options.outdir, options.prefix)
try:
    os.makedirs(options.outdir)
except OSError:
    pass
try:
    os.makedirs(options.logdir)
except OSError:
    pass
#print >>sys.stderr, "Output directory:", options.outdir
#print >>sys.stderr, "Output prefix:", options.prefix
#mk_fname = lambda fname: "%s.%s"%(options.prefix, fname)
#mk_log_fname=lambda fname: os.path.join(options.logdir, fname)
print >>sys.stderr, "Maximum number of parallel jobs :", options.max_procs

print >>sys.stderr, "All LMs will be %d-grams"%( options.order)
#if options.ttable is not None:
#    print >>sys.stderr, "LMs will be extended using the bilingual table:", options.ttable
#if options.w2v_vectors is not None:
#    print >>sys.stderr,  "LMs will be extended using the word2vec vectors:", options.w2v_vectors 
print >>sys.stderr, "Counting lines in the corpus..."
total_lines=lcount(args[0])
#totals= #map(lcount, args[1:]) #[corp, None] for corp in args[1:]]
#print >>sys.stderr, "Totals=", totals
#total_lines=sum(totals)
#for f, n in izip(args[1:], totals):    
#    print >>sys.stderr, "\t%s\t%d"%(f, n)
    
#if len(args[1:])>1:
#    print >>sys.stderr, "Concatenating files..."
#    
#    cat(args[1:], open_outfile(mk_fname("big-corp"))) #open_outfile(args[1]))
#    big_corp=mk_fname("big-corp")
#else:
big_corp=args[0]
#sys.exit(0)
    
MULTI_ARG_SEP=","
print >>sys.stderr, "The corpus contains %d lines"%(total_lines)
#if options.percent is not None:
#    options.percent=map(float, options.percent.split(MULTI_ARG_SEP))
#    assert(all(0<=p <=100 for p in options.percent))
#    options.nlines=sorted(set(int(round(p/100.0*total_lines)) for p in options.percent))
#elif options.nlines is None:
#        options.nlines=   [int(round(0.1*total_lines))]
#else:
#    options.nlines=sorted(set(int(n) for n in options.nlines.split(MULTI_ARG_SEP)))

#print >>sys.stderr, "Number of lines to be selected from the out-of-domain corpus:", (options.nlines)
print >>sys.stderr, "Number of iterations: %d"%(options.niters)
samplers=[]
counters=[]
lm_trainers=[]
evaluators=[]
tmp_fifos=[]
rand_init=time.time()
#q=total_lines/options.niters
def meanstd(seq):
#    from math import sqrt
    sumx, sumx2, lseq=reduce(lambda x,y:(x[0]+y,x[1]+y*y, x[2]+1),seq,(0,0,0))
    mean=sumx*1./lseq
    return mean, sqrt((sumx2 / lseq) - (mean * mean)) 
    
def read_evaluator(evaluator_out, evals):
    for line in evaluator_out:
        evals.append(line.strip())
        
scores=[]
for it in range(options.niters):
    ############### Sampling
    mk_title("Selecting the sample")
    cmd="%s %s %s %s %d"%(SELECT_RAND_SAMPLE, options.ssize, total_lines, args[0], rand_init+2*it+1)
#    cmd="%s %s %s %s %d %d %d"%(SELECT_RAND_SAMPLE, options.ssize, total_lines, args[0], rand_init+2*it+1, it*q+1, (it+1)*q+((it+1)/options.niters)*(total_lines%options.niters))
#    ood_fname=mk_fname("ood")
#    log=mk_log_fname("select-ood.log")
    samplers.append(RUN(cmd, no_exec=PRINT_ONLY, out=subprocess.PIPE , wait=False) )
    
    ############### COUNT
    mk_title("Training LM on sampple")
    cmd="%s"%(NG_COUNT %("-", options.order)) 
        
    counters.append(RUN(cmd, no_exec=PRINT_ONLY, stdin=samplers[-1].stdout, out=subprocess.PIPE, wait=False))
    
    
    tmp_fifos.append(named_pipe())
    

    cmd="%s %s %s"%(SCORER, args[0], tmp_fifos[-1].path)
    evaluators.append( RUN(cmd, no_exec=PRINT_ONLY, out=subprocess.PIPE, wait=False) )
    scores.append([])
    
    t = Thread(target=read_evaluator, args=(evaluators[-1].stdout, scores[-1]))
    t.daemon = True 
    t.start()
    
    
    if True:
        cmd="%s -order %d -read - -lm %s"%(COMPUTE_LM, options.order, tmp_fifos[-1].path) 
    else:
        cmd="%s -o %d - %s"%(COMPUTE_LM, options.order, tmp_fifos[-1].path) 
    lm_trainers.append( RUN(cmd, no_exec=PRINT_ONLY, stdin=counters[-1].stdout, wait=False) )
    
#        evaluators[-1].wait()
       
done=[False]*len(evaluators)
while not all(done):
    for pi, e in enumerate(evaluators):
        if e.poll() is not None:
            done[pi]=True
#    print >>sys.stderr, "Done:", sum(done)        
#scores=[map(float, evaluator.communicate()[0].split()) for evaluator in evaluators]
scores=[map(float, score) for score in scores]
avg=lambda t:sum(t)/len(t)
out=open_outfile(options.output)

scores=map(avg, izip(*scores))
m, std=meanstd(scores)
scores=[(s-m)/std for s in scores]

w0, wf=0.01, 0.999
x0=-log(1/w0-1)
xf=-log(1/wf-1)
s0, sf=min(scores), max(scores)
scale=lambda s: x0+(xf-x0)*(s-s0)/(sf-s0)
for s in scores:
    print >>out, 1/(1+exp(-scale(s)))
for fifo in tmp_fifos:
#    evaluator.wait()    
    fifo.kill_pipe()
    
exit(0)
    

###########
## SELECT OOD
###########

mk_title("Selecting the out-of-domain sample")
cmd="%s %s %s %s"%(SELECT_OOD if not options.rand_ood_selection else SELECT_RAND_OOD, args[0], big_corp, options.outdir)
ood_fname=mk_fname("ood")
log=mk_log_fname("select-ood.log")
RUN(cmd, no_exec=PRINT_ONLY, out=open_outfile(ood_fname), err=open_outfile(log))


###########
## SELECT Voc
###########
intermediate_voc_pre=mk_fname("voc")
mk_title("Selecting the vocabulary")
cmd="%s %s %s"%(SELECT_VOC%(intermediate_voc_pre, options.selection), args[0], ood_fname)
voc_fname=mk_fname("VOC")
log=mk_log_fname("select-voc.log")

RUN(cmd, no_exec=PRINT_ONLY, out=open_outfile(voc_fname), err=open_outfile(log))


###########
## Making LMs
###########
mk_title("LM training")
print >>sys.stderr, "1. Counting..."
############### INDOM COUNT
indom_count_file=mk_fname("indom.counts")
cmd="%s -n %d -o %s %s %s"%(NG_COUNT, options.order, indom_count_file, args[0], voc_fname) 

log=mk_log_fname("indom-count.log")

ind_count_p=RUN(cmd, no_exec=PRINT_ONLY, out=None, err=open_outfile(log), wait=False)

############### OOD COUNT
ood_count_file=mk_fname("ood.counts")
cmd="%s -n %d -o %s %s %s"%(NG_COUNT, options.order, ood_count_file, ood_fname, voc_fname)

log=mk_log_fname("ood-count.log")

ood_count_p=RUN(cmd, no_exec=PRINT_ONLY, out=None, err=open_outfile(log), wait=False)
#
#ind_count_p.wait()
#ood_count_p.wait()
normalized_lex=None

############### IF NEED TO EXTEND USING W2V vectors
if options.w2v_vectors is not None:
    print >>sys.stderr, "\n1.1. Computing associations from word2vec vectors..."

############### Find the corresponding vocab    
    
    if options.intersect_only:
        sub_voc= intermediate_voc_pre+".inter"
    else:
        sub_voc= voc_fname
    if options.max_ext is not None:
        new_sub_voc=mk_fname("sub-voc.voc")
        VOC_SAMPLER="/project/mt/user/mmediani/simsel/select-subvoc.sh"
        cmd="%s %s %d %s"%(VOC_SAMPLER, sub_voc, options.max_ext, " ".join((args[0], ood_fname)))
        log=mk_log_fname("sub-voc.log")
        RUN(cmd, no_exec=PRINT_ONLY, out=open_outfile(new_sub_voc), err=open_outfile(log))     
        sub_voc=new_sub_voc
        
    big_ood_voc=mk_fname("big-ood.voc")
    if options.intersect_only:
        cmd="%s %s %s"%(OOD_VOC, big_corp, intermediate_voc_pre+".indom-diff") #".inter") #".indom")    
#    cmd="%s %s %s"%(OOD_VOC, big_corp, intermediate_voc_pre+".inter") #".indom")    
    else:
        cmd="%s %s %s"%(OOD_VOC, big_corp, intermediate_voc_pre+".excl")     
    log=mk_log_fname("big-ood-voc.log")
    ood_voc_p=RUN(cmd, no_exec=PRINT_ONLY, out=open_outfile(big_ood_voc), err=open_outfile(log)) #, wait=False)
#    
#    prune_voc=mk_fname("prune.voc")
#    if options.intersect_only:
#        cmd="%s %s %s "%(UNION_VOC, big_ood_voc, intermediate_voc_pre+".inter")#voc_fname) #intermediate_voc_pre+".indom")    
#    else:
#        cmd="%s %s %s "%(UNION_VOC, big_ood_voc, voc_fname) #intermediate_voc_pre+".indom")    
##    cmd="%s %s %s "%(UNION_VOC, big_ood_voc, intermediate_voc_pre+".indom-diff")#voc_fname) #intermediate_voc_pre+".indom")    
#    log=mk_log_fname("prune-voc.log")
#    RUN(cmd, no_exec=PRINT_ONLY, out=open_outfile(prune_voc), err=open_outfile(log))     
################  Run the extractor
#    pruned_lex=mk_fname("pruned-lex")
#    pruner_cmd="%s %s -o %s -v %s -B %d %s"%(MONO_ASSOCS, options.ttable, pruned_lex, prune_voc, options.pnbest, options.pdprune_args)
#    cmd='%s "%s" "%s"'%(MPI_WRAPPER, options.slurm_args, pruner_cmd)    
#    log=mk_log_fname("mpi-wrapper.log")
#    RUN(cmd, no_exec=PRINT_ONLY, out=None, err=open_outfile(log))         
################  Normalize
    normalized_lex=mk_fname("pruned-lex.norm")    
    
    cmd="%s -o %s -s %s -n %d %s %s"%(MONO_ASSOCS, normalized_lex, big_ood_voc, options.nnbest, options.w2v_vectors, sub_voc)
#    cmd="%s -o %s -t %s -s %s -n %d %s"%(NORMALIZER, normalized_lex,  intermediate_voc_pre+".indom-diff", big_ood_voc, options.nnbest, pruned_lex)

    log=mk_log_fname("normalize-pruned.log")
    RUN(cmd, no_exec=PRINT_ONLY, out=None, err=open_outfile(log))        
#    sys.exit(0)
    


############### IF NEED TO EXTEND
if options.ttable is not None:
    print >>sys.stderr, "\n1.1. Pruning the lexicon..."

    if options.intersect_only:
        sub_voc= intermediate_voc_pre+".inter"
    else:
        sub_voc= voc_fname
    if options.max_ext is not None:
        new_sub_voc=mk_fname("sub-voc.voc")
        VOC_SAMPLER="/project/mt/user/mmediani/simsel/select-subvoc.sh"
        cmd="%s %s %d %s"%(VOC_SAMPLER, sub_voc, options.max_ext, " ".join((args[0], ood_fname)))
        log=mk_log_fname("sub-voc.log")
        RUN(cmd, no_exec=PRINT_ONLY, out=open_outfile(new_sub_voc), err=open_outfile(log))     
        sub_voc=new_sub_voc
############### Find the corresponding vocab    
    big_ood_voc=mk_fname("big-ood.voc")
    if options.intersect_only:
        cmd="%s %s %s"%(OOD_VOC, big_corp, intermediate_voc_pre+".indom-diff") #".inter") #".indom")    
#    cmd="%s %s %s"%(OOD_VOC, big_corp, intermediate_voc_pre+".inter") #".indom")    
    else:
        cmd="%s %s %s"%(OOD_VOC, big_corp, intermediate_voc_pre+".excl")     
    log=mk_log_fname("big-ood-voc.log")
    ood_voc_p=RUN(cmd, no_exec=PRINT_ONLY, out=open_outfile(big_ood_voc), err=open_outfile(log)) #, wait=False)
    prune_voc=mk_fname("prune.voc")
#    if options.intersect_only:
#        cmd="%s %s %s "%(UNION_VOC, big_ood_voc, intermediate_voc_pre+".inter")#voc_fname) #intermediate_voc_pre+".indom")    
#    else:
    cmd="%s %s %s "%(UNION_VOC, big_ood_voc, sub_voc) #intermediate_voc_pre+".indom")    
#    cmd="%s %s %s "%(UNION_VOC, big_ood_voc, intermediate_voc_pre+".indom-diff")#voc_fname) #intermediate_voc_pre+".indom")    
    log=mk_log_fname("prune-voc.log")
    RUN(cmd, no_exec=PRINT_ONLY, out=open_outfile(prune_voc), err=open_outfile(log))     
###############  Run the pruner
    pruned_lex=mk_fname("pruned-lex")
    pruner_cmd="%s %s -o %s -v %s -B %d %s"%(PRUNER, options.ttable, pruned_lex, prune_voc, options.pnbest, options.pdprune_args)
    cmd='%s "%s" "%s"'%(MPI_WRAPPER, options.slurm_args, pruner_cmd)    
    log=mk_log_fname("mpi-wrapper.log")
    RUN(cmd, no_exec=PRINT_ONLY, out=None, err=open_outfile(log))         
###############  Normalize
    normalized_lex=mk_fname("pruned-lex.norm")    
    if options.intersect_only:            
        cmd="%s -o %s -t %s -s %s -n %d %s"%(NORMALIZER, normalized_lex, intermediate_voc_pre+".inter", big_ood_voc, options.nnbest, pruned_lex)
    else:
        cmd="%s -o %s -t %s -s %s -n %d %s"%(NORMALIZER, normalized_lex, voc_fname, big_ood_voc, options.nnbest, pruned_lex)
#    cmd="%s -o %s -t %s -s %s -n %d %s"%(NORMALIZER, normalized_lex,  intermediate_voc_pre+".indom-diff", big_ood_voc, options.nnbest, pruned_lex)

    log=mk_log_fname("normalize-pruned.log")
    RUN(cmd, no_exec=PRINT_ONLY, out=None, err=open_outfile(log))        
    


print >>sys.stderr, "\n2. Training..."
############### INDOM TRAIN
indom_lm=mk_fname("indom.lm")
#cmd="%s -o %d %s %s %s"%(COMPUTE_LM, options.order, "-T %s"%(normalized_lex) if normalized_lex is not None else "", indom_count_file, indom_lm) 
cmd="%s -order %d -read %s -lm %s"%(COMPUTE_LM, options.order, indom_count_file, indom_lm) 

log=mk_log_fname("indom-lm.log")

indom_lm_p=RUN(cmd, no_exec=PRINT_ONLY, dep=[ind_count_p], out=None, err=open_outfile(log), wait=False)

############### OOD TRAIN

ood_lm=mk_fname("ood.lm")
#cmd="%s -o %d %s %s %s"%(COMPUTE_LM, options.order, "-T %s"%(normalized_lex) if normalized_lex is not None else "", ood_count_file, ood_lm) 
cmd="%s -order %d -read %s -lm %s"%(COMPUTE_LM, options.order, ood_count_file, ood_lm) 

log=mk_log_fname("ood-lm.log")

ood_lm_p=RUN(cmd, no_exec=PRINT_ONLY, dep=[ood_count_p], out=None, err=open_outfile(log), wait=False)
#indom_lm_p.wait()
#ood_lm_p.wait()



###########
## Scoring
###########
mk_title("Scoring")
scorer_ps=[]
score_files=[]
for f in args[1:]:
    score_files.append(mk_fname(os.path.basename(f)+".scores"))
    cmd="%s -o %s %s %s %s"%(SCORER, score_files[-1], f, indom_lm, ood_lm) 
    log=mk_log_fname(os.path.basename(score_files[-1])+".log")
    scorer_ps.append(RUN(cmd, no_exec=PRINT_ONLY, dep=[indom_lm_p, ood_lm_p],  out=None, err=open_outfile(log), wait=False))
    if len(scorer_ps)>= options.max_procs:
        for p in scorer_ps: #[-options.max_procs:]:
#            print >>sys.stderr, "waiting for %d jobs.."%(options.max_procs)
            p.wait()
        scorer_ps=[]
#
#for p in scorer_ps:
#    p.wait()

###########
## Sorting
###########
mk_title("Sorting scores")
global_score=mk_fname("global-scores")
cmd="%s %s"%(SORT, " ".join(score_files))
log=mk_log_fname("global-scores.log")

RUN(cmd, no_exec=PRINT_ONLY,  dep=scorer_ps, out=open_outfile(global_score), err=open_outfile(log))

###########
## Selecting
###########
mk_title("Selecting")
##output=<scorefilename>.select.percent_index
POST_SELECT=".select.%d"
select_ps=[]
for n in (options.nlines):
    post_f=POST_SELECT%(n)
    cmd="%s %d %s %s"%(SELECTOR, n, post_f, global_score)
    log=mk_log_fname("%s.log"%(post_f[1:]))
    select_ps.append(RUN(cmd, no_exec=PRINT_ONLY, out=None, err=open_outfile(log), wait=False))
    if len(select_ps)>= options.max_procs:
        for p in select_ps:
#            print >>sys.stderr, "waiting for %d jobs.."%(options.max_procs)
            p.wait()
        scorer_ps=[]
    
if not PRINT_ONLY:
    for p in select_ps:
        p.wait()
    
################
## Saving the selection
################
mk_title("Saving selected")
save_ps=[]
for sf, of in izip(score_files, args[1:]):
    for n in (options.nlines):
        post_f=POST_SELECT%(n)
        if PRINT_ONLY or os.path.isfile(sf+post_f):
            outfname=os.path.join(options.outdir, os.path.basename(of))+post_f
            cmd="%s -o %s %s %s"%(SAVE_SELECT, outfname, sf+post_f, of)
            log=mk_log_fname(os.path.basename(outfname)+".log")
            save_ps.append(RUN(cmd, no_exec=PRINT_ONLY, dep=[],  out=None, err=open_outfile(log), wait=False))
            if len(save_ps)>= options.max_procs:
                for p in save_ps: #[-options.max_procs:]:
#                print >>sys.stderr, "waiting for %d jobs.."%(options.max_procs)
                    p.wait()
                save_ps=[]
        
        else:
            pass
