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
                              
parser.add_option("-r", "--rand-init", action="store",type="float",
                              dest="randinit",
                              help="A random initialization seed (default: current time )",
                              default=time.time())       
                              
parser.add_option("-w", "--w0", action="store",type="float",
                              dest="w0",
                              help="The lowest sentence weight. Weights will be linearly scaled to fit between w0 and wf (default: 0.1)",
                              default=0.1)       
                              
parser.add_option("-W", "--wf", action="store",type="float",
                              dest="wf",
                              help="The highest sentence weight. Weights will be linearly scaled to fit between w0 and wf (default: 0.999)",
                              default=0.999)       
                              
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
SELECT_RAND_SAMPLE="/home/medmediani/code/wlm-train/rand-sample.sh"
#SELECT_VOC="python /project/mt/user/mmediani/simsel/mk-indom-ood-voc.py -f -o %s" #%(options.prefix)
#SELECT_VOC="python /project/mt/user/mmediani/simsel/mk-indom-ood-voc.py -o %s -s %s" #%(options.prefix)
#NG_COUNT="python /project/mt/user/mmediani/simsel/mk-ngrams.py"

NG_COUNT="/home/medmediani/code/wlm-train/wng-count -c %s -n %d "
#COMPUTE_LM="mpirun python /project/mt/user/mmediani/simsel/compute-kn-lm.mpi.py -i -k -r -a -s kn -x xi_kn"
#COMPUTE_LM="/project/mt/user/mmediani/tools/mpi/openmpi/bin/mpirun python /project/mt/user/mmediani/simsel/compute-kn-lm.only-unigrams.mpi.py -i -k -r -a -s kn -x xi_kn"
SRILM_TRAIN=True
if SRILM_TRAIN:
    COMPUTE_LM="/project/mt/user/mmediani/tools/srilm/bin/i686-m64/ngram-count -wbdiscount -interpolate -gt3min 0 -gt4min 0"
else:
   COMPUTE_LM="mpirun -n 1 python /project/mt/user/mmediani/wngram/train-lm.count-discount.assoc.py  -i -k -o 4 -s kn -a -x xi_exp -u nonparametric-lindley -m log-likelihood-ratio"
#    COMPUTE_LM="mpirun -n 1 python /project/mt/user/mmediani/ker-smooth/train-lm.pen-p.new.mpi.py  -i -k -s kn -a -x xi_exp -u good-turing -p 2:small-noisy.nllr.2,3:small-noisy.nllr.3,4:small-noisy.nllr.4 -S 0.1 "
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

PREPRO_CMD="tr '[:upper:]' '[:lower:]'"

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
rand_init=options.randinit
#rand_init=123454321
#q=total_lines/options.niters
prepro1=[]
prepro2=[]

def meanstd(seq):
#    from math import sqrt
    sumx, sumx2, lseq=reduce(lambda x,y:(x[0]+y,x[1]+y*y, x[2]+1),seq,(0,0,0))
    mean=sumx*1./lseq
    return mean, sqrt((sumx2 / lseq) - (mean * mean)) 
    
for it in range(options.niters):
    ############### Sampling
    mk_title("Selecting the sample")
    cmd="%s %s %s %s %d"%(SELECT_RAND_SAMPLE, options.ssize, total_lines, args[0], rand_init+2*it+1)
#    cmd="%s %s %s %s %d %d %d"%(SELECT_RAND_SAMPLE, options.ssize, total_lines, args[0], rand_init+2*it+1, it*q+1, (it+1)*q+((it+1)/options.niters)*(total_lines%options.niters))
#    ood_fname=mk_fname("ood")
#    log=mk_log_fname("select-ood.log")
    samplers.append(RUN(cmd, no_exec=PRINT_ONLY, out=subprocess.PIPE , wait=False) )
    
    prepro1.append(RUN(PREPRO_CMD,no_exec=PRINT_ONLY, stdin= samplers[-1].stdout, out=subprocess.PIPE , wait=False ))
    
    ############### COUNT
    mk_title("Training LM on sampple")
    cmd="%s"%(NG_COUNT %("-", options.order)) 
        
    counters.append(RUN(cmd, no_exec=PRINT_ONLY, stdin=prepro1[-1].stdout, out=subprocess.PIPE, wait=False))
    
    
    tmp_fifos.append(named_pipe())
    
    prepro2.append(RUN(PREPRO_CMD,no_exec=PRINT_ONLY, stdin= file(args[0]), out=subprocess.PIPE , wait=False ))
    cmd="%s - %s"%(SCORER,  tmp_fifos[-1].path)
    evaluators.append( RUN(cmd, no_exec=PRINT_ONLY, stdin=prepro2[-1].stdout , out=subprocess.PIPE, wait=False) )
    
    if SRILM_TRAIN:
        cmd="%s -order %d -read - -lm %s"%(COMPUTE_LM, options.order, tmp_fifos[-1].path) 
    else:
        cmd="%s -o %d - %s"%(COMPUTE_LM, options.order, tmp_fifos[-1].path) 
    lm_trainers.append( RUN(cmd, no_exec=PRINT_ONLY, stdin=counters[-1].stdout, out=subprocess.PIPE, wait=False) )
    
#        evaluators[-1].wait()
        
def logsumexp(l):
    m=max(l)
    return m+sum(exp(x-m) for x in l)
def normalize(l):
    rss=sqrt(sum(x**2 for x in l))
    return [x/rss for x in l]

def scale(l):
    m=abs(max(l))
    return [x/m for x in l]


def standardize(l):
    m, std=meanstd(l)
    return [(x-m)/std for x in l]
    
scores=[standardize(map(float, evaluator.communicate()[0].split()) ) for evaluator in evaluators]
#scores=[scale(map(float, evaluator.communicate()[0].split()) ) for evaluator in evaluators]
#scores=[normalize(map(float, evaluator.communicate()[0].split()) ) for evaluator in evaluators]

#e=zip(*scores)[0]
#print >>sys.stderr, e
#print >>sys.stderr, logsumexp(e)
#
#print >>sys.stderr, [x-logsumexp(e) for x in e]
#
#exit(0)
avg=lambda t:sum(t)/len(t)
out=open_outfile(options.output)

scores=map(avg, izip(*scores))
#m, std=meanstd(scores)
#scores=[(s-m)/std for s in scores]


w0, wf=options.w0, options.wf

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
    
