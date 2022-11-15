#!/usr/bin/env python



import argparse
import sys


GRAMS_MARKER="-grams:"
END_MARKER="\\end\\"
WORD_SEP=" "


BOS="<s>"
EOS="</s>"
UNK="<unk>"
LOG_ZERO=-100

def open_infile(f, mode="r"):
    return open(f,mode) if f !="-" else sys.stdin
  
def open_outfile(f, mode="w"):
    return open(f,mode) if f !="-" else sys.stdout
    
def print_err(*args, **kwargs):
    print(*args, **kwargs, file=sys.stderr)

def load_lm(fname, max_ord=None):
    f=open_infile(fname)
    od={}
    if max_ord is not None:
        orders=set(range(1,max_ord+1))
    else:
        orders=set()
        
    print_err("Loading...")
    
    line=f.readline()
    while line:
        if END_MARKER in line:
            break
        elif GRAMS_MARKER in line:
            curord=int(line.split("\\",1)[1].strip().replace(GRAMS_MARKER,""))
            if max_ord is None or curord in orders:
                od[curord]={}

                orders.discard(curord)
                for line in f:
                    line=line.strip()
                    if line:
                        if line.startswith("\\"):
                            break
                        line=line.split("\t")

                        
#                        od[curord][line[1]]=(float(line[0]), ) #{"prob":float(line[0])} #p}
                        try:
                            od[curord][line[1]]=(float(line[0]), float(line[2]))
                        except IndexError:
                            od[curord][line[1]]=(float(line[0]), ) 
                            pass
        else:
#            print_err("Skipping line:",line)
            line=f.readline()
                
    if orders:
        print_err ("Could not load orders:",orders)
    if UNK not in od[1]:
        print_err("*The ARPA file is missing <unk>.  Substituting log10 probability -100.")
        od[1][UNK]=(LOG_ZERO, )
    return od 
    
def bow(ng, od): 
    try: 
        return od[ng][1]; 
    except (IndexError, KeyError): 
        return 0.0

def getprob(ng_toks, od,  ord=None, backoff=None):
    if ord is None:
        ord=len(ng_toks)
    ng=WORD_SEP.join(ng_toks)
    
    if ng in od[ord]:
#            print >>sys.stderr,"P[order=%d]('%s')=%f"%(ord, ng, od[ord][ng]["prob"])
        p= od[ord][ng][0]
        
        if backoff is not None :
            return p+backoff #od[ord][ng]["bow"]
        return p
#        print >>sys.stderr,"\t\t
#            print >>sys.stderr,"Returning 0 prob (could not find an entry)"
    elif (not ng) or ord==1:
        return -99
    cr=ng_toks[1:]
    lc=WORD_SEP.join(ng_toks[:-1])
    if not lc or not cr:
        return -99
        cr=""
#        print >>sys.stderr,"Recursively looking for '%s' (Weight('%s')=%f)"%((cr), ng, od[ord-1][lc]["bow"])
    try:
        if backoff is None :
            return getprob(cr, od, ord-1, backoff=bow(lc,  od[ord-1]) )
        else:
            return backoff+getprob(cr, od, ord-1, backoff=bow(lc,  od[ord-1]))
    except KeyError:
        print_err("ngram '%s' has '%s' as prefix. BOW expected at the latter but was not found"%(ng,lc))
        raise
#        print >>sys.stderr,"Recursively looking for '%s' (Weight('%s')=%f)"%((cr, w), c, ctxts[ord][c])
#        p=(od[ord-1][WORD_SEP.join((cr,w))]["bow"])+getprob(cr, w, od, ctxts, ord-1)
#
#        return p

def get_sent_prob(sent, od, order=None, ignore_OOV=False):
    if ignore_OOV: 
        tokens=[BOS]+[w for w in sent.split() if w in od[1] ]+[EOS]
    else:
        tokens=[BOS]+[w if w in od[1] else UNK for w in sent.split()]+[EOS]
    l=len(tokens)
    max_ord=max(od.keys())
    if order is not None:
         max_ord=min(max_ord,order)
    return l,sum(getprob(tokens[:i], od) for i in range (2,min(max_ord, l)))+\
                sum(getprob(tokens[i-max_ord:i], od) for i in range (max_ord,l+1))
    
if __name__=="__main__":
    parser = argparse.ArgumentParser(description="This script evaluates the log10 probabilty of each sentence "
                                                 "from data (in text form), given a language model in arpa format "
                                                 "and a specific ngram order.",                                
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("arpa_lm", type=str,
                        help="Input language model in arpa form.")
    parser.add_argument("-o","--order", type=int, default=None,
                        help="ngram order")
    
    parser.add_argument("-i","--ignore-OOV", action='store_true', #=int, default=None,
                        help="Ignore OOVs")

    args = parser.parse_args()

    lm=load_lm(args.arpa_lm)

    print_err("Orders loaded:", list(lm.keys()))
    
    for i in lm:
        print_err("Number of ngrams for order %i:"%i, len(lm[i]))
    total_lprob=0
    total_tokens=0
    for l in sys.stdin:
        l=l.strip()
        ntokens,lprob=get_sent_prob(l, lm,args.order, args.ignore_OOV)
        total_lprob+=lprob
        total_tokens+=ntokens
        ppl=10**-(lprob/ntokens)
        print(lprob, ppl)
    if total_tokens ==0:
        sys.exit(0)
    total_ppl=10**-(total_lprob/total_tokens)
    print("Total:",total_lprob,total_ppl)
