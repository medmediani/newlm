#!/usr/bin/env python
# -*- coding: utf-8 -*-


from optparse import OptionParser
import os, sys, gc, socket
from numpy import *

from functools import reduce

from scipy.optimize import fmin_powell, fmin, fmin_l_bfgs_b


from scipy.special import gammaln, gamma, gammainc, hyperu

import time
from assoc import compute_assoc,methods


os.system("taskset -p 0xFFFFFFFF %d >/dev/stderr" % os.getpid())
BOS="<s>"
EOS="</s>"
WORD_SEP=" "
ARG_SEP=","
ARG_ARG_SEP=":"
SCORE_ARG_SEP="_"
RANGE_SEP="-"
COMMENT_LINE="#"

MAX_ORD=4
DISCOUNT_METHODS=set(["kn"])
ADJUST_METHODS=set(["kn"])
UNINTERPOLATED_METHODS=set(["none"])

ZERO=1e-100
INC_H=1.1
LOG10_ZERO=log10(ZERO)
HIGHEST_POSITIVE_BIN=4
MAX_BINS=30000
MIN_BINS=HIGHEST_POSITIVE_BIN+1
EPS=1e-3
HIGHEST_XI=.9999
XMIN_PERCENT=8.5e-1

DEFAULT_BOW_START=.38
DEFAULT_BOW_STEP=.01
DEFAULT_NPOINTS=100
NBR_KN_CONSTS=3
HIGHEST_COUNT=3

class UnknownEstimatorError(Exception):
    pass

def formatted_span(span):
    total=int(span);
    sec_frac=span-total;
    days=total/(3600*24)
    total%=(3600*24)
    hr=(total)/3600;
    total%=(3600)
    mn=(total)/60;
    sc=(total)%60 + sec_frac;
    if days:
        return "%d-%02d:%02d:%02.4g"%(days, hr, mn, sc)
    else:
        return "%02d:%02d:%02.4g"%(hr, mn, sc)


def load_counts(cf, adjust=False, order=MAX_ORD, full_load=None, min_assocs=None): # assigns=None, min_assocs=None):

    if full_load is not None:
        prefixes=dict((i+1, set()) for i in full_load if i<order)
    else:
        prefixes=None
        
    odicts=dict((i+1, dict()) for i in range(order))
#        attdicts=dict((i+1, dict()) for i in range(order))
#        cngdict=dict((i+1, 0) for  i in range(order))
    for line in cf:
        tup, assoc=line.rsplit(None, 1)
        #Lower order ngrams will be computed while adjustig
        lineord=tup.count(WORD_SEP)+1
        if lineord >order:
            continue
        if prefixes is not None and lineord in prefixes:
            if lineord in min_assocs :
                try:
                    assoc=int(assoc)                
#                odicts[len(tup)-1][WORD_SEP.join(tup[:-2]),tup[-2]]=
                except ValueError:
                    assoc=float(assoc)
                if assoc >= min_assocs[lineord] :
                    prefixes[lineord].add(tup.rsplit(None,1)[0])
            else:
                prefixes[lineord].add(tup.rsplit(None,1)[0])
               
#        prevord=lineord-1
#        rec=min_assocs is not None and prevord in assigns[rank][0] and lineord in min_assocs
#        if rec:
##            print >>sys.stderr,"Recording"
#            try:
#                assoc=int(assoc)
#                
##                odicts[len(tup)-1][WORD_SEP.join(tup[:-2]),tup[-2]]=
#            except ValueError:
#                assoc=float(assoc)
#            if assoc >= (min_assocs[lineord] if lineord in min_assocs else 0):
##                print >>sys.stderr,"Recording22"
#                prefixes[prevord].add(tup.rsplit(None,1)[0])
            
        if full_load is not None:
            if lineord not in full_load:
                ## this is just to remember the order of keys 
                if lineord+1 in full_load:
                    odicts[lineord][tup]=0
                if adjust and lineord != order and (not tup.startswith(BOS )):
                    continue            
        else:
            if lineord==1 :
                odicts[lineord][tup]=0
            elif adjust and lineord != order and (not tup.startswith(BOS )):
               continue
        
#                count=int(tup[-1])
        if  prefixes is None or lineord not in prefixes or lineord not in min_assocs:
            try:
                assoc=int(assoc)                

            except ValueError:
                assoc=float(assoc)
#                if min_freqs is not None:
#                    if len(tup)-1 in min_freqs:
#                        if float(tup[-1]) < min_freqs[len(tup)-1]:
#                            continue
#            if assoc <= 0:
#                print >>sys.stderr,"Inappropriate association in line:",line
          
##        try:   
        odicts[lineord][tup]=assoc
    
#                if "N" in attdicts[len(tup)-1]:
                
#                    attdicts[len(tup)-1]["N"][count]=attdicts[len(tup)-1]["N"].get(count, 0)+1
#                else:
#                    attdicts[len(tup)-1]["N"]={count:1}
#                cngdict[len(tup)-1]+=1
##        except KeyError:
##            pass

#    if BOS not in odicts[1]:
#        odicts[1][BOS]=0
    return odicts, prefixes #, attdicts
 
    
def load_counts_multi(cf, adjust=False, order=MAX_ORD, nscores=1, rank=None, full_load=None, min_assocs=None):
#    with cf as f:
    if nscores<=1:
        return load_counts(cf, adjust, order, rank, full_load,  min_assocs)
    ### TODO add the cutoff to the multiple scores case!!!
    odicts=dict((i+1, dict()) for i in range(order))
#        attdicts=dict((i+1, dict()) for i in range(order))
#        cngdict=dict((i+1, 0) for  i in range(order))
    for line in cf:
        rec=line.rsplit(None, nscores)
        tup, assocs=rec[0], rec[1:]
        
        #Lower order ngrams will be computed while adjustig
        lineord=tup.count(WORD_SEP)+1
#        print >>sys.stderr,"Record=",rec,"N scores:",nscores,"Line ord=",lineord,"Order=",order
#        sys.exit(0)
        if adjust and lineord != order and (not tup.startswith(BOS )) and nscores <= 1:
            continue
        try:
#                count=int(tup[-1])
            try:
                assocs=[int(assocs[0])]+list(map(float, assocs[1:]))
                
                
#                odicts[len(tup)-1][WORD_SEP.join(tup[:-2]),tup[-2]]=
            except ValueError:
                assocs=list(map(float, assocs))
#                if min_freqs is not None:
#                    if len(tup)-1 in min_freqs:
#                        if float(tup[-1]) < min_freqs[len(tup)-1]:
#                            continue
#            if assoc <= 0:
#                print >>sys.stderr,"Inappropriate association in line:",line
            
            odicts[lineord][tup]=assocs
#                if "N" in attdicts[len(tup)-1]:
                
#                    attdicts[len(tup)-1]["N"][count]=attdicts[len(tup)-1]["N"].get(count, 0)+1
#                else:
#                    attdicts[len(tup)-1]["N"]={count:1}
#                cngdict[len(tup)-1]+=1
        except KeyError:
            pass

#    if BOS not in odicts[1]:
#        odicts[1][BOS]=0
    return odicts  #, attdicts
 

#Need to send some parts to other workers
def update_dicts(odicts, assigns, rank, full=False):
#    depend=set(i-1 for i in assigns[rank][0] if i>1)    
    if len(assigns)>1:    
        print("Process:",rank,"assigns=", assigns, file=sys.stderr)   
        for i in odicts:
            print("Process:",rank,"Number of %d-grams:%d"%(i, len(odicts[i])), file=sys.stderr)

        sreqs=[]
        for r in assigns:
            if r != rank:
                if full:
                    tosend=[(i, odicts[i]) for i in odicts ]
                else:
                    depend=set([i-1 for i in assigns[r][0] if i>1]+assigns[r][0])
                    tosend=[(i, odicts[i]) for i in depend]
                    
                print("Process:",rank,"sending to process ", r, ", ".join("%d:%d"%(x, len(y)) for x, y in tosend), file=sys.stderr)   
                sreqs.append(comm.isend(tosend, dest=r, tag=222))
    ##Receive if I have to
    
    if rank in assigns:        
        dict_q = Queue.Queue() 
        sizebutone=sum(1+len(assigns[a][1]) for a in assigns)-1
        print("Process:",rank,"Number of other processes=", sizebutone, file=sys.stderr)
        def receiver():
            for _ in range(sizebutone):
#                if r==rank:
#                    continue
                print("Process:",rank,"receiving from any available worker", file=sys.stderr)
                other={} 
                other=comm.recv(source=MPI.ANY_SOURCE,tag=222) #MPI.ANY_SOURCE, tag=MPI.ANY_TAG)
                dict_q.put(other)
            dict_q.put(StopIteration)
        
        recv_th=threading.Thread(target=receiver)
        recv_th.start()
        receivedd={}
        while True:
            other=dict_q.get()
            if other is StopIteration: break
            print("Process",rank,"received %d dictionaries"%(len(other)), file=sys.stderr)
            
            print("New data at process:", rank, file=sys.stderr)
            
            for i, newd in other:
#                newdkeys=set(newd)
                try:
                    receivedd[i].update( (k,odicts[i].get(k, 0)+newd.get(k) ) for k in newd) #.intersection(odicts[i]))
                except KeyError:
                    receivedd[i]=newd
#                odicts[i].update( (k, newd.get(k)) for k in newdkeys.difference(odicts[i]))
                
                print("Process:",rank,"New number of %d-grams:%d"%(i, len(odicts[i])), file=sys.stderr)
        recv_th.join()        
##Waiti until we are done transmitting data
    for s in sreqs:
        s.wait()
    for i in odicts:
        odicts[i].update( (k,odicts[i].get(k, 0)+receivedd[i].get(k) ) for k in receivedd[i])
    for i in receivedd:
        if i not in odicts:
            odicts[i]=receivedd[i]
    
    
#        s.Free()
    #Remove 

def remove_other_dicts(odicts, assigns, rank, keephigher=False):
    #Remove dicts which we don't need. I need all a in my assignments as well as a-1
    depend=set(i-1 for i in assigns[rank][0] if i>1)
#    print >>sys.stderr, "Process", rank, "My model(s):", assigns[rank][0], "Depend on:", depend

    for r in assigns:
        if r ==rank:
            continue
        for i in  assigns[r][0]:
            if i not in depend and i in odicts: 
                if not keephigher or all([i<a for a in assigns[rank][0]]):
                    print("Process", rank, "Deleting dictionary:", i, file=sys.stderr)
                    del odicts[i]

def adjust_counts_multi(odicts, order, assigns=None, rank=None, keephigher=False, interpolation=True, nscores=1):
    
    orders=sorted(odicts, reverse=True)
    if not orders: return
    minord=orders[-1]
        
    if assigns is not None :
        if not interpolation:
            minord=min(assigns[rank][0])
        else:
            minord=min(set([o-1 for o in assigns[rank][0] if o>1]+assigns[rank][0]))
        orders=[o for o in orders if o>=minord]
        mindictord=min(odicts)
        for i in range(mindictord, minord):
            del odicts[i]
    
    print("Process", rank, "orders=", orders, file=sys.stderr)
    for i in orders[1:]:#range(1, order):
        for ng in odicts[i]:
            if (not ng.startswith(BOS)) or (ng.endswith( BOS)):
                odicts[i][ng][0]=0
    
    for i in orders[:-1]: #range(order, 1, -1):
        if i >1:             
            for ng in odicts[i]:
                try:
                    _, right=ng.split(None, 1)
                except ValueError:
                    right=""
                try:
                    odicts[i-1][right][0]+=1
                except KeyError:
                    odicts[i-1][right]=[0]*nscores
        if assigns is not None and i not in set([o-1 for o in assigns[rank][0] if o>1]+assigns[rank][0]):
            
            if not keephigher or all([i<a for a in assigns[rank][0]]):
#                print >>sys.stderr, "Process", rank, "deleting", i, "Assigns=", assigns[rank][0], "Keephiger=", keephigher, "All=",  all([i<a for a in assigns[rank][0]])
                del odicts[i]
        if not interpolation and i in odicts:
            if (assigns is not None) and (i not in assigns):
                del odicts[i]
            
            
#    if BOS not in od[1]:
#        od[1][BOS]=0
#            except KeyError:
#                odicts[i-1][right,r]=1
#             
#                pass

    
def adjust_counts(odicts, order,  keephigher=False, interpolation=True):
    
    orders=sorted(odicts, reverse=True)
    if not orders: return
    
    print("Orders=", orders, file=sys.stderr)
    for i in orders[1:]:#range(1, order):
        for ng in odicts[i]:
            if (not ng.startswith(BOS)) or (ng.endswith( BOS)):
                odicts[i][ng]=0
    
    for i in orders[:-1]: #range(order, 1, -1):
        if i >1:             
            for ng in odicts[i]:
                try:
                    _, right=ng.split(None, 1)
                except ValueError:
                    right=""
    #            try:
                odicts[i-1][right]=odicts[i-1].get(right, 0)+1
#        if assigns is not None and i not in set([o-1 for o in assigns[rank][0] if o>1]+assigns[rank][0]):
#            
#            if not keephigher or all([i<a for a in assigns[rank][0]]):
##                print >>sys.stderr, "Process", rank, "deleting", i, "Assigns=", assigns[rank][0], "Keephiger=", keephigher, "All=",  all([i<a for a in assigns[rank][0]])
#                del odicts[i]
#        if not interpolation and i in odicts:
#            if (assigns is not None) and (i not in assigns):
#                del odicts[i]
#            
            

def discard_rare(od, min_freq,  mustkeep=None):
    if mustkeep is not None:
        return set(ng for ng in od if od[ng]<min_freq and ng not in mustkeep)
    else:
        return set(ng for ng in od if od[ng]<min_freq )
    

def must_keep_for_lower(od,  rare_ngs):
#    for i, k in enumerate(od):
#        if i>=10: break
#        print >>sys.stderr,"I have to add '", k.rsplit(None, 1)[0], "' to the set"
#    print >>sys.stderr, "List=", [k.rsplit(None, 1)[0] for k in od if k not in rare_ngs][:10]
    return set(k.rsplit(None, 1)[0] for k in od if k not in rare_ngs )
    
        
def tobe_discarded(od, min_freqs):
    torem={}
    
    ng2keep=set()
    for i in range(max(od.keys()), min(min_freqs.keys())-1, -1):
#        print >>sys.stderr,"treating order", i
#        print >>sys.stderr,"Current ngrams to keep", ng2keep
        
        curkeep=set()
        if i>1:
            for ng in ng2keep:
                l, _=ng.rsplit(None, 1)
#                print >>sys.stderr,"Prefix has to be propagated to the next lower order:", l
                curkeep.add(l)
#        if min_freqs[i]>1:
        if i not in min_freqs or min_freqs[i]<=1:
            
#            print >>sys.stderr,"Passing through to the next lower order. No ngrams have to be thrown for order:", i
            if i>1:
                for  ng in od[i]: 
#                    l, _=ng.rsplit(None, 1)
#                    print >>sys.stderr,"Passing prefix:", ng[0]
                    curkeep.add(ng.rsplit(None, 1)[0])
        else:
            torem[i]=set()
#            ngrams=od[i].keys()[:]
            for ng in od[i]: #ngrams:
                
                if od[i][ng]<min_freqs[i] and ng not in ng2keep:
#                    print >>sys.stderr,"Throwing low frequency ngram:", ng
#                    od[i].pop(ng)
                    torem[i].add(ng)
                else:
                    if i>1:
#                        print >>sys.stderr,"Passing prefix:", ng[0]
                        curkeep.add(ng.rsplit(None, 1)[0])
#                    torem.append((i,ng))
        ng2keep=curkeep
    return torem

def compute_smoothing_attributes(acounts, e=None, smoothing=None):    
#    if order in odicts:
    D=[0.0]*(HIGHEST_POSITIVE_BIN-1)
    N=[0]*HIGHEST_POSITIVE_BIN
    if e is None:
        
        
    #        acounts=array(odicts[order].values())
        for i in range(HIGHEST_POSITIVE_BIN):
            N[i]=sum(acounts==i+1)
    #        
    #        for  v in odicts[order].values():
    #            c=int(v)-1
    #            if c<4:
    #                N[c]+=1 
    else:
#        M=e[1:][:HIGHEST_POSITIVE_BIN]
#        print >>sys.stderr, "MAX bin values are:", M
        for i in range(HIGHEST_POSITIVE_BIN):
            N[i]=sum(acounts[(e[i]<=acounts) & (acounts<e[i+1])]) /e[i]
#            M[i]=max(acounts[(e[i]<=acounts) & (acounts<e[i+1])])
            print("N%d=%g, e%d=%g, e%d=%g, Sum=%g"%(i+1, N[i], i+1, e[i], i+2, e[i+1], sum(acounts[(e[i]<=acounts) & (acounts<e[i+1])])), file=sys.stderr) 
               
    print("N=", N, file=sys.stderr)
#    print >>sys.stderr, "M=", M

   
    if e is None:
        y=1./(1+(2.*N[1])/N[0])
        for i in range(HIGHEST_POSITIVE_BIN-1):
            D[i]=max(i+1-(i+2)*y*N[i+1]/N[i],0) #if N[i] >0 else 0
#        D[0]=1./(1+(2.*N[1])/N[0])
#        D[1]=2-3*((D[0]*N[2])/N[1])
#        D[2]=3-4*((D[0]*N[3])/N[2])
    else:
        NNOM=1
        nom=0
        for i in range(NNOM):
            nom+=e[i]*N[i]
        denom=nom+e[NNOM]*N[NNOM]
        y=nom/denom #(e[0]*N[0])/(e[])
#        1./(1+(e[1]/e[0])*(N[1]/N[0]))
        for i in range(HIGHEST_POSITIVE_BIN-1):
            D[i]=max(e[i]-e[i+1]*y*N[i+1]/N[i],0.0) #if N[i] >0 else 0
#        D[0]=e[0]-e[1]*y*N[1]/N[0]
#        D[1]=e[1]-e[2]*y*N[2]/N[1]
#        D[2]=e[2]-e[3]*y*N[3]/N[2]
    return D
#        print >>sys.stderr, "N=",N,"D=", D
                
                

def compute_probs_kn(sdict, inter_dist, xi_func, xi_arg=None, mass_from_arg=False, smoothing_dist="uniform"):
    ctxt_att={}
    mass_from_arg=mass_from_arg and (xi_arg is not None)
    
    if mass_from_arg:
        probs=xi_arg.astype(float) #empty_like(xi_arg)
#        probs[:]=xi_arg
#    else:
#        probs=fromiter(sdict.values(), dtype=float) #fromiter(sdict.itervalues(), dtype=float)
    if True: #xi_arg is None:
        xi_arg=fromiter(sdict.values(), dtype=float) #probs

    mask=xi_arg>0
    
####################    
#    counts=sorted(set(sdict.itervalues()))
#    if counts[0]==0:
#        counts=counts[1:]
#    acounts=array(counts, dtype=float)
#    D=acounts*(1-xi_func(acounts)  )
#    maximum.accumulate(D, out=D)
#    counts_d=dict(zip(counts, D))
#    counts_d[0]=0
#    probs=fromiter(map(lambda v: v-counts_d[v], sdict.itervalues()), dtype=float) 
    probs=fromiter([v*xi_func(v) if v>0 else 0 for v in sdict.values()], dtype=float) 
#    try:
#        D=xi_func(xi_arg  )  
#    except FloatingPointError:
##    if any(xi_arg<=0):
#        probs[mask]*=xi_func(xi_arg[mask])        
        
#################        
#    else:
              
    get_ctxt=lambda ng: ng.rsplit(None, 1) #str.rsplit(ng, None, 1)
    BOS_ind=None
#    del mask
    for i, (ng, sc) in enumerate(sdict.items()):
    
        if ng==BOS: #if sc<=0:
#            print >>sys.stderr, "Score('<s>')=", sc
            BOS_ind=i
            continue    
#        if ng=="<unk>":
#            print >>sys.stderr, "Counts of <unk>=", sc
        
        try:
            l, _=get_ctxt(ng) #.rsplit(None, 1) 
        except ValueError:
            get_ctxt=lambda ng:("", None)
            l=""
        try:
            item=ctxt_att[l]
            item[0]+=sc
            item[1]+= xi_arg[i]
            item[2]+= probs[i]
#            ctxt_att[l][0]+=sc
#            ctxt_att[l][1]+= xi_arg[i]
#            ctxt_att[l][2]+= probs[i]
        except KeyError:
#            ctxt_att[l]=[sc, xi_arg[i], probs[i]]
            ctxt_att[l]=fromiter((sc, xi_arg[i], probs[i]), dtype=float)
    if mass_from_arg:
        for ctxt in ctxt_att:
            item=ctxt_att[ctxt]
            item[0]*=item[1]/item[2]
            item[2]=1-item[2]/item[1]
        probs[:]=fromiter(sdict.values(), dtype=float)
    else:
        for ctxt in ctxt_att:
            item=ctxt_att[ctxt]
            item[2]=1-item[2]/item[0]
#            ctxt_att[ctxt][2]=1-ctxt_att[ctxt][2]/ctxt_att[ctxt][0
    if len(ctxt_att)==1: #simpler computation for order 1
        _, (norm, _, gained_mass)=ctxt_att.popitem()
#        print >>sys.stderr, "Gained=", gained_mass, "Normalizer=", norm
#        sys.exit(0)
        mask[:]=1 #probs>0
        mask[BOS_ind]=0
        probs[mask]=probs[mask]/norm 
        if any(probs[mask]<=0) and smoothing_dist=="zerotons-only":
            mask[:]=probs<=0
            mask[BOS_ind]=0
            print("The freed mass will be distributed over %d words (with 0-probs)"%(sum(mask)), file=sys.stderr)
            probs[mask]+= gained_mass/sum(mask)            
        else:
            print("Redistributing the freed mass over all unigrams", end=' ', file=sys.stderr) 
            if smoothing_dist=="uniform":
                print("(uniformly)", file=sys.stderr)
                probs[mask]+= gained_mass/sum(mask) #len(probs) 
            elif smoothing_dist=="unigram":
                print("(depending on the unigram probability)", file=sys.stderr)
                rel_f=fromiter(sdict.values(), dtype=float) +1
                rel_f/=sum(rel_f)
                probs[mask]+= gained_mass*rel_f[mask]
            elif smoothing_dist=="cunigram":
                print("(depending on the count unigram probability)", file=sys.stderr)
                rel_f=xi_arg.astype(float) 
                rel_f/=sum(rel_f)
                probs[mask]+= gained_mass*rel_f[mask]
        return  probs #ctxt_att[""][0]+ctxt_att[""][2]/len(probs)
#    if False:
#        other_ctxt={}        
#        for i, (ng, s) in enumerate(sdict.iteritems()):
#            l, r=ng.rsplit(None, 1)[0],  ng.split(None, 1) [1]
#            swp, sp=other_ctxt.get(l, (0, 0))
#            p=inter_dist[r]
#            other_ctxt[l]=(swp+s*p, sp+p)
#        for i, (ng, s) in enumerate(sdict.iteritems()):
#            l, r=ng.rsplit(None, 1)[0],  ng.split(None, 1) [1]
#    #        _, right=ng.split(None, 1) 
#            swp, sp=other_ctxt[l]
#            probs[i]=probs[i]/ctxt_att[l][0]+ctxt_att[l][2]*inter_dist[r]*s*sp/swp        
#        return probs
    for i, (ng) in enumerate(sdict):
        l, r=ng.rsplit(None, 1)[0],  ng.split(None, 1) [1]
#        _, right=ng.split(None, 1) 
        probs[i]=probs[i]/ctxt_att[l][0]+ctxt_att[l][2]*inter_dist[r]
    return probs

def compute_prob_args_kn(odicts, order, D=None,lprobs=None,discount=False, e=None, xi_fun=None, freqd=None):
#    D, lprobs=args
####
    if xi_fun is not None:
    #        Since computing dfun is time consuming lets compute it only once for each data point
#        kvals={}
        if freqd is None or freqd[order] is None:
            Di=fromiter(odicts[order].values(), dtype=float)
#            d=odicts[order]
        else:
            Di=freqd[order].astype(float)
#            d=freqd[order]
#        f=fromiter(set(v for v in d.itervalues() if v >0), dtype=int)
#        kvals=dict(zip(f, f*(1-xi_fun(f))))
#        kvals[0]=0.
        dfun=lambda x: x #kvals[x]
#        _, ui=unique(acounts if freqd is None else freqs, return_index=True)
#        if freqd is None:
#            for v in acounts[ui]:
#                try:
#                    kvals[v]=dfuns[i]["func"](v) 
#                except FloatingPointError:
#                    print >> sys.stderr, "Problem in order %d for count value %g"%(i, v)
#                    kvals[v]=dfuns[i]["func"](v)
#            
#            if options.force_increasing_discounts:
#                kvals=dict(zip(kvals.keys(), increaseit(kvals.keys(), kvals.values())))
#        else:
#            for v in freqs[ui]:
#                kvals[v]=dfuns[i]["xi"](v) if v>0 else 0
#                    
                    ####
    def get_discount(ng):
#        x, y=None):
        try:
            return dfun(odicts[order][ng])
        except TypeError:
            return dfun(odicts[order][ng], freqd[order][ng])
    
#    dfun=None
    if  order in odicts:
        nitems=len(odicts[order])
        if D is None or not discount:
            Di=0.0
            interp=0.0
        else:
            if False:                
                if dfun is None:
                    D=array(D)
                    e=array(e)
                    print("Discount constants through function: D=",D,"e=",e, file=sys.stderr)
                    dfun=discount_params(D[D>0], e[D>0])
                print("Discount(1)=", dfun(1.), file=sys.stderr)
                if e is not None:
                    print("Smallest discount(%g)= %g"%(e [0],  dfun(e[0])), file=sys.stderr)
                else:
                    mins=1e-5
                    print("Smallest discount(%g)= %g"%(mins,  dfun(mins)), file=sys.stderr)
#            Di=fromiter(d.itervalues(), dtype=float)
#            if freqd is None or freqd[order] is None:
#                V=Di
#            else:                
#                V=fromiter(odicts[order].itervalues(), dtype=float)
                
            mask=Di>0
            Di[mask]=Di*(1-xi_fun(Di[mask]))
            del mask
#            if freqd is not None and freqd[order] is not None:
#                del V
#            Di=fromiter((kvals[v] for v in odicts[order].itervalues()), dtype=float) #zeros(nitems) 
#    del kvals
#        if order==1:
#            print >> sys.stderr, lprobs
#            leftcounts={"":zeros(3)}
#            for i, ng in enumerate(odicts[order]):
##                left, _=ng.rsplit(None, 1)
#                Di[i]=D[min(3, odicts[order][ng])-1]
#            return Di, 0, 0
#        
#        print >>sys.stderr, "Len(dict(order=%d))=%d"%(order, len(odicts[order]))
        leftcounts={} #zeros(nitems)
        for i, (ng, c) in enumerate(odicts[order].items()):
            index=None
            
            if D is not None and discount:
                if e is None:
                    Di[i]=D[min(HIGHEST_POSITIVE_BIN-1, c)-1]
                else:
                    
#                    print >>sys.stderr, "Positioning",odicts[order][l,r],"in ",e
                    if dfun is not None:
#                        Di[i]=kvals[c] #get_discount(ng)
                        if c>0 and (Di[i]>= c or Di[i]<0):
                            print("WARNING: ngram= '%s'"%ng,"Occurrence=", c,"Di=",Di[i], file=sys.stderr)
                    else:
                        if c>=e[0]:
                            index=HIGHEST_POSITIVE_BIN-1
                            for j in range(1,HIGHEST_POSITIVE_BIN-1):
                                if c<e[j]:
                                    index=j
                                    break
                        if index is not None:
    #                        print >>sys.stderr,"Index found=",index,"D=D",index-1
                            Di[i]=D[index-1]
    #                    else:
#                        print >>sys.stderr,"Index not found, leaving NULL"
#                    if e[0]<=odicts[order][l,r]<e[1]:
#                        Di[i]=D[0]
#                    elif e[1]<=odicts[order][l,r]<e[2]:
#                        Di[i]=D[1]
#                    elif e[2]<=odicts[order][l,r]:
#                        Di[i]=D[2]
             
            if c<=0:
#                print >> sys.stderr, "%d-gram '%s' has occurrences: %d"%(order, ng, odicts[order][ng])
                continue       
            if order>1:
                l, _=ng.rsplit(None, 1)
            else:
                l=""
            try:
                if D is not None and discount:
                    leftcounts[l][0]+=Di[i]
#                if e is None:
#                    leftcounts[l][0]+=D[min(HIGHEST_POSITIVE_BIN-1, odicts[order][l,r])-1]
#                else:
#                    if index is None:
#                        if odicts[order][l,r]>=e[0]:
#                            index=HIGHEST_POSITIVE_BIN-1
#                            for j in range(1,HIGHEST_POSITIVE_BIN-1):
#                                if odicts[order][l,r]<e[j]:
#                                    index=j
#                                    break
#                    if index is not None:
#                        leftcounts[l][0]+=D[index-1]
#                    if e[0]<=odicts[order][l,r]<e[1]:
#                        leftcounts[l][0]+=1 #odicts[order][ng]/e[0]
#                    elif e[1]<=odicts[order][l,r]<e[2]:
#                        leftcounts[l][1]+=1 #odicts[order][ng]/e[1]
#                    elif e[2]<=odicts[order][l,r]:
#                        leftcounts[l][2]+=1 #odicts[order][ng]/e[2]
                leftcounts[l][1]+=c #odicts[order][ng]
            except KeyError: 
                leftcounts[l]=zeros(2)#, dtype=int)    
                if D is not None and discount:
                    leftcounts[l][0]=Di[i]      
#                if e is None:
#                    leftcounts[l][0]=D[min(HIGHEST_POSITIVE_BIN-1, odicts[order][l,r])-1]
#                else:
#                    if index is None:
#                        if odicts[order][l,r]>=e[0]:
#                            index=HIGHEST_POSITIVE_BIN-1
#                            for j in range(1,HIGHEST_POSITIVE_BIN-1):
#                                if odicts[order][l,r]<e[j]:
#                                    index=j
#                                    break
#                    if index is not None:
#                        leftcounts[l][0]=D[index-1]
#                    if e[0]<=odicts[order][l,r]<e[1]:
#                        leftcounts[l][0]=1 #odicts[order][ng]/e[0]
#                    elif e[1]<=odicts[order][l,r]<e[2]:
#                        leftcounts[l][1]=1 #odicts[order][ng]/e[1]
#                    elif e[2]<=odicts[order][l,r]:
#                        leftcounts[l][2]=1 #odicts[order][ng]/e[2]
                leftcounts[l][1]=c #odicts[order][ng]

        if order==1: #lprobs is None:
            if D is not None:
                
                interp=leftcounts[""][0]/((nitems)*leftcounts[""][1])
#                print >>sys.stderr, "Leftcounts for the empty word=",leftcounts[""], "x", D
#                print >>sys.stderr,"Sum of all occurrences:",sum(od[1].values())
#### VErification
#            weights={}
#            for l in leftcounts:
#                weights[l]=leftcounts[l][0]/leftcounts[l][1]
#            return Di, leftcounts[""][1], interp ,weights
#######ENd Verif
#            print >>sys.stderr,"Sum(Di)=", leftcounts[""][0]

            return Di, float(leftcounts[""][1]), interp #,weights
#        print >>sys.stderr, "First ten values of the previous probs:", lprobs.items()[:10]
#        print >>sys.stderr, "First ten values of the leftcounts:", leftcounts.items()[:10]
#        sys.exit(0)
        leftall=empty(nitems, dtype=float) #zeros(nitems)
        if lprobs is not None:
            interp= zeros(nitems)
            
        if D is None:
            for i, (ng) in enumerate(odicts[order]): 
    #            if order>1:
                l, _=ng.rsplit(None, 1)
    #            else:
    ##                r=ng.rsplit(None, 1)
    #                l=""
    #            if c<=0:
    #                leftall[i]= interp[i]= 0.
    #                continue
    #            left, _=ng.rsplit(None, 1) #left context
    #            try:
                _, right=ng.split(None, 1) #ng.split(None, 1) #drop earliest
    #            except ValueError:
    #                right=ng
                
    #            print >>sys.stderr, right, "\t", 
    #            if leftcounts[l][1] <=0:
    #                print >>sys.stderr,"Leftcount=0 for left context:",l
                leftall[i]=leftcounts[l][1]
            return Di, leftall, interp
        for i, (ng) in enumerate(odicts[order]): 
#            if order>1:
            l, _=ng.rsplit(None, 1)
#            else:
##                r=ng.rsplit(None, 1)
#                l=""
#            if c<=0:
#                leftall[i]= interp[i]= 0.
#                continue
#            left, _=ng.rsplit(None, 1) #left context
#            try:
            _, right=ng.split(None, 1) #ng.split(None, 1) #drop earliest
#            except ValueError:
#                right=ng
            
#            print >>sys.stderr, right, "\t", 
#            if leftcounts[l][1] <=0:
#                print >>sys.stderr,"Leftcount=0 for left context:",l
            leftall[i]=leftcounts[l][1]
#            if leftcounts[left][3]==0:
#                print >>sys.stderr, "0 counts for:", left
            
#                try:
            interp[i]=leftcounts[l][0]*lprobs[right]/leftcounts[l][1]
#                except KeyError:
#                    print >>sys.stderr, "************** Could not find ngram '%s' in lower-order while processing ngram '%s'"% (right, ng)
#                    interp[i]=0.
#                print >>sys.stderr, "Interpolation weight:",dot(leftcounts[left][:3], D[:3])/leftcounts[left][3] , "Previous prob=", lprobs[right]
#        Ns=zeros(HIGHEST_POSITIVE_BIN)
#        for l in leftcounts:
#            Ns+=leftcounts[l]
#        print >>sys.stderr, "All Ns=",Ns
######verification
#        weights={}
#        for l in leftcounts:
#            weights[l]=leftcounts[l][0]/leftcounts[l][1]
#        return Di, leftall, interp ,weights
##### End verif
        return Di, leftall, interp #,weights
   
##########################

def compute_probs_wb(sdict, inter_dist, xi_func, xi_arg=None, mass_from_arg=False, smoothing_dist="uniform"):
    
    ctxt_att={}
    mass_from_arg=mass_from_arg and (xi_arg is not None)

    get_ng_attributes=lambda sc, arg: (1, sc, arg )
    get_discount_param=lambda attributes: attributes[0]/attributes[2]
#    if not mass_from_arg:
#        get_discount_param=lambda attributes: attributes[0]/attributes[1]
#    else:
#        #freqd[order][i]
#        
#    if mass_from_arg:
#        probs=xi_arg.astype(float) #empty_like(xi_arg)
##        probs[:]=xi_arg
#    else:
    probs=fromiter(list(sdict.values()), dtype=float) #fromiter(sdict.itervalues(), dtype=float)
    if xi_arg is None:
        xi_arg=probs
    
#
#    try:
#        probs*=xi_func(xi_arg  )  
#    except FloatingPointError:
##    if any(xi_arg<=0):
#        mask=xi_arg>0
#        probs[mask]*=xi_func(xi_arg[mask])        
#    else:
    mask=xi_arg>0
    
    get_ctxt=lambda ng: ng.rsplit(None, 1) #str.rsplit(ng, None, 1)
    BOS_ind=None
#    del mask
    for i, (ng, sc) in enumerate(sdict.items()):
        if ng==BOS: #if sc<=0:
            BOS_ind=i
            continue    
#        if ng=="<unk>":
#            print >>sys.stderr, "Counts of <unk>=", sc
            
        try:
            l, _=get_ctxt(ng) #.rsplit(None, 1)
        except ValueError:
            get_ctxt=lambda ng:("", None)
            l=""
        try:
            ctxt_att[l]+=get_ng_attributes(sc, xi_arg[i])
        
        except KeyError:
#            ctxt_att[l]=[sc, xi_arg[i], probs[i]]
            ctxt_att[l]=fromiter(get_ng_attributes(sc, xi_arg[i]), dtype=float)
    for ctxt, att in ctxt_att.items():
        ctxt_att[ctxt]=att[1], xi_func(get_discount_param(att))
    
#    #################################
#        if order==1: #lprobs is None:
#            if D is not None:
#                locxi=xi_fun(get_discount_param(ctxt_att[""]))
##                if freqd is None or freqd[order] is None:
##                    locxi=xi_fun(leftcounts[""][0]/leftcounts[""][1])   
##                else:
##                    locxi=xi_fun(leftcounts[""][0]/leftcounts[""][2])
#                    
#                interp=(1-locxi)/((nitems))
##                print >>sys.stderr, "Leftcounts for the empty word=",leftcounts[""], "x", D
##                print >>sys.stderr,"Sum of all occurrences:",sum(od[1].values())
##### VErification
##            weights={}
##            for l in leftcounts:
##                weights[l]=leftcounts[l][0]/leftcounts[l][1]
##            return Di, leftcounts[""][1], interp ,weights
########ENd Verif
##            print >>sys.stderr,"Sum(Di)=", leftcounts[""][0]
##            print >>file("o1.inter.%s"%(time.time()),"wb"),  interp
##            print >> file("o1.lc.%s"%(time.time()), "wb"), (leftcounts[""][1])
##            savetxt("o1.Di.%s"%(time.time()),  (1-leftcounts[""][0])*fromiter((odicts[order].itervalues()), dtype=float))
##            savetxt("o1.V.%s"%(time.time()),  fromiter((odicts[order].itervalues()), dtype=float))
#            return (1-locxi)*fromiter((odicts[order].itervalues()), dtype=float), (leftcounts[""][1]), interp #,weights
#            
#            #######################
#    if mass_from_arg:
#        for ctxt in ctxt_att:
#            item=ctxt_att[ctxt]
#            item[0]*=item[1]/item[2]
#            item[2]=1-item[2]/item[1]
#        probs[:]=fromiter(sdict.values(), dtype=float)
#    else:
#        for ctxt in ctxt_att:
#            item=ctxt_att[ctxt]
#            item[2]=1-item[2]/item[0]
#            ctxt_att[ctxt][2]=1-ctxt_att[ctxt][2]/ctxt_att[ctxt][0]
    if len(ctxt_att)==1: #simpler computation for order 1
        _, (norm, xival)=ctxt_att.popitem()
        gained_mass=1-xival
#        print >>sys.stderr, "Normalizer=", norm
#        print >>sys.stderr, "XI value=", xival
#        print >>sys.stderr, "Gained mass=", gained_mass
        

        mask[:]=1 #probs>0
        mask[BOS_ind]=0
        probs[BOS_ind]=0
        probs[mask]*=xival/norm 
        
#        print >>sys.stderr, "Sum of discounted probs=", probs[mask].sum()
#        print >>sys.stderr, "P['<s>']=", probs[BOS_ind]
        if False: #any(probs[mask]<=0):
            mask[:]=probs<=0
            mask[BOS_ind]=0
            print("The freed mass will be distributed over %d words (with 0-probs)"%(sum(mask)), file=sys.stderr)
            probs[mask]+= gained_mass/sum(mask)            
        else:
            print("Redistributing the freed mass over all unigrams", end=' ', file=sys.stderr) 
            if smoothing_dist=="uniform":
                print("(uniformly)", file=sys.stderr)
                probs[mask]+= gained_mass/sum(mask) #len(probs) 
            elif smoothing_dist=="unigram":
                print("(depending on the unigram probability)", file=sys.stderr)
                rel_f=fromiter(list(sdict.values()), dtype=float) 
                rel_f/=sum(rel_f)
                probs[mask]+= gained_mass*rel_f[mask]
            elif smoothing_dist=="cunigram":
                print("(depending on the count unigram probability)", file=sys.stderr)
                rel_f=xi_arg.astype(float) 
                rel_f/=sum(rel_f)
                probs[mask]+= gained_mass*rel_f[mask]
#        sys.exit(0)
        return  probs #ctxt_att[""][0]+ctxt_att[""][2]/len(probs)
    for i, (ng) in enumerate(sdict):
        l, r=ng.rsplit(None, 1)[0],  ng.split(None, 1) [1] 
        norm, xival=ctxt_att[l]
#        _, right=ng.split(None, 1) 
        probs[i]=probs[i]*xival/norm+(1-xival)*inter_dist[r]
    return probs
##############################

##########################

def compute_probs_comb(sdict, inter_dist, xi_func, xi_arg=None, mass_from_arg=False, smoothing_dist="uniform"):
    
    ctxt_att={}
    mass_from_arg=mass_from_arg and (xi_arg is not None)

    get_ng_attributes=lambda sc, arg: (1, sc, arg )
    get_discount_param=lambda attributes: attributes[0]/attributes[2]
#    if not mass_from_arg:
#        get_discount_param=lambda attributes: attributes[0]/attributes[1]
#    else:
#        #freqd[order][i]
#        
#    if mass_from_arg:
#        probs=xi_arg.astype(float) #empty_like(xi_arg)
##        probs[:]=xi_arg
#    else:
    probs=fromiter(list(sdict.values()), dtype=float) #fromiter(sdict.itervalues(), dtype=float)
    if xi_arg is None:
        xi_arg=probs
    
#
#    try:
#        probs*=xi_func(xi_arg  )  
#    except FloatingPointError:
##    if any(xi_arg<=0):
#        mask=xi_arg>0
#        probs[mask]*=xi_func(xi_arg[mask])        
#    else:
    mask=xi_arg>0
    
    get_ctxt=lambda ng: ng.rsplit(None, 1) #str.rsplit(ng, None, 1)
    BOS_ind=None
#    del mask
    for i, (ng, sc) in enumerate(sdict.items()):
        if ng==BOS: #if sc<=0:
            BOS_ind=i
            continue    
#        if ng=="<unk>":
#            print >>sys.stderr, "Counts of <unk>=", sc
            
        try:
            l, _=get_ctxt(ng) #.rsplit(None, 1)
        except ValueError:
            get_ctxt=lambda ng:("", None)
            l=""
        try:
            ctxt_att[l]+=get_ng_attributes(sc, xi_arg[i])
        
        except KeyError:
#            ctxt_att[l]=[sc, xi_arg[i], probs[i]]
            ctxt_att[l]=fromiter(get_ng_attributes(sc, xi_arg[i]), dtype=float)
    for ctxt, att in ctxt_att.items():
        ctxt_att[ctxt]=att[1], get_discount_param(att)
    
#    #################################
#        if order==1: #lprobs is None:
#            if D is not None:
#                locxi=xi_fun(get_discount_param(ctxt_att[""]))
##                if freqd is None or freqd[order] is None:
##                    locxi=xi_fun(leftcounts[""][0]/leftcounts[""][1])   
##                else:
##                    locxi=xi_fun(leftcounts[""][0]/leftcounts[""][2])
#                    
#                interp=(1-locxi)/((nitems))
##                print >>sys.stderr, "Leftcounts for the empty word=",leftcounts[""], "x", D
##                print >>sys.stderr,"Sum of all occurrences:",sum(od[1].values())
##### VErification
##            weights={}
##            for l in leftcounts:
##                weights[l]=leftcounts[l][0]/leftcounts[l][1]
##            return Di, leftcounts[""][1], interp ,weights
########ENd Verif
##            print >>sys.stderr,"Sum(Di)=", leftcounts[""][0]
##            print >>file("o1.inter.%s"%(time.time()),"wb"),  interp
##            print >> file("o1.lc.%s"%(time.time()), "wb"), (leftcounts[""][1])
##            savetxt("o1.Di.%s"%(time.time()),  (1-leftcounts[""][0])*fromiter((odicts[order].itervalues()), dtype=float))
##            savetxt("o1.V.%s"%(time.time()),  fromiter((odicts[order].itervalues()), dtype=float))
#            return (1-locxi)*fromiter((odicts[order].itervalues()), dtype=float), (leftcounts[""][1]), interp #,weights
#            
#            #######################
#    if mass_from_arg:
#        for ctxt in ctxt_att:
#            item=ctxt_att[ctxt]
#            item[0]*=item[1]/item[2]
#            item[2]=1-item[2]/item[1]
#        probs[:]=fromiter(sdict.values(), dtype=float)
#    else:
#        for ctxt in ctxt_att:
#            item=ctxt_att[ctxt]
#            item[2]=1-item[2]/item[0]
#            ctxt_att[ctxt][2]=1-ctxt_att[ctxt][2]/ctxt_att[ctxt][0]
    if len(ctxt_att)==1: #simpler computation for order 1
        _, (norm, y_xiparam)=ctxt_att.popitem()
#        print >>sys.stderr, "Normalizer=", norm
#        print >>sys.stderr, "XI value=", xival
#        print >>sys.stderr, "Gained mass=", gained_mass
        

        mask[:]=1 #probs>0
        mask[BOS_ind]=0
        probs[BOS_ind]=0
        
        probs[mask]*=xi_func(probs[mask], y_xiparam)/norm
        
        gained_mass=1-sum(probs[mask])
        print("Gained mass=", gained_mass, file=sys.stderr)
#        exit(0)
#        print >>sys.stderr, "Sum of discounted probs=", probs[mask].sum()
#        print >>sys.stderr, "P['<s>']=", probs[BOS_ind]
        if False: #any(probs[mask]<=0):
            mask[:]=probs<=0
            mask[BOS_ind]=0
            print("The freed mass will be distributed over %d words (with 0-probs)"%(sum(mask)), file=sys.stderr)
            probs[mask]+= gained_mass/sum(mask)            
        else:
            print("Redistributing the freed mass over all unigrams", end=' ', file=sys.stderr) 
            if smoothing_dist=="uniform":
                print("(uniformly)", file=sys.stderr)
                probs[mask]+= gained_mass/sum(mask) #len(probs) 
            elif smoothing_dist=="unigram":
                print("(depending on the unigram probability)", file=sys.stderr)
                rel_f=fromiter(dict.values(), dtype=float) +1
                rel_f/=sum(rel_f)
                probs[mask]+= gained_mass*rel_f[mask]
            elif smoothing_dist=="cunigram":
                print("(depending on the count unigram probability)", file=sys.stderr)
                rel_f=xi_arg.astype(float) +1
                rel_f/=sum(rel_f)
                probs[mask]+= gained_mass*rel_f[mask]
#        sys.exit(0)
        return  probs #ctxt_att[""][0]+ctxt_att[""][2]/len(probs)
    total_mass={}
    for i, (ng) in enumerate(sdict):
        l=ng.rsplit(None, 1)[0]
        norm, y_xiparam=ctxt_att[l]
        probs[i]=probs[i]*xi_func(probs[i], y_xiparam)/norm
        try:
            total_mass[l]+=probs[i]
        except KeyError:
            total_mass[l]=probs[i]
        
    for i, (ng) in enumerate(sdict):
        l, r=ng.rsplit(None, 1)[0],  ng.split(None, 1) [1] 
#        norm, y_xiparam=ctxt_att[l]
#        _, right=ng.split(None, 1) 
        probs[i]+=(1-total_mass[l])*inter_dist[r]
    return probs

########################

def compute_prob_args_wb(odicts, order, D=None,lprobs=None,discount=False, e=None, xi_fun=None, freqd=None):
#    D, lprobs=args
####
   
    dfun=lambda x: x #kvals[x]
#    print >>sys.stderr,"Computing args for WB"
#        _, ui=unique(acounts if freqd is None else freqs, return_index=True)
#        if freqd is None:
#            for v in acounts[ui]:
#                try:
#                    kvals[v]=dfuns[i]["func"](v) 
#                except FloatingPointError:
#                    print >> sys.stderr, "Problem in order %d for count value %g"%(i, v)
#                    kvals[v]=dfuns[i]["func"](v)
#            
#            if options.force_increasing_discounts:
#                kvals=dict(zip(kvals.keys(), increaseit(kvals.keys(), kvals.values())))
#        else:
#            for v in freqs[ui]:
#                kvals[v]=dfuns[i]["xi"](v) if v>0 else 0
#                    
                    ####
    if freqd is None or freqd[order] is None:
        get_ng_attributes=lambda i, c: 1, c
        get_discount_param=lambda attributes: attributes[0]/attributes[1]
    else:
        get_ng_attributes=lambda i, c: 1, c, freqd[order][i]
        get_discount_param=lambda attributes: attributes[0]/attributes[2]
        
    def get_discount(ng):
#        x, y=None):
        try:
            return dfun(odicts[order][ng])
        except TypeError:
            return dfun(odicts[order][ng], freqd[order][ng])
    
#    dfun=None
    if  order in odicts:
        nitems=len(odicts[order])
#        if order==1:
#            print >> sys.stderr, lprobs
#            leftcounts={"":zeros(3)}
#            for i, ng in enumerate(odicts[order]):
##                left, _=ng.rsplit(None, 1)
#                Di[i]=D[min(3, odicts[order][ng])-1]
#            return Di, 0, 0
#        
#        print >>sys.stderr, "Len(dict(order=%d))=%d"%(order, len(odicts[order]))
        leftcounts={} #zeros(nitems)
        for i, (ng, c) in enumerate(odicts[order].items()):
            index=None
            
    #                    else:
#                        print >>sys.stderr,"Index not found, leaving NULL"
#                    if e[0]<=odicts[order][l,r]<e[1]:
#                        Di[i]=D[0]
#                    elif e[1]<=odicts[order][l,r]<e[2]:
#                        Di[i]=D[1]
#                    elif e[2]<=odicts[order][l,r]:
#                        Di[i]=D[2]
             
            if c<=0:
#                print >> sys.stderr, "%d-gram '%s' has occurrences: %d"%(order, ng, odicts[order][ng])
                continue       
            if order>1:
                l, _=ng.rsplit(None, 1)
            else:
                l=""
            try:
                leftcounts[l]+=get_ng_attributes(i, c) #(1, c) #Di[i]
#                if e is None:
#                    leftcounts[l][0]+=D[min(HIGHEST_POSITIVE_BIN-1, odicts[order][l,r])-1]
#                else:
#                    if index is None:
#                        if odicts[order][l,r]>=e[0]:
#                            index=HIGHEST_POSITIVE_BIN-1
#                            for j in range(1,HIGHEST_POSITIVE_BIN-1):
#                                if odicts[order][l,r]<e[j]:
#                                    index=j
#                                    break
#                    if index is not None:
#                        leftcounts[l][0]+=D[index-1]
#                    if e[0]<=odicts[order][l,r]<e[1]:
#                        leftcounts[l][0]+=1 #odicts[order][ng]/e[0]
#                    elif e[1]<=odicts[order][l,r]<e[2]:
#                        leftcounts[l][1]+=1 #odicts[order][ng]/e[1]
#                    elif e[2]<=odicts[order][l,r]:
#                        leftcounts[l][2]+=1 #odicts[order][ng]/e[2]
#                leftcounts[l][1]+=c #odicts[order][ng]
            except KeyError: 
                leftcounts[l]=array(get_ng_attributes(i, c), dtype=float) #zeros(2)#, dtype=int)    
#                if D is not None and discount:
#                    leftcounts[l][0]= 1 #Di[i]      
#                if e is None:
#                    leftcounts[l][0]=D[min(HIGHEST_POSITIVE_BIN-1, odicts[order][l,r])-1]
#                else:
#                    if index is None:
#                        if odicts[order][l,r]>=e[0]:
#                            index=HIGHEST_POSITIVE_BIN-1
#                            for j in range(1,HIGHEST_POSITIVE_BIN-1):
#                                if odicts[order][l,r]<e[j]:
#                                    index=j
#                                    break
#                    if index is not None:
#                        leftcounts[l][0]=D[index-1]
#                    if e[0]<=odicts[order][l,r]<e[1]:
#                        leftcounts[l][0]=1 #odicts[order][ng]/e[0]
#                    elif e[1]<=odicts[order][l,r]<e[2]:
#                        leftcounts[l][1]=1 #odicts[order][ng]/e[1]
#                    elif e[2]<=odicts[order][l,r]:
#                        leftcounts[l][2]=1 #odicts[order][ng]/e[2]
#                leftcounts[l][1]=c #odicts[order][ng]
#        print >>sys.stderr, "The first item before:"
#        for x, y in leftcounts.iteritems():
#            print >>sys.stderr, x, "==>", y
#            break
        
#        dis=file("disc.%s"%(time.time()),"wb")
#        for v in leftcounts.itervalues():
#            print >>dis,v,xi_fun(v[0]/v[1]) 
#            v[0]=xi_fun(v[0]/v[1])  
#        dis.close()
#        print >>sys.stderr, "The first item after:"
#        for x, y in leftcounts.iteritems():
#            print >>sys.stderr, x, "==>", y
#            break
            
        if order==1: #lprobs is None:
            if D is not None:
                locxi=xi_fun(get_discount_param(leftcounts[""]))
#                if freqd is None or freqd[order] is None:
#                    locxi=xi_fun(leftcounts[""][0]/leftcounts[""][1])   
#                else:
#                    locxi=xi_fun(leftcounts[""][0]/leftcounts[""][2])
                    
                interp=(1-locxi)/((nitems))
#                print >>sys.stderr, "Leftcounts for the empty word=",leftcounts[""], "x", D
#                print >>sys.stderr,"Sum of all occurrences:",sum(od[1].values())
#### VErification
#            weights={}
#            for l in leftcounts:
#                weights[l]=leftcounts[l][0]/leftcounts[l][1]
#            return Di, leftcounts[""][1], interp ,weights
#######ENd Verif
#            print >>sys.stderr,"Sum(Di)=", leftcounts[""][0]
#            print >>file("o1.inter.%s"%(time.time()),"wb"),  interp
#            print >> file("o1.lc.%s"%(time.time()), "wb"), (leftcounts[""][1])
#            savetxt("o1.Di.%s"%(time.time()),  (1-leftcounts[""][0])*fromiter((odicts[order].itervalues()), dtype=float))
#            savetxt("o1.V.%s"%(time.time()),  fromiter((odicts[order].itervalues()), dtype=float))
            return (1-locxi)*fromiter((iter(odicts[order].values())), dtype=float), (leftcounts[""][1]), interp #,weights
#        print >>sys.stderr, "First ten values of the previous probs:", lprobs.items()[:10]
#        print >>sys.stderr, "First ten values of the leftcounts:", leftcounts.items()[:10]
#        sys.exit(0)
        leftall=empty(nitems, dtype=float) #zeros(nitems)
        if D is not None:
            Di=zeros(nitems) 
        if lprobs is not None:
            interp= zeros(nitems)
            
        if D is None:
            for i, (ng) in enumerate(odicts[order]): 
    #            if order>1:
                l, _=ng.rsplit(None, 1)
    #            else:
    ##                r=ng.rsplit(None, 1)
    #                l=""
    #            if c<=0:
    #                leftall[i]= interp[i]= 0.
    #                continue
    #            left, _=ng.rsplit(None, 1) #left context
    #            try:
                _, right=ng.split(None, 1) #ng.split(None, 1) #drop earliest
    #            except ValueError:
    #                right=ng
                
    #            print >>sys.stderr, right, "\t", 
    #            if leftcounts[l][1] <=0:
    #                print >>sys.stderr,"Leftcount=0 for left context:",l
                leftall[i]=leftcounts[l][1]
            return Di, leftall, interp
        for i, (ng) in enumerate(odicts[order]): #.iteritems()): 
#            if order>1:
            l, _=ng.rsplit(None, 1)
#            else:
##                r=ng.rsplit(None, 1)
#                l=""
#            if c<=0:
#                leftall[i]= interp[i]= 0.
#                continue
#            left, _=ng.rsplit(None, 1) #left context
#            try:
            _, right=ng.split(None, 1) #ng.split(None, 1) #drop earliest
#            except ValueError:
#                right=ng
            
#            print >>sys.stderr, right, "\t", 
#            if leftcounts[l][1] <=0:
#                print >>sys.stderr,"Leftcount=0 for left context:",l
            
            leftall[i]=leftcounts[l][1]
#            if leftcounts[left][3]==0:
#                print >>sys.stderr, "0 counts for:", left
            
#                try:
            interp[i]=lprobs[right] #(1-leftcounts[l][0])*lprobs[right]
            Di[i]= get_discount_param(leftcounts[l]) #[0]/leftcounts[l][1]#(1-leftcounts[l][0])*c
#                except KeyError:
#                    print >>sys.stderr, "************** Could not find ngram '%s' in lower-order while processing ngram '%s'"% (right, ng)
#                    interp[i]=0.
#                print >>sys.stderr, "Interpolation weight:",dot(leftcounts[left][:3], D[:3])/leftcounts[left][3] , "Previous prob=", lprobs[right]
#        Ns=zeros(HIGHEST_POSITIVE_BIN)
#        for l in leftcounts:
#            Ns+=leftcounts[l]
#        print >>sys.stderr, "All Ns=",Ns
######verification
#        weights={}
#        for l in leftcounts:
#            weights[l]=leftcounts[l][0]/leftcounts[l][1]
#        return Di, leftall, interp ,weights
##### End verif
        Di=(1-xi_fun(Di))
        interp*=Di
        Di*=list(odicts[order].values())
#        savetxt("inter.%s"%(time.time()), interp)
#        savetxt("lc.%s"%(time.time()), leftall)        
#        savetxt("Di.%s"%(time.time()),  Di)
        return Di, leftall, interp #,weights
 
##########################   
    
def compute_kn_probs( *args):
    Di, acounts, prevcounts, second=args
#    questionable=where((acounts-Di)/prevcounts+second >1)
#    if len(questionable)>0:
#        for i in questionable:
#            print >>sys.stderr, "Problematic entry (p=%s>1): First=%s, second= %s"%(((acounts-Di)/prevcounts+second )[i], ((acounts-Di)/prevcounts)[i],((acounts-Di)/prevcounts+second )[i]- ((acounts-Di)/prevcounts)[i])
    #    if any(prevcounts<=0):
    #        print >> sys.stderr,"Some previous counts are zero"
    return where((acounts-Di)>0, (acounts)/prevcounts+(-Di/prevcounts+second), 0)
    print(prob, file=sys.stderr)
    print("MAX=", max(prob), file=sys.stderr)
    

    try:            
        cbos=odicts[1][BOS]
            
    except KeyError:
        cbos=0
    odicts[1][BOS]=0
    norm=float(sum(odicts[1].values()) )
    odicts[1][BOS]=cbos
    return norm
    
    if order==1:
        try:            
            cbos=odicts[1][BOS]
            
        except KeyError:
            cbos=0
            
        odicts[1][BOS]=0
        probs={}
        norm=float(sum(odicts[1].values()) )
        for ng in odicts[1]:
            c=odicts[1][ng]
            probs[ng]=c/norm
#            print log10(c/norm) if c else -99, ng
            
        odicts[1][BOS]=cbos
        return probs
    
   
    ngrams=sorted((ng, i) for i, ng in enumerate(od[order]))
    N=len(ngrams)
    startindx=part*N/nparts
    endindx=max(0, min((part+1)*N/nparts, N-1))
    print("Process", rank, "Computing backoffs for part: %d (out of %d)"%(part, nparts), file=sys.stderr)
    if endindx<N-1:
#        lastng, _=ngrams[endindx]
        if order>1:
            lastleftctxt, _=ngrams[endindx][0].rsplit(None,1)
        else:
            lastleftctxt=""
        print("Process", rank, "Last left contxt: '%s'"%( lastleftctxt), file=sys.stderr)
        for ei in range(endindx+1, min((part+2)*N/nparts, N)+1):
            
            if order>1:
                l,_=ngrams[ei][0].rsplit(None,1)
            else:
                l=""
            if l != lastleftctxt:
                print("Process", rank, "Breaking at :'%s'"%( l), file=sys.stderr)
                break
    else:
        ei=N
        
    if part>0:
#        firstng, _=ngrams[startindx]
        if order>1:
            firstleftctxt, _=ngrams[startindx][0].rsplit(None,1)
        else:
            firstleftctxt=""
        print("Process", rank, "First left contxt: '%s'"%( firstleftctxt), file=sys.stderr)
        for si in range(startindx, ei):            
            if order>1:
                l,_=ngrams[si][0].rsplit(None,1)
            else:
                l=""
            if l != firstleftctxt:
                print("Process", rank, "Breaking at :'%s'"%( l), file=sys.stderr)                
                break
    else:
        si=0
                
    bow={}
    lowerex=set() #exclude[order-1] if order-1 in exclude else set()
    curex=exclude[order] if order in exclude else set()
    
    print("Process", rank, "all ngrams: %d, my part: %d"%(len(ngrams), ei-si), file=sys.stderr)
    print("Process", rank, "starting at: %d (%s); previous %s"%(si, ngrams[si], ngrams[max(0, si-1)]), file=sys.stderr)
    print("Process", rank, "ending at: %d (%s), next %s "%(ei, ngrams[ei-1], ngrams[min(N-1, ei)]), file=sys.stderr)
    print("Number of lower order ngrams to exclude:",len(lowerex), file=sys.stderr)
    print("Number of current order ngrams to exclude:",len(curex), file=sys.stderr)
    getlower=lowerorderprobs.get
    
    for ng, j in ngrams[si:ei]:        
#    for j, ng in enumerate(od[order]):
        if ng in curex:
            continue
        
        if order>1:
            l,_=ng.rsplit(None,1)
            _,right=ng.split(None, 1) #ng.split(None,1)
        else:
            l=""
            right = ng
#                r=ng
#        if order>1: #try:
#           
#        else: #except ValueError:
#            right = ng
        try:   
#            bow[l]-=(curprobs[j] ,getlower(right) if (right) not in lowerex else 0)
            if ng not in curex:
                bow[l][0]  -=curprobs[j]
#            if (right,r) not in lowerex:
            if (right) not in lowerex:
                bow[l][1]  -=getlower((right),0)
        except KeyError:
#            bow[l]  = array([1-(curprobs[j] ),1-(getlower(right) if (right) not in lowerex else 0)])
            bow[l]  = [1-(curprobs[j] if ng not in curex else 0),1-(getlower((right),0) if (right) not in lowerex else 0)]
    return bow
    
def compute_bow(od,order,curprobs,lowerorderprobs, ng2rem):
    
    bow={}
#    lowerex=set() #exclude[order-1] if order-1 in exclude else set()
    is_rem=(lambda x: ng2rem[order][x]) if order in ng2rem else (lambda x: False) #set()
#    print >> sys.stderr,"Number of lower order ngrams to exclude:",len(lowerex)
    print("Number of ngrams to exclude for order %i:"%order,sum( ng2rem[order]!=0) if order in ng2rem else 0, file=sys.stderr)
    getlower=lowerorderprobs.get
    if order >1:
        get_ctxt=lambda ng: (ng.rsplit(None, 1)[0] , ng.split(None, 1)[1])#str.rsplit(ng, None, 1)
    else:
        get_ctxt=lambda ng: ("", ng)
            
    for j, ng in enumerate(od[order]):
#        if ng=="obtenez . </s>":
#            print >>sys.stderr, "Indicator('obtenez . </s>')=", ng2rem[order][j]
#            sys.exit(0)
      
#        if order -1 in include:
            
#        if ng.startswith("dessus ,"):
#            print >>sys.stderr,"NG=",ng,"is rem=",is_rem(j),
        
##        if is_rem(j)>=1:  
        if is_rem(j)>2 or (order==options.order and is_rem(j)>=1): #ng in curex:
#            print >>sys.stderr,"Context:",l,"exists in the lower orde:",l in lowerorderprobs
            continue   
         #####   
#        try:
#            l, _=get_lctxt(ng) 
#            _,right=get_rctxt(ng) 
#        
#        except ValueError:
#            get_lctxt=lambda ng:("", None)
#            l=""
#            get_rctxt=lambda ng:(None, ng)
#            right=ng
        l, right=get_ctxt(ng)
#        if order>1:
#            l,_=ng.rsplit(None,1)
#            _,right=ng.split(None, 1) #ng.split(None,1)
#        else:
#            l=""
#            right = ng 
#        
           
#        if ng.startswith("dessus ,"):
#            print >>sys.stderr,"Prob=",curprobs[j],"Prob(%s)=%g"%(right,getlower((right),0))
        
        if 1<=is_rem(j)<=2: #ng in curex:
            if l not in bow:
#                bow[l]=ones(2, dtype=float) #[1]*2 #, 1-getlower((right),0)]
                bow[l]=[1.]*2 #, 1-getlower((right),0)]
            
##            bow[l][1]  -=getlower((right),0)
            continue
        ###
        # If the lower order has no indicators, , we will give a 1 backoff to all prefixes whose all suffixes were deleted
        ###
#        if order -1 not in include:
#            if not curinc(j): #ng in curex:
#                if l in lowerorderprobs and l not in bow:
#                    bow[l]=[1]*2 
#                continue
     #####   
        
#                r=ng
#        if order>1: #try:
#           
#        else: #except ValueError:
#            right = ng
        try:   
#            bow[l]-=(curprobs[j] ,getlower(right) if (right) not in lowerex else 0)
            item=bow[l]
            item[0]  -=curprobs[j]
#            bow[l][0]  -=curprobs[j]
#            if (right,r) not in lowerex:
            item[1]  -=getlower((right),0)
#            bow[l][1]  -=getlower((right),0)
        except KeyError:
#            bow[l]  = fromiter((1-(curprobs[j] ),1-(getlower((right),0) )), dtype=float)
            bow[l]  = [1-(curprobs[j] ),1-(getlower((right),0) )]
    return bow
             
    print(file=sys.stderr);"BINS=", bins
    hist, e=histogram(vec, bins=int(bins) )
    if e[0]<=0:
#        print >>sys.stderr,"Counts before dropping the first:",hist,"edges:",e
        e=e[1:]
        hist=hist[1:]
    return hist, e


    def ks_test( xmin):
        if sum(d>xmin) <= 16:
#            print >> sys.stderr, "Penalizing XMIN=", xmin,  "since sum(d>xmin)=",sum(d>xmin) ,"<= 16"
            return 10, 
        try:
            fitxmin= plfit.plfit(d,xmin=xmin,usefortran=True, discrete=False)#, verbose=False, quiet=True, silent=True) #._ks
            if fitxmin._alpha==0 and  fitxmin._likelihood==0:
                return 10, 
            return fitxmin._ks, 
        except (ValueError, FloatingPointError):
            return 10, 
    def init():
        ##TODO try wider range e.g 0.005-- 0.2 or so
        ## Better make the lower bound dependent on the length of the vector. e.g 1./max(0.1*len_vec,HIGHEST_POSITIVE_BIN), requiring at max 10 values per bin and at least HIGHEST_POSITIVE_BIN bins
#        choice=random.randint(100)
        if random.randint(100)%10==0:
           v=d[random.randint(len(d))]
        else:
            if random.randint(100) %4==0:
                v=random.uniform(max(d)/2.0, max(d))
            else:
                v=random.uniform(min(d), max(d)/2.0) 
             #random.randint(min(d), max(d))      
#        print >>sys.stderr,"Generated:",v
        return v
    def cross(h1, h2):
#            h1, h2
        b1,b2=h1[0],h2[0] #),max(h1[0],h2[0])
        b1+=(b2-b1)/3
        b2-=(b2-b1)/3
        return [b1],[b2] #[0.5*(sum(h1+h2))]
    def mutate(h):
        delta=random.uniform(0.5,2.5)
#        print >>sys.stderr, "H before:", h,
        if random.randint(100)%2==0:
            nh=h[0]*delta
        else:
            nh= init()
#        print  >>sys.stderr,"New H:", nh
        return [nh]
    
    xmin=ga_min(ks_test, init, cross, mutate,maxiter=10,mutatep=0.4, full_output=1 )
    xmin=float(xmin[0])
    print("Best XMIN=", xmin, "n[>XMIN]=", sum(d>xmin) , "D statistic=", ks_test(xmin), file=sys.stderr)
    return (xmin)
        

def open_infile(f, mode="r"):
    return open(f,mode) if f !="-" else sys.stdin
  
def open_outfile(f, mode="w"):
    return open(f,mode) if f !="-" else sys.stdout
#    
#    params={}
#    for line in file(f):
#        if line.startswith(COMMENT_LINE):
#            continue
##        print >>sys.stderr, "Line=", line
#        try:
#            k, v=line.split(None, 1)
#            v=v.split()
#            if len(v)==1:
#                v,=v
##            print >>sys.stderr, "K=", k, "V=", v
#        except ValueError:
#            continue
#        params[k]=v
##    
#    print("Parameters:", file=sys.stderr)
#    for k, v in params.items():
#        print("\t%s\t%s"%(k, v), file=sys.stderr)
#    return params
#
##    acounts=od[order]
#    if (max(acounts)-min(acounts))/h >MAX_BINS:
#        print("Huge number of bins", file=sys.stderr)
#        return False
#    counts,e=mk_hist(acounts, bins=ceil((max(acounts)-min(acounts))/h))
#    
#    for i in range(HIGHEST_POSITIVE_BIN):
#        if 0 >= sum(acounts[(e[i]<=acounts) & (acounts<e[i+1])]) :
#            return False
#    return True
#                
   
def start4testing(od):
    
    for i in od:
#        print >>sys.stderr,"Number of values for order %d=%d"%(i, len(od[i]))
        od[i]=fromiter((v for v in list(od[i].values()) if v>0), dtype=float)
#        print >>sys.stderr,"Values after changing:", od[i]
    print("Reading values from input", file=sys.stderr)
    
    
    for line in sys.stdin:
        kv=line.split()
        
        kv=dict(list(zip(kv[::2],kv[1::2])))
#        print >>sys.stderr,"Line=", line
#        print >>sys.stderr,"KV=", kv
        for k,v in kv.items():
            if k.startswith("h"):
                order=int(k[1:])
                print("Number of values in the vector for order %d: %d" %(order,len(od[order])), file=sys.stderr)
                h=float(v)
                print("Testing h=", h, "MAX=", max(od[order]), "MIN=", min(od[order]), "BINS=", (max(od[order])-min(od[order]))/h, file=sys.stderr)
                if isgood_bandwidth(h,od[order]):              
                    print("%s\t%s"%(k, v), file=sys.stdout)
                else:
                    print("h%d=%g is not good"%(order, h), file=sys.stderr)
                    
#                sys.stdout.flush()
                        
            if k=="0" and v=="0":
                print("Testing done, exiting", file=sys.stderr)
                sys.exit(0)
       

class ddict(dict):
    def __init__(self, default, *args, **kargs):
        self._default=default
        super(ddict, self).__init__(*args, **kargs)
    def __getitem__(self, k):
        return self.get(k, self._default)
    @property
    def default(self):
        return self._default
        
    @default.setter
    def default(self, v):
        self._default=v
    def __str__(self):
        return "Default: "+str( self._default)+", " +super(ddict, self).__str__()
                  
def analyse_arg(arg, dtype=None, default=None, allowed_vals=None):
    if dtype is not None and default is not None:
        default=dtype(default)
#    print >>sys.stderr, "Allowed=", allowed_vals
    if allowed_vals is not None and default not in allowed_vals:   
        default=None
        
    if not isinstance(arg, str):
        if allowed_vals is not None and arg not in allowed_vals:
            return ddict(default)
        return ddict(arg)
        
#        return ddict(lambda: default)
    
#    print >>sys.stderr, "default=", default
    d={}
    
        
    for kv in (mw.split(ARG_ARG_SEP) for mw in arg.split(ARG_SEP)):
        try:
            k, v=kv
            try:
                if dtype is not None:
                    v=dtype(v) 
                if allowed_vals is not None and v not in allowed_vals:
                    continue
                d[int(k)]=v
            except ValueError:
                if dtype is not None:
                    v=[dtype(x) for x in v.split(SCORE_ARG_SEP)]
                if allowed_vals is not None and not set(v) <= allowed_vals:
                    continue
                d[int(k)]=v
        except ValueError:
            kv, =kv
            try:
                if dtype is not None:
                    kv=dtype(kv) 
                if allowed_vals is not None and kv not in allowed_vals:
                    continue
                default=dtype(kv) if dtype is not None else kv
            except ValueError:
                if dtype is not None:
                    kv=[dtype(x) for x in kv.split(SCORE_ARG_SEP)]
                if allowed_vals is not None and not set(kv) <= allowed_vals:
                    continue                
                default=kv
                    
                
                
            
        
    return ddict( default, d)

       
def analyse_bool_arg(arg, default=False, value=True):
#    if dtype is not None and default is not None:
#        default=dtype(default)
#    print >>sys.stderr, "Allowed=", allowed_vals
#    if allowed_vals is not None and default not in allowed_vals:   
#        default=None
        
    if not isinstance(arg, str):
        if not isinstance(arg, bool):
            return ddict(default, {arg: value})
        return ddict(arg)
        
#        return ddict(lambda: default)
    
#    print >>sys.stderr, "default=", default
    d={}
    
        
    for k in arg.split(ARG_SEP):
        
        if k=="T" or k=="True":
           default=True
           continue
        elif k=="F" or k=="False":
            default=False
            continue 
        
        try:
            
            d[int(k)]=value
        except ValueError:
            try:
                a, b=k.split(RANGE_SEP)
                a, b=int(a), int(b)
                d.update((i, value) for i in range(min(a, b), max(a, b)+1))
            except ValueError:
                pass
            
        
    return ddict( default, d)
    
    
smethods=set([
              "none",
              "kn"
          
          ])

UNKNOWN_METHOD="unknown"        
select_scott=lambda vec:  bw_scott(vec)
select_silverman=lambda vec:  bw_silverman(vec)
selection_methods={
                       "scott": select_scott , 
                       "silverman":select_silverman, 
                       
                       UNKNOWN_METHOD:select_scott                       
                       }
                       
                       



MIX_METHODS=set(["mix-gamma-mle", "mix-gamma-mom", "mix-lindley-mle", "mix-lindley-mom","mix-lomax-mle", "mix-lomax-mom", "mix-lomax-chi2", "mix-lindley-chi2"])
UNSEEN_ESTIMATORS=["efron-thisted","bhat-sproat", "chao", "gandolfi-sastri", "mix-exp", "boneh-boneh-caron", "good-turing", "bcorrected-chao", "chao-bunge", "ace","extended-chao"] +list(MIX_METHODS)

description= "Train a language model"
usage = "Usage: %prog [{incounts|-}] [{outfile|-}]"
parser = OptionParser(usage)
parser.set_description(description)

parser.add_option("-i", "--interpolate", action="store_true",
                      dest="interpolate",
                      help="Interpolate probabilities with lower order",
                      default=False) ## Just for testing, in the Final version,,, this should be False

parser.add_option("-f", "--force-interpolate", action="store_true",
                      dest="finterpolate",
                      help="Interpolate probabilities with lower order (in the presence of no discounting interpolation is disabled since it could lead to probabilities greater than 1, use this option to force interpolation in such cases)",
                      default=False)

parser.add_option("-r", "--use-raw-counts", action="store_true",
                      dest="use_raw",
                      help="Don't adjust the counts",
                      default=False)
                                            
parser.add_option("-o", "--order", action="store",type="int",
                      dest="order",
                      help="The n-gram order",
                      default=4)
#     
parser.add_option("-k", "--use-parametric-smoothing", action="store_true",
                      dest="kern_smooth",
                      help="Use parametric smoothing",
                      default=False)

parser.add_option("-s", "--smoothing-method", action="store",type="string",
                      dest="smoothing",
                      help="Smoothing method to be used. One of %s"%(smethods),
                      default="kn")
  
parser.add_option("-a", "--force-adjust", action="store_true",
                      dest="fadjust",
                      help="Force adjusting the counts for methods which do not use adjusted counts",
                      default=False)
parser.add_option("-m", "--method", action="store",type="string",
                          dest="method",
                          help="Scoring method. Can be one  of  '%s' or the index of the method name. Default 'cooc'"%(sorted(methods.keys())),
                          default="cooc")
parser.add_option("-n", "--n-scores", action="store",type="int",
                      dest="n_scores",
                      help="The number of scores per ngram (at least 1. The first is supposed to be the cooccurrence)",
                      default=1)                      
                      
parser.add_option("-d", "--dump-associations", action="store_true",
                      dest="save_assocs",
                      help="Save associations of ngrams in files per order",
                      default=False)                    
     
parser.add_option("-F", "--mix-func", action="store",type="string",
                      dest="mix_func",
                      help="The function to use for mixing probability distributions",
                      default=None)   
   
parser.add_option("-W", "--mix-weights", action="store",type="string",
                      dest="mix_weights",
                      help="The weights to use for mixing probability distributions",
                      default=None)   
                
parser.add_option("-w", "--worst-assocs-nbr", action="store",type="string",
                      dest="nworst",
                      help="The number of ngrams to penalize based on their associations.",
                      default=None)   
                    
parser.add_option("-L", "--lowest-penalty", action="store",type="string",
                      dest="lowest_pen",
                      help="The smallest penalty value.",
                      default=None)     
parser.add_option("-H", "--highest-penalty", action="store",type="string",
                      dest="highest_pen",
                      help="The highest penalty value.",
                      default=None)   
                      
parser.add_option("-N", "--n-tune-counts", action="store",type="string",
                      dest="n_tune_counts",
                      help="The N-first counts will be used for MLE parameter tuning.",
                      default=None)   
                      
parser.add_option("-M", "--min-frequency", action="store",type="string",
                      dest="min_freqs",
                      help="The minimum frequency for n-grams in a comma-separated list ng1:minf1,ng2:minf2,...",
                      default=None)
             
parser.add_option("", "--no-discount-for-order", action="store",type="string",
                      dest="nodiscount",
                      help="Don't discount this orders (comma-separated list)",
                      default="")                          
#                      
                
                
parser.add_option("-t", "--start-as-test-server", action="store_true",
                      dest="astester",
                      help="Start the program as server waiting for incoming queries to test different bandwidths",
                      default=False)
                      

  
parser.add_option("-c", "--compute-associations-and-exit", action="store_true",
                      dest="compute_assocs_only",
                      help="Compute association scores and exit",
                      default=False)


parser.add_option("-x", "--xi-func", action="store",type="string",
                      dest="xi_func",
                      help="Type of the function to be used as discount function (xi_hyp: hyperbolic; xi_log: logarithmic; xi_exp: exponential)",
                      default="xi_hyp")                      

parser.add_option("", "--stupid-backoff",  action="store_true",
                      dest="sb",
                      help="Use stupid backoff instead of Katz BO",
                      default=False)

parser.add_option("", "--sbw", action="store",type="string",
                      dest="sbw",
                      help="Stupid backoff default weights in the form <order>:<weight>[,<order>:<weight>[,<order>:<weight>...]]",
                      default=None)
 
parser.add_option("-X", "--mix-dists",  action="store_true",
                      dest="mix_dists",
                      help="Mix distributions from different associations",
                      default=False)     

parser.add_option("-u", "--estimated-unseen",  action="store",type="string",
                      dest="estimate_unseen",
                      help="Method to be used to estimate the number of unseen events. Must be one of %s"%(UNSEEN_ESTIMATORS),
                      default=None)
                                           
  
parser.add_option("-C", "--mle-ctxt",  action="store",type="string",
                      dest="mle_ctxt",
                      help="Add the different contexts to the maximum-likelihood expression.",
                      default=None)                    
                      

parser.add_option("", "--mle-ctxts",  action="store_true",
                      dest="mle_ctxts",
                      help="Add the different contexts to the maximum-likelihood for all orders",
                      default=False)     
 
parser.add_option("-R", "--renorm-scores-to-coocs",  action="store_true",
                      dest="renorm2coocs",
                      help="Renormalize the association scores so that they sum up to the sum of cooccurrences (for all orders)",
                      default=False)          
 
parser.add_option("", "--renorm-scores-to-cooc",  action="store",type="string",
                      dest="renorm2cooc",
                      help="Add the different contexts to the maximum-likelihood expression.",
                      default=None)                    
  
parser.add_option("-D", "--use-count-distribution-as-xi-args",  action="store_true",
                      dest="xi_arg_is_count_dists",
                      help="Use an approximate count distribution function as xi argument for all orders.",
                      default=False)
                      
parser.add_option("", "--use-count-distribution-as-xi-arg",  action="store",type="string",
                      dest="xi_arg_is_count_dist",
                      help="Use an approximate count distribution function as xi argument for a specific order.",
                      default=None)                    
                      
parser.add_option("-A", "--association-is-xi-params",  action="store_true",
                      dest="assoc_xi_params",
                      help="Use the computed association as XI prameter. By default the XI is prameterized with the cooccurrence (for all orders)",
                      default=False)          
 
parser.add_option("-U", "--use-count-unseen-mass",  action="store_true",
                      dest="reuse_unseen",
                      help="The gained mass is estimated based on the counts, then the association scores are renormalized to fit this sum",
                      default=False)     
parser.add_option("", "--association-is-xi-param",  action="store",type="string",
                      dest="assoc_xi_param",
                      help="Use the computed association as XI prameter for a given order",
                      default=None)                    
   
parser.add_option("-S", "--scales", action="store",type="string",
                      dest="assoc_scales",
                      help="The minimum frequency for n-grams in a comma-separated list ng1:minf1,ng2:minf2,...",
                      default=None)
 
parser.add_option("-1", "--smoothing_unigram_dist",  action="store",type="string",
                      dest="smoothing_uni_dist",
                      help="Determine how to redistribute the gained mass on unigrams (uniform (default), unigram, cunigram)",
                      default="uniform")            
                      
parser.add_option("-p", "--penalty-file",  action="store",type="string",
                      dest="penalty",
                      help="Name of the file containing n-gram penalties (default None)",
                      default=None)            
                      
parser.add_option("-O", "--original-witten-bell-discounts",  action="store_true",
                      dest="original_wbs",
                      help="Use the original Witten-Bell discounting (for all orders)",
                      default=None)           
                      
parser.add_option("", "--original-witten-bell-discount",  action="store",type="string",
                      dest="original_wb",
                      help="Use the original Witten-Bell discounting (for selected orders)",
                      default=None)       
global_start=time.time()
(options, args) = parser.parse_args()

gparams={}

print("Training started @",time.ctime(), file=sys.stderr)
#try:
#    HIGHEST_POSITIVE_BIN=options.nhighestbins
#except ValueError:
HIGHEST_POSITIVE_BIN=4
    
MIN_BINS=HIGHEST_POSITIVE_BIN+1

random.seed()

#ord=options.order
if not options.use_raw:
    options.adjust=(options.smoothing in ADJUST_METHODS )
else:
    options.adjust=options.fadjust
if options.interpolate and options.smoothing in UNINTERPOLATED_METHODS:
    options.interpolate=False
if options.finterpolate:
    options.interpolate=True
    
try:
    s=args[0]
    print("Reading counts from file:",args[0], file=sys.stderr)
except IndexError:
    print("Reading counts from STDIN", file=sys.stderr)
try:
    s=args[1]
    print("ARPA format LM will be written to file:",args[1], file=sys.stderr)
except IndexError:
    print("Writing counts to STDOUT", file=sys.stderr)


supported=list(methods.keys())+[str(i) for i in range(len(list(methods.keys()))) ]
gparams["method"]=analyse_arg(options.method, default="cooc", allowed_vals= supported)


if gparams["method"].default not in supported:
    gparams["method"].default="cooc"
try:
    gparams["method"].default=sorted(methods.keys())[int(gparams["method"].default)]
except ValueError:
    pass
except IndexError:
    print("Index out of range: %d"%(gparams["method"].default), file=sys.stderr)
    gparams["method"].default="cooc"
    
for m in gparams["method"]:
    try:
        gparams["method"][m]=sorted(methods.keys())[int(gparams["method"][m])]
    except ValueError:
        pass
    except IndexError:
        print("Index out of range: %d"%(gparams["method"][m]), file=sys.stderr)
        gparams["method"][m]="cooc"
        
DEFAULT_BOW=dict(zip(range(1,options.order),arange(DEFAULT_BOW_START, DEFAULT_BOW_START+(options.order-2)*DEFAULT_BOW_STEP, DEFAULT_BOW_STEP)))

if options.sbw is not None:
    options.sbw=analyse_arg(options.sbw, dtype=float)
    for o in DEFAULT_BOW:
        if options.sbw[o] is not None:
            DEFAULT_BOW[o]= options.sbw[o]

#    
#if options.sbw is not None:
#    try:
##        mwstr=(mw.split(":") for mw in options.sbw.split(","))
#        for mstr, wstr in (mw.split(":") for mw in options.sbw.split(",")):
#            try:
#                DEFAULT_BOW[int(mstr)]=float(wstr)
#            except ValueError:
#                pass
#    except ValueError:
#        pass
    print("Stupid backoff weights:", DEFAULT_BOW, file=sys.stderr)
#sys.exit(0)
print("Association scoring method: '%s'"%(gparams["method"]), file=sys.stderr)
#                                                       options.method)
#sys.exit(0)
print("Order:",options.order, file=sys.stderr)
print("Interpolation:","yes" if options.interpolate else "no", file=sys.stderr)
print("Adjusting the counts:","yes" if options.adjust else "no", file=sys.stderr)
print("Using kernel smoothing:","yes" if options.kern_smooth else "no", file=sys.stderr)
print("Smoothing method:",options.smoothing, file=sys.stderr)


if options.nodiscount != "":
    try:
        options.nodiscount=[int(o) for o in options.nodiscount.split(ARG_SEP)]
    except ValueError:
        options.nodiscount=list(range(1, options.order+1))
    print("The following orders will not be discounted:",options.nodiscount, file=sys.stderr)
else:
    options.nodiscount=[]
    
    
fixed_b=False
try:
    b=1. #float(options.bandwidth)
    options.bandwidth= lambda x: b
    fixed_b=True
except ValueError:
    if options.bandwidth not in selection_methods:
        options.bandwidth=UNKNOWN_METHOD
    options.bandwidth=selection_methods[options.bandwidth]

if options.min_freqs is not None:  
    options.min_freqs= analyse_arg  (options.min_freqs, dtype=int, default=1)
#    options.min_freqs=dict((int(x) for x in ngminfreq.split(ARG_ARG_SEP, 1) ) for ngminfreq in options.min_freqs.split(ARG_SEP))
    print("Minimum frequencies for ngram orders:", options.min_freqs, file=sys.stderr)


if options.assoc_scales is not None:  
    options.assoc_scales= analyse_arg  (options.assoc_scales, dtype=float, default=1)
#    options.min_freqs=dict((int(x) for x in ngminfreq.split(ARG_ARG_SEP, 1) ) for ngminfreq in options.min_freqs.split(ARG_SEP))
    print("Association scales:", options.assoc_scales, file=sys.stderr)
    
if options.nworst is not None:
    options.nworst=analyse_arg  (options.nworst, dtype=int, default=200)
    print("Number of worst points to be penalized:", options.nworst, file=sys.stderr)
       
if options.lowest_pen is not None:
    options.lowest_pen=analyse_arg  (options.lowest_pen, dtype=float, default=.6)
    print("Lowest penalization values:", options.lowest_pen, file=sys.stderr)
    
if options.highest_pen  is not None:
    options.highest_pen=analyse_arg  (options.highest_pen, dtype=float, default=.6)
    print("Highest penalization values:", options.highest_pen, file=sys.stderr)
    
 
if options.n_tune_counts is not None:
    options.n_tune_counts =analyse_arg  (options.n_tune_counts , dtype=int, default=None)
    print("Number of counts which will be used in mLE tuning:", options.n_tune_counts, file=sys.stderr)
    
if options.estimate_unseen is not None:  
    options.estimate_unseen= analyse_arg  (options.estimate_unseen, dtype=str, default=None)
#    options.min_freqs=dict((int(x) for x in ngminfreq.split(ARG_ARG_SEP, 1) ) for ngminfreq in options.min_freqs.split(ARG_SEP))
    print("Estimated unseen data points for different orders:", options.estimate_unseen, file=sys.stderr)

if options.mle_ctxt is not None or options.mle_ctxts:
    if options.mle_ctxts:
        options.mle_ctxt= ddict(True, {})
    else:
        options.mle_ctxt= analyse_bool_arg  (options.mle_ctxt,  default=True)
    
#    options.min_freqs=dict((int(x) for x in ngminfreq.split(ARG_ARG_SEP, 1) ) for ngminfreq in options.min_freqs.split(ARG_SEP))
    print("Using contexts in maximum likelihood expression for different orders:", options.mle_ctxt, file=sys.stderr)

if options.original_wb is not None or  options.original_wbs:
    
    if options.original_wbs:
        options.original_wb= ddict(True, {})
    else:
        options.original_wb= analyse_bool_arg  (options.original_wb,  default=True)
    
#    options.min_freqs=dict((int(x) for x in ngminfreq.split(ARG_ARG_SEP, 1) ) for ngminfreq in options.min_freqs.split(ARG_SEP))
    print("Using the original Witten-Bell discounting:", options.original_wb, file=sys.stderr)
    
if options.renorm2cooc is not None or options.renorm2coocs:
    if options.renorm2coocs:
        options.renorm2cooc= ddict(True, {})
    else:
        options.renorm2cooc= analyse_bool_arg  (options.renorm2cooc,  default=True)
    
#    options.min_freqs=dict((int(x) for x in ngminfreq.split(ARG_ARG_SEP, 1) ) for ngminfreq in options.min_freqs.split(ARG_SEP))
    print("Renormalization of associations so that they sum up to cooccurrence sum for different orders:", options.renorm2cooc, file=sys.stderr)
   
if options.xi_arg_is_count_dist is not None or options.xi_arg_is_count_dists:
    if options.xi_arg_is_count_dists:
        options.xi_arg_is_count_dist= ddict(True, {})
    else:
        options.xi_arg_is_count_dist= analyse_bool_arg  (options.xi_arg_is_count_dist,  default=False)
    
#    options.min_freqs=dict((int(x) for x in ngminfreq.split(ARG_ARG_SEP, 1) ) for ngminfreq in options.min_freqs.split(ARG_SEP))
    print("Using an approximate power law distribution as xi argument:", options.xi_arg_is_count_dist, file=sys.stderr)
      
   
if options.assoc_xi_param is not None or options.assoc_xi_params:
    if options.assoc_xi_params:
        options.assoc_xi_param= ddict(True, {})
    else:
        options.assoc_xi_param= analyse_bool_arg  (options.assoc_xi_param,  default=True)
    
#    options.min_freqs=dict((int(x) for x in ngminfreq.split(ARG_ARG_SEP, 1) ) for ngminfreq in options.min_freqs.split(ARG_SEP))
    print("Passing the computed associations as parameters to the XI function for different orders:", options.assoc_xi_param, file=sys.stderr)
 
if options.mix_func is not None:  
    options.mix_func= analyse_arg  (options.mix_func,  dtype=str,default=None)
#    options.min_freqs=dict((int(x) for x in ngminfreq.split(ARG_ARG_SEP, 1) ) for ngminfreq in options.min_freqs.split(ARG_SEP))
    print("Mixing function:", options.mix_func, file=sys.stderr)

if options.penalty is not None:  
    options.penalty= analyse_arg  (options.penalty,  dtype=str, default=None)
#    options.min_freqs=dict((int(x) for x in ngminfreq.split(ARG_ARG_SEP, 1) ) for ngminfreq in options.min_freqs.split(ARG_SEP))
    print("Penalty files:", options.penalty, file=sys.stderr)
    
if options.smoothing_uni_dist not in ["unigram", "cunigram", "uniform"]:
    print("\nWARNING: unknown smoothing distribution for unigrams ('%s'), using 'uniform' instead"% (options.smoothing_uni_dist ), file=sys.stderr)
    options.smoothing_uni_dist ="uniform"
print("Smoothing distribution for unigrams: '%s'"%(options.smoothing_uni_dist ), file=sys.stderr)
if options.mix_weights is not None:  
    options.mix_weights= analyse_arg  (options.mix_weights,  dtype=float,default=None)
    if options.mix_weights.default is not None:
#        print >>sys.stderr, options.mix_weights.default
        assert all(x>=0 for x in options.mix_weights.default) and (len(options.mix_weights.default) == max(options.n_scores, 2))
        s=float(sum(options.mix_weights.default))
        options.mix_weights.default=[x/s for x in options.mix_weights.default]
    for o, w in options.mix_weights.items():
        assert all(x>=0 for x in w) and (len(w)== max(options.n_scores, 2))
        s=float(sum(w))
        options.mix_weights[o]=[x/s for x in w]
            
#    options.min_freqs=dict((int(x) for x in ngminfreq.split(ARG_ARG_SEP, 1) ) for ngminfreq in options.min_freqs.split(ARG_SEP))
    print("Mixing weights:", options.mix_weights, file=sys.stderr)

if options.reuse_unseen:
    print("The association scores will be renormalized to the count mass ", file=sys.stderr)
else:
    print("The relative discounts will be estimated from counts", file=sys.stderr)

quota=-1
#assigns=dict((x, ([], [])) for x in range(size))
#if size>1:
#    
#    print("Determining the topology...", file=sys.stderr)
#    quota=os.path.getsize(args[0])/size
#    tasks=list(range(1, options.order+1))
#    
#    modelat={}
#    
#    for i, a in enumerate(assigns):
#        if not tasks:
#            break 
#        tasksperworker=len(tasks)/(len(assigns)-i)
#        assigns[a][0].extend(tasks[:tasksperworker])
#        for t in tasks[:tasksperworker]:
#            modelat[t]=a
#        tasks=tasks[tasksperworker:]
#       
##    
##    while tasks:
##        for a in assigns:
##            if not tasks:
##                break
##            assigns[a][0].append(tasks.pop())
##        if tasks:
##            tasks=tasks[::-1]
#    freeworkers=[a for a, (m, _) in assigns.items() if not m]
#    for fw in freeworkers:
#        assigns.pop(fw)
#    while freeworkers:
#        for a in assigns:
#            if not freeworkers:
#                break
#            assigns[a][1].append(freeworkers.pop())
##    for a in assigns:
#    if rank in assigns:
#        print("Process", rank, "will build model for order(s)", assigns[rank][0], "and will be helped by processes", assigns[rank][1], file=sys.stderr)
#        
#else:
#    assigns[0]=(list(range(1, options.order+1)), [])
#    modelat=dict((x, 0) for x in range(1, options.order+1))
#sys.exit(0)

distribute_backoff_computation=False
print("Number of scores per n-gram:", options.n_scores, file=sys.stderr)
print("Loading n-gram counts...", file=sys.stderr)

try:
    infile=open_infile(args[0])
except IndexError:
    infile=sys.stdin

#distributed_load=False

if options.n_scores >1 :

    iter_counts=lambda d:(v[0] for v in list(d.values()))
    iter_nz_counts=lambda d:(v[0] for v in list(od[i].values()) if v[0]>0)
else:
    iter_counts=lambda d:(v for v in list(d.values()))
    iter_nz_counts=lambda d:(v for v in list(d.values()) if v>0)
    
#if not distributed_load:
full=None
loadprefs=False
if options.min_freqs is not None:
    full=[i for i in range(1,max(options.min_freqs)+1) if i in assigns[rank][0] ]  or None
    
print("Full-loading:",full, file=sys.stderr)
od, pref=load_counts(infile, options.adjust, options.order, full_load=full, min_assocs=options.min_freqs)
#else:
#    od=dist_load_counts(infile, assigns, rank, quota, options.adjust, options.order)
ng2rem={}
#ind_recv,ind_send=None,None
#pref={}
if options.min_freqs is not None:
    for o in sorted(assigns[rank][0], reverse=True):
        if o in options.min_freqs:
            
            print("Finding n-grams to exclude from order:", o, file=sys.stderr)
            
            ng2rem[o]=fromiter(( (v<options.min_freqs[o])  for v in list(od[o].values())),dtype=int8)
#        if o-1 in options.min_freqs:
#            
#            pref[o]=set(k.rsplit(None, 1)[0] for k in od[o] if ( (o in options.min_freqs) and (od[o][k] >=options.min_freqs[o]) ) or ( o not in options.min_freqs))
            

#
#for i in od:
##    if i<options.order:
##        continue
#    print >>sys.stderr,"Process",rank,"Number of %d-grams: %d"%(i, len(od[i]))
#    print >>sys.stderr,"Process",rank,"First ten values in %d-grams after adjusting:"%(i)
#    print >>sys.stderr,od[i].items()[:10]
    

#sys.exit(0)
#od=threaded_load(infile, options.order)
#for i in od:
#    print >>sys.stderr,"Number of %d-grams: %d"%(i, len(od[i]))
#print >>sys.stderr,"First ten values in unigrams:"
#print >>sys.stderr,od[1].items()[:10]
#if size>1 and distributed_load :
#    print >>sys.stderr,"Updating dictionaries from other workers..."
#    update_dicts(od, assigns, rank, full=options.adjust)
 
#
#for i in od:
#    if i<options.order:
#        continue
#    print >>sys.stderr,"Process",rank,"Number of %d-grams: %d"%(i, len(od[i]))
#    print >>sys.stderr,"Process",rank,"First ten values in %d-grams after adjusting:"%(i)
#    print >>sys.stderr,od[i].items()[:10]
    
if options.adjust and options.n_scores <=1:
    print("Adjusting counts...", file=sys.stderr)
    adjust_counts(od, options.order, keephigher=distribute_backoff_computation, interpolation=(options.interpolate and not options.sb))


if options.min_freqs is not None:
    for o in sorted(assigns[rank][0], reverse=True):
        if o< options.order:
            if o in options.min_freqs:
                print("Updating the n-grams to exclude from order:", o, file=sys.stderr)
                ng2rem[o]+=fromiter(((v<options.min_freqs[o])*(1  + (k not in pref[o+1]))  for k, v in od[o].items()),dtype=int8)
                
del pref


ind_recv,ind_send=None,None
#if prefixes is not None:
#    print >>sys.stderr, "Process",rank, "Orders with prefixes:",prefixes.keys()
if options.min_freqs is not None:
    for o in sorted(assigns[rank][0], reverse=True):
        if o in options.min_freqs:          
            if o != options.order:
                #send the indicators to where the ngrams will be output
                if modelat[o+1] !=rank:
                    print("Process",rank,"sending",len(ng2rem[o]),"indicators to process",modelat[o+1], file=sys.stderr)
                    ind_send=comm.Isend([ng2rem[o], MPI.CHAR], dest=modelat[o+1], tag=888)                   
        
                    
        if o>1:
            if o-1 in options.min_freqs:
                #receive the indicators from where they have been generated
                
                if modelat[o-1] !=rank:
                    print("Process",rank,"receiving",len(od[o-1]),"indicators from process",modelat[o-1], file=sys.stderr)
                    ng2rem[o-1]=empty(len(od[o-1]),dtype=int8)
                    ind_recv=comm.Irecv(ng2rem[o-1], source=modelat[o-1], tag=888)
          

print("Dictionaries:", [i for i in od], file=sys.stderr)
    
sys.stderr.flush()
#

Dd=dict((order,[0]*(HIGHEST_POSITIVE_BIN-1))     for order in range(1, options.order+1))
dfuns={}
 
less=lambda x, y: (x-y) < EPS
less_eq=lambda x, y: (x-y) <= EPS

gt=lambda x, y: (x-y) > EPS
gt_eq=lambda x, y: (x-y) >= EPS


def tune_xi_kn(count_counts,  N_XI_PARAMS):
    y=float(count_counts[0])/(count_counts[0]+2*count_counts[1])
    
    p=arange(1, N_XI_PARAMS+1, 1)
    return p -(p+1)*y*count_counts[1:N_XI_PARAMS+1]/count_counts[:N_XI_PARAMS]

def tune_comb_xi(dx, dy, xif, N_XI_PARAMS, w0, use_loo=True) :
    def ll_counts(p,  counts,  up,  w0,  inv_cnt):
#        if any(p<0) or any(p>1e3):
#            return  -1e100     
        try:   
            p=exp(p)
#            p[1]=0
#            p[2]=1.
#            print >>sys.stderr, "Evaluating for p=", p
#            pweights=weights[gt(datapoints, pivot)]
#            pdatapoints=datapoints[gt(datapoints, pivot)]
#            d_pdatapoints=pdatapoints-pivot
#            mask=gt(ddp, 0)
            
#            locxi=xif(inv_cnt, up, exp(p)) #*sum(disc_w*datapoints)/sum(wdp)
            locxi=xif(inv_cnt, up, p) #*sum(disc_w*datapoints)/sum(wdp)
#            print >>sys.stderr, "First 10 XI vals:", locxi[:10]
            ll1=sum((1+counts)*log(locxi))
            
            ll0=w0*log(sum((1-locxi)*counts))
            return (ll1+ll0) #+penal
        except FloatingPointError:
            return -1e100     
    negll_counts=lambda p, cnt, u, w0,  inv_cnt: -ll_counts(p, cnt, u, w0, inv_cnt)
    
    def ll_loo(p,  counts,  w0,  inv_cnt):
#        if any(p<0) or any(p>1e3):
#            return  -1e100     
        try:   
            p=exp(p)
#            p[0]=p_x
#            p[1]=pwb
#            p[2]=0.5  #1./(1+p[2])
#            print >>sys.stderr, "Evaluating for p=", p
#            pweights=weights[gt(datapoints, pivot)]
#            pdatapoints=datapoints[gt(datapoints, pivot)]
#            d_pdatapoints=pdatapoints-pivot
#            mask=gt(ddp, 0)
            
#            locxi=xif(inv_cnt, up, exp(p)) #*sum(disc_w*datapoints)/sum(wdp)
            locxi=xif(inv_cnt, p) #*sum(disc_w*datapoints)/sum(wdp)
#            print >>sys.stderr, "First 10 XI vals:", locxi[:10]
            ll1=sum((1+counts)*log(locxi))
            
            ll0=w0*log(sum((1-locxi)*counts))
            return (ll1+ll0) #+penal
        except FloatingPointError:
            return -1e100     
    negll_loo=lambda p, cnt, w0,  inv_cnt: -ll_loo(p, cnt, w0, inv_cnt)
    
    def ll_unseen(p,  counts,  w0,  inv_cnt):
#        if any(p<0) or any(p>1e100):
#            return  -1e100 
        try:     
            p=exp(p)
#            p[2]=1./(1+p[2])
#            print >>sys.stderr, "Evaluating for p=", p
#            pweights=weights[gt(datapoints, pivot)]
#            pdatapoints=datapoints[gt(datapoints, pivot)]
#            d_pdatapoints=pdatapoints-pivot
#            mask=gt(ddp, 0)
            
            locxi=xif(inv_cnt, (p)) #*sum(disc_w*datapoints)/sum(wdp)
            
            ll1=sum(counts*log(locxi))
            
            ll0=w0*log(sum((1-locxi)*counts))
            return (ll1+ll0) #+penal
        except FloatingPointError:
            return -1e100     
    negll_unseen=lambda p, cnt, w0,  inv_cnt: -ll_unseen(p, cnt, w0, inv_cnt)
    if use_loo:
        print("Using Leave-One-Out...", file=sys.stderr)
        ll=negll_loo
    else:
        
        print("The unseen is given:", w0, file=sys.stderr)
        ll=negll_unseen
    
    dx=array(dx, dtype=float)
#    dy=array(len(dx)/sum(dx)) #array(dy)
    inv_x=dx**-1*(dy)
#    print >>sys.stderr, "Size dx=", len(dx),  "Size dy=", len(dy)
#    
#    print >>sys.stderr, "First 10 dx=", dx[:10]
#    print >>sys.stderr, "First 10 dy=", dy[:10]
#    print >>sys.stderr, "First 10 inv x=", inv_x[:10]
    
#    curxi=zeros(N_XI_PARAMS)    
#    
##    xip, negllk, _, _, _, _=fmin_powell(ll,curxi,args=( dx, dy, w0, inv_x), maxiter=1000,maxfun=1000, full_output=True, disp=0)
#    xip, negllk, _, _, _=fmin(negll_counts,curxi,args=( dx, dy, w0, inv_x), maxiter=1000,maxfun=1000, full_output=True, disp=0)
#    p_x=exp(xip[0])
#    
#    print >>sys.stderr,"Best XI parameters for counts=", p_x, "Likelihood value=", -negllk #, "augmented entropy=", sum(weights*datapoints)*ent1(xip) #, "Sum LL+ENT=", sum(weights*datapoints)*ent(xip)+llk 
    
    curxi=zeros(N_XI_PARAMS)    
    
#    xip, negllk, _, _, _, _=fmin_powell(ll,curxi,args=( dx, w0, inv_x), maxiter=1000,maxfun=1000, full_output=True, disp=0)
    xip, negllk, _, _, _=fmin(ll,curxi,args=( dx, w0, inv_x), maxiter=1000,maxfun=1000, full_output=True, disp=0)
    
    xip=exp(xip)
#    xip[0]=p_x
#    xip[1]=pwb
#    xip[2]=0.5 #1./(1+p[2])
    print("Best XI parameters for counts=", xip, "Likelihood value=", -negllk, file=sys.stderr) #, "augmented entropy=", sum(weights*datapoints)*ent1(xip) #, "Sum LL+ENT=", sum(weights*datapoints)*ent(xip)+llk 
    
    locxi=xif(inv_x, xip)
    print("Gained mass=", 1- sum(dx*locxi)/sum(dx), file=sys.stderr)
    print("First 10 XI vals:", locxi[:10], file=sys.stderr)
    print("First 10 counts:", dx[:10], file=sys.stderr)
    print("First 10 uniq/counts:", dy[:10], file=sys.stderr)
    print("First 10 discounted counts:", dx[:10]*locxi[:10], file=sys.stderr)
#    sys.exit(0)
    return xip
            

def tune_comb_xi_sep(dx, dy, xif, N_XI_PARAMS, w0, use_loo=True) :
    def ll_counts(p,  counts,  up,  w0,  inv_cnt):
#        if any(p<0) or any(p>1e3):
#            return  -1e100     
        try:   
            p=exp(p)
#            p[1]=0
#            p[2]=1.
#            print >>sys.stderr, "Evaluating for p=", p
#            pweights=weights[gt(datapoints, pivot)]
#            pdatapoints=datapoints[gt(datapoints, pivot)]
#            d_pdatapoints=pdatapoints-pivot
#            mask=gt(ddp, 0)
            
#            locxi=xif(inv_cnt, up, exp(p)) #*sum(disc_w*datapoints)/sum(wdp)
            locxi=xif(inv_cnt, up, p) #*sum(disc_w*datapoints)/sum(wdp)
#            print >>sys.stderr, "First 10 XI vals:", locxi[:10]
            ll1=sum((1+counts)*log(locxi))
            
            ll0=w0*log(sum((1-locxi)*counts))
            return (ll1+ll0) #+penal
        except FloatingPointError:
            return -1e100     
    negll_counts=lambda p, cnt, u, w0,  inv_cnt: -ll_counts(p, cnt, u, w0, inv_cnt)
    
    def ll_loo(p,  counts,  w0,  xx, yy):
#        if any(p<0) or any(p>1e3):
#            return  -1e100     
        try:   
            p=exp(p)
#            p[0]=p_x
#            p[1]=pwb
#            p[2]=0.5  #1./(1+p[2])
#            print >>sys.stderr, "Evaluating for p=", p
#            pweights=weights[gt(datapoints, pivot)]
#            pdatapoints=datapoints[gt(datapoints, pivot)]
#            d_pdatapoints=pdatapoints-pivot
#            mask=gt(ddp, 0)
            
#            locxi=xif(inv_cnt, up, exp(p)) #*sum(disc_w*datapoints)/sum(wdp)
#            print >>sys.stderr, "yy=", yy, "xx=", xx
            locxi=xif(xx, yy, p) #*sum(disc_w*datapoints)/sum(wdp)
#            print >>sys.stderr, "First 10 XI vals:", locxi[:10]
            ll1=sum((1+counts)*log(locxi))
            
            ll0=w0*log(sum((1-locxi)*counts))
            return (ll1+ll0) #+penal
        except FloatingPointError:
            return -1e100     
    negll_loo=lambda p, cnt, w0,  x, y: -ll_loo(p, cnt, w0, x, y)
    
    def ll_unseen(p,  counts,  w0,  xx, yy):
#        if any(p<0) or any(p>1e100):
#            return  -1e100 
        try:     
            p=exp(p)
#            p[2]=1./(1+p[2])
#            print >>sys.stderr, "Evaluating for p=", p
#            pweights=weights[gt(datapoints, pivot)]
#            pdatapoints=datapoints[gt(datapoints, pivot)]
#            d_pdatapoints=pdatapoints-pivot
#            mask=gt(ddp, 0)
            
            locxi=xif(xx, yy, (p)) #*sum(disc_w*datapoints)/sum(wdp)
            
            ll1=sum(counts*log(locxi))
            
            ll0=w0*log(sum((1-locxi)*counts))
            return (ll1+ll0) #+penal
        except FloatingPointError:
            return -1e100     
    negll_unseen=lambda p, cnt, w0,  x, y: -ll_unseen(p, cnt, w0, x, y)
    if use_loo:
        print("Using Leave-One-Out...", file=sys.stderr)
        ll=negll_loo
    else:
        
        print("The unseen is given:", w0, file=sys.stderr)
        ll=negll_unseen
    
    dx=array(dx, dtype=float)
    dy=array(dy, dtype=float)
#    dy=array(len(dx)/sum(dx)) #array(dy)
    inv_x=dx**-1
#    print >>sys.stderr, "Size dx=", len(dx),  "Size dy=", len(dy)
#    
#    print >>sys.stderr, "First 10 dx=", dx[:10]
#    print >>sys.stderr, "First 10 dy=", dy[:10]
#    print >>sys.stderr, "First 10 inv x=", inv_x[:10]
    
#    curxi=zeros(N_XI_PARAMS)    
#    
##    xip, negllk, _, _, _, _=fmin_powell(ll,curxi,args=( dx, dy, w0, inv_x), maxiter=1000,maxfun=1000, full_output=True, disp=0)
#    xip, negllk, _, _, _=fmin(negll_counts,curxi,args=( dx, dy, w0, inv_x), maxiter=1000,maxfun=1000, full_output=True, disp=0)
#    p_x=exp(xip[0])
#    
#    print >>sys.stderr,"Best XI parameters for counts=", p_x, "Likelihood value=", -negllk #, "augmented entropy=", sum(weights*datapoints)*ent1(xip) #, "Sum LL+ENT=", sum(weights*datapoints)*ent(xip)+llk 
    
    curxi=zeros(N_XI_PARAMS)    
    
#    xip, negllk, _, _, _, _=fmin_powell(ll,curxi,args=( dx, w0, inv_x), maxiter=1000,maxfun=1000, full_output=True, disp=0)
    xip, negllk, _, _, _=fmin(ll,curxi,args=( dx, w0, inv_x, dy), maxiter=1000,maxfun=1000, full_output=True, disp=0)
    
    xip=exp(xip)
#    xip[0]=p_x
#    xip[1]=pwb
#    xip[2]=0.5 #1./(1+p[2])
    print("Best XI parameters for counts=", xip, "Likelihood value=", -negllk, file=sys.stderr) #, "augmented entropy=", sum(weights*datapoints)*ent1(xip) #, "Sum LL+ENT=", sum(weights*datapoints)*ent(xip)+llk 
    
    locxi=xif(inv_x, dy, xip)
    print("Gained mass=", 1- sum(dx*locxi)/sum(dx), file=sys.stderr)
    print("First 10 XI vals:", locxi[:10], file=sys.stderr)
    print("First 10 counts:", dx[:10], file=sys.stderr)
    print("First 10 uniq/counts:", dy[:10], file=sys.stderr)
    print("First 10 discounted counts:", dx[:10]*locxi[:10], file=sys.stderr)
#    sys.exit(0)
    return xip
            
            
x0=0.5
def tune_xi(counts, count_counts,  xif, N_XI_PARAMS, indexes=None, W0=None, xi_arg="kn", xi_arg_is_count_dist=False):
    
    print("Tuning XI function...", file=sys.stderr)
    if xi_arg=="wb" and N_XI_PARAMS > 1:
        print("############### WARNING ###############", file=sys.stderr)
        print("Using large number of parameters with", file=sys.stderr)
        print("Witten-Bell discounting may lead to", file=sys.stderr)
        print("unexpected results !!!", file=sys.stderr)
        print("#######################################", file=sys.stderr)
        
    def ll_wb_multi(p,  ddp, indexes, w0, mask, xi_args, xi_mask):
        if any(p<0) or any(p>1e100):
            return  -1e100        
        try:
#            pweights=weights[gt(datapoints, pivot)]
#            pdatapoints=datapoints[gt(datapoints, pivot)]
#            d_pdatapoints=pdatapoints-pivot
#            mask=gt(ddp, 0)
            locxi=xif(xi_args, None, p) #*sum(disc_w*datapoints)/sum(wdp)
            
            ll1_all=sum(data[mask])*log(locxi[0])
            
            
            ll1_parts=sum(add.reduceat(data, indexes)[xi_mask[1:]]*log(locxi[1:]))
            ll0=sum(w0[xi_mask]*log(1-locxi))
            
            return (ll1_all+ll1_parts+ll0) #+penal
        except FloatingPointError:
            
            return -1e100     
#    negll_wb_multi=lambda p, ddp, i, w0, m, a, m2: -ll_wb_multi(p, ddp, i, w0, m, a, m2)
    
    def ll_kn_multi(p,  ddp, indexes, w0, mask, xi_args):
        if any(p<0) or any(p>1e100):
            return  -1e100        
        try:
#            pweights=weights[gt(datapoints, pivot)]
#            pdatapoints=datapoints[gt(datapoints, pivot)]
#            d_pdatapoints=pdatapoints-pivot
#            mask=gt(ddp, 0)
            locxi=zeros_like(ddp, dtype=float)
            locxi[mask]=xif(xi_args[mask], None, p) #*sum(disc_w*datapoints)/sum(wdp)
            
            ll1=sum(data[mask]*log(locxi[mask]))
            
            ll0=w0[0]*log(sum((1-locxi[mask])*ddp[mask]))
            ll00=sum(w0[1:]*ma.log(add.reduceat((1-locxi)*ddp, indexes)))
            return (2*ll1+ll0+ll00) #+penal
        except FloatingPointError:
            return -1e100     
#    negll_kn_multi=lambda p, ddp, i, w0, m, a: -ll_kn_multi(p, ddp, i, w0, m, a)
    def ll(p,  ddp,  w0,  xi_args, i0):
#        if any(p<0) or any(p>1e100):
#            return  -1e100        
#        p=exp(p)
        try:
#            pweights=weights[gt(datapoints, pivot)]
#            pdatapoints=datapoints[gt(datapoints, pivot)]
#            d_pdatapoints=pdatapoints-pivot
#            mask=gt(ddp, 0)
            
            locxi=ddp*(1-xif(xi_args, None, exp(p))) #*sum(disc_w*datapoints)/sum(wdp)
            maximum.accumulate(locxi, out=locxi)
            
            
            ll1=sum(counts[i0:]*count_counts[i0:]*log(ddp-locxi))
            
            ll0=w0*log(sum(locxi*count_counts[i0:]))
            
#            print >>sys.stderr, "Value of:", p, "=", (ll1+ll0) 
            return (ll1+ll0) #+penal
        except FloatingPointError:
#            print >>sys.stderr, "Value of:", p, "= -INF" 
            return -inf #1e100     
            
    def ll1(p,  ddp,  w0,  xi_args, i0):
#        if any(p<0) or any(p>1e100):
#            return  -1e100        
#        p=exp(p)
        try:
#            pweights=weights[gt(datapoints, pivot)]
#            pdatapoints=datapoints[gt(datapoints, pivot)]
#            d_pdatapoints=pdatapoints-pivot
#            mask=gt(ddp, 0)
            
            locxi=xif(xi_args, None, exp(p)) #*sum(disc_w*datapoints)/sum(wdp)
            
            ll1=sum(counts[i0:]*count_counts[i0:]*log(locxi))
            
            ll0=w0*log(sum((1-locxi)*count_counts[i0:]*ddp))
            return (ll1+ll0) #+penal
        except FloatingPointError:
            return -inf #1e100     
    negll=lambda p, ddp, w0,  a, i0=1: -ll(p, ddp, w0, a, i0)
    i0=1
    
    if W0 is None:
#        pivot=1. #datapoints[0]
    #    print >>sys.stderr, "First term:", wdp[datapoints<=pivot], "sum:", sum(wdp[datapoints<=pivot])
        
        
        print("\nThe unseen data mass will be estimated through leaving-one-out", file=sys.stderr)
    #    dp=data[:, 0]
    #    w0=sum(wdp[less_eq(datapoints, pivot)])
    #    w0=sum(clip(dp, None,  pivot))
    #    w0=sum(wdp[less_eq(datapoints, pivot)]) +pivot*sum(weights[gt(datapoints, pivot)])
        print("Leaving %d from each occurrence: total number of data points: %d" %(counts[0], sum(count_counts)), file=sys.stderr)
      
        ddp=(counts- counts[0])[1:].astype(float)
        
        w0=count_counts[0]
#        w0=pivot*len(data) #sum(ddp<=0)
        if indexes is not None:
            W0=empty(len(indexes)+1,dtype=float)
            W0[0]=w0
            W0[1:]=pivot*add.reduceat(ddp<=0,indexes)
            w0=W0
            
    else:
        
        print("\nThe unseen data mass is fixed (w0=",W0,")", file=sys.stderr)
        ddp=counts.astype(float)
        w0=W0
        i0=0
    

    curxi=ones(N_XI_PARAMS)    
    if  xi_arg.startswith( "wb"):
        print("Witten-Bell discouting strategy: XI depends on the ratio of unique over the total number of data points", file=sys.stderr)
        
        if indexes is None:
            xi_arg=ones(sum(mask), dtype=float)*(float(sum(mask))/sum(ddp)  )
        else:
#            xi_arg=empty_like(W0)
            m2=ones_like(W0, dtype=bool)
            p_sddp=add.reduceat(ddp , indexes)
            m2[1:]=p_sddp>0
            xi_arg=empty(sum(m2), dtype=float)
            xi_arg[0]=(float(sum(mask))/sum(ddp)  )
            xi_arg[1:]=add.reduceat(mask, indexes)[m2[1:]].astype(float)/p_sddp[m2[1:]]
            
        
            negll=lambda p, ddp, i, w0, m, a: -ll_wb_multi(p, ddp, i, w0, m, a, m2)
    elif xi_arg== "kn":
        print("Kneser-Ney/Good-Turing discouting strategy: XI depends on the counts of data points", file=sys.stderr)
        
        if indexes is not None:
            xi_arg=zeros_like(ddp, dtype=float)
            xi_arg[mask]=ddp[mask]**-1
            
            negll=lambda p, ddp, i, w0, m, a: -ll_kn_multi(p, ddp, i, w0, m, a)
        else:
            if not xi_arg_is_count_dist:
                xi_arg=ddp**-1
            else:
                slog=sum(count_counts[i0:]*(log(ddp)))
                scc=sum(count_counts[i0:])
#            cprobs=count_counts/float(scc)
#            def dist(p):
#                
#                pre=-(1+p[0]**2)*log(counts)+p[1]
#                return sum((pre-cprobs)**2)
#            def pll(p):
#                x0=1./(1+p**2)
#                
#                alpha=1+scc/(slog-scc*log(x0))
#                return -((log(alpha-1)+(alpha-1)*log(x0))*scc -alpha*slog)
#                
#            x0p, negllk, _, _, _, _=fmin_powell(pll, 1.0,args=( ), maxiter=1000,maxfun=1000, full_output=True, disp=0)
#            x0p, negllk, _, _, _, _=fmin_powell(dist, [1., 1.],args=( ), maxiter=1000,maxfun=1000, full_output=True, disp=0)
#            print >>sys.stderr, "X0P=", x0p
#                x0=0.5 #1./(1+x0p**2)
                alpha=1+scc/(slog-scc*log(x0))
                print("Alpha=", alpha, file=sys.stderr)
#            alpha=1+x0p[0]**2
#            x0=((alpha-1)*x0p[1])**(1./(1-alpha))
                xi_arg=2*(alpha-1)*(2*ddp)**-alpha
                print("First 10 XI-ARG:", xi_arg[:10], file=sys.stderr)
        
#    if xi_arg=="wb":         
#        #by LOO the unique will be all but those appearing once
#        du=float(sum(ddp>0))
#        ds=sum(ddp)        
#        f=w0/sum(data[ddp>0])
#        print >>sys.stderr, "XI depends on unique left context occurrence (Witten-Bell)"
#        print >>sys.stderr, "Closed form xi value=", 1/(1+f)
#        
#        xi_arg=ones(sum(ddp>0), dtype=float)*(du/ds) #data[:, 2]-data[:, 1]    
#        cfxi= 1/(1+f) #sum(ddp)/sum(data)
#        def err(p):
#            if any(p<0) or any(p>1e100):
#                return  1e100        
#            return (xif(du/ds, None, p) -cfxi)**2
#        xip, negllk, _, _, _, _=fmin_powell(err,curxi, maxiter=1000,maxfun=1000, full_output=True, disp=0)
#        print >>sys.stderr, "Closed form parameters=", xip, "corresponding likelihood:", ll(xip, ddp, w0, xi_arg)
#        print >>sys.stderr,"XI value for 1=", xif(1, None, xip), "D=", 1-xif(1, None, xip)
#    elif xi_arg=="occurrence":
#        #by LOO the total occurrence will be unique occurrnces        
#        print >>sys.stderr, "XI depends on the total number of occurrences of the left context"
#        xi_arg=data[:, 3]-data[:, 2]    
#    else:        
#        print >>sys.stderr, "XI depends on the count"
#        xi_arg=ddp[ddp>0]**-1
#    restw=sum((datapoints[gt(datapoints, pivot)] -pivot)*weights[gt(datapoints, pivot)])
#    print >>sys.stderr, "Sum weight=",sum(wdp),"Removed weight:", w0, "Rest weight:",restw,"removed+rest=",w0+restw,"weight of the first 3 kept elements:", (datapoints[gt(datapoints, pivot)][:3] -pivot)[:3]*weights[gt(datapoints, pivot)][:3]
#    kept=sum(wdp)-w0
    
    print("W0=",w0, file=sys.stderr) #,"Entropy deleted:", w0*log(w0), "Entrop
    xip, negllk, _, _, _, _=fmin_powell(negll,curxi,args=( ddp, w0, xi_arg, i0), maxiter=1000,maxfun=1000, full_output=True, disp=0)
#    xip, negllk,d=fmin_l_bfgs_b(negll,curxi,args=( ddp, w0, xi_arg, i0), approx_grad=1, maxiter=1000,maxfun=1000, disp=0)
    xip=exp(xip)
        
    
    print("Best XI parameters=", xip, "Likelihood value=", -negllk, file=sys.stderr) #, "augmented entropy=", sum(weights*datapoints)*ent1(xip) #, "Sum LL+ENT=", sum(weights*datapoints)*ent(xip)+llk 
#    sys.exit(0)
    return xip
            



    

        
    print("Tuning XI function", file=sys.stderr)
    def pstd(x, p):
        m=sum(x*p)
        return sqrt(sum(p*(x-m)**2))
        
    N=len(data)
       
    if hasattr(dist, '__call__'):#dist is a fcunction?
        datapoints=unique(data) 
        weights=dist(datapoints)
    else:
        datapoints, ind=unique(data, return_inverse=True) 
        
        weights=bincount(ind)
        
    print("First 10 data points: %s"%(",".join("%g"%x for x in  datapoints[:10])), file=sys.stderr)
    print("First 10 weights: %s"%(",".join("%g"%x for x in  weights[:10])), file=sys.stderr)
    
    x_weights={"x":datapoints, "w":weights}
    normalizer=1.
    if contract:
        print("Number of unique data points before contracting:", len(datapoints), file=sys.stderr)
        distdata, datapoints=sample_data(datapoints, dist, N, 500, selection=MAX_SELECTION)    
        normalizer=sum(distdata)
        distdata/=normalizer
#        regularize_dist(datapoints, dist, N, 1000)    
#        print >>sys.stderr, "Data points=", datapoints
#        print >>sys.stderr, "Dist data points=", distdata
#    else:
#        distdata=dist(datapoints)
##       print >>sys.stderr, "Dist data before norm="%(distdata)
##       print >>sys.stderr, "Sum(distdata)= %g"%(sum(distdata))
#        distdata=distdata/sum(distdata)
#        print >>sys.stderr, "Keeping the data points unchanged. (no contraction)"
    print("Final number of unique data points:", len(datapoints), "Sum weights:", sum(weights), "Sum weighted data points:",  sum(weights*datapoints), file=sys.stderr)
#    
#    try:
#        tdp=transx(datapoints)
#    except TypeError:
#        tdp=array([f(datapoints) for f in transx]) 
#        
#    if contract:
#        tdp=tdp*distdata/dist(datapoints)
#        
#    print >>sys.stderr, "Transformed data points=", tdp
#    print >>sys.stderr, "Distribution of data points=", distdata    
#    print >>sys.stderr, "tdp shape:", tdp.shape
        
#    return normalizer, x_weights, ga_tune(datapoints, weights, xif, N_XI_PARAMS, del_rate=DEL_RATE, maxiter=10)
     
#    x=0.3*(datapoints)
    wdp=weights*datapoints
    K=3
#    gamma=empty((len(datapoints), K))
        
        
#    gamma/=sum(gamma, axis=1)[:, newaxis]
#    gamma[:,-1]=1-sum(gamma[:,:-1],axis=1) ##make sure
#    
#    w0=0.3*sum(weights*datapoints)
    curxi=ones(N_XI_PARAMS)       
    seterr(over="ignore", under="ignore")
    def w02pivot(mass, datapoints, weights):
        
        wdp=datapoints*weights
        if (0<mass<1):
            mass=mass*sum(wdp)
        
        def eval_pivot(pivot):
            if pivot <=0 or pivot >= max(datapoints):
                return 1e100
            return abs(sum(wdp[less_eq(datapoints, pivot)])+pivot*sum(weights[gt(datapoints, pivot)])-mass)
        p0=min(datapoints)
        pivot, dist, _, _, _, _=fmin_powell(eval_pivot,p0,args=( ), maxiter=1000,maxfun=1000, full_output=True, disp=0)
#        print >>sys.stderr, "MASS=", mass, "==> PIVOT=", pivot, "Distance=", dist
        return pivot, mass
    def pivot2w0(pivot, datapoints, weights):
        wdp=datapoints*weights
        return pivot, (sum(wdp[less_eq(datapoints, pivot)])+pivot*sum(weights[gt(datapoints, pivot)]))
           
    
    def ll(p, pivot, w0, datapoints, weights):
        if any(p<0) or any(p>1e100):
            return  -1e100        
        try:
            pweights=weights[gt(datapoints, pivot)]
            pdatapoints=datapoints[gt(datapoints, pivot)]
            d_pdatapoints=pdatapoints-pivot
            locxi=xif(d_pdatapoints**-1, pweights, p) #*sum(disc_w*datapoints)/sum(wdp)
            
            ll1=sum(pweights*pdatapoints*log(locxi*d_pdatapoints))
            
            ll0=w0*log(sum((1-locxi)*pweights*d_pdatapoints))
            return (ll1+ll0) #+penal
        except FloatingPointError:
            return -1e100     
#   
#    def ll(p, pivot, w0, datapoints, weights):
#        if any(p<0) or any(p>1e100):
#            return  -1e100        
##        try:
#        pweights=weights[gt(datapoints, pivot)]
#        pdatapoints=datapoints[gt(datapoints, pivot)]
#        d_pdatapoints=pdatapoints-pivot
##        print >>sys.stderr, "p=", p, "Discounted data:", d_pdatapoints
#        locxi=xif(d_pdatapoints**-1, pweights, p) #*sum(disc_w*datapoints)/sum(wdp)
#        logp=log(where(gt(locxi, 0), locxi*d_pdatapoints, 1)) #log(locxi*d_pdatapoints) #
#        ll1=sum(pweights*pdatapoints*logp)
#        
#        ll0=w0*log(sum((1-locxi)*pweights*d_pdatapoints))
#        return (ll1+ll0) #+penal
#        except FloatingPointError:
#            return -1e100     
   
    negll=lambda p, piv, w0, d, w: -ll(p, piv, w0, d, w)
    
    
    pivot=1. #datapoints[0]
#    print >>sys.stderr, "First term:", wdp[datapoints<=pivot], "sum:", sum(wdp[datapoints<=pivot])
    
    print("\nStarting at PIVOT=",pivot, file=sys.stderr)
    w0=sum(wdp[less_eq(datapoints, pivot)])  #+pivot*sum(weights[gt(datapoints, pivot)])
    restw=sum((datapoints[gt(datapoints, pivot)] -pivot)*weights[gt(datapoints, pivot)])
    print("Sum weight=",sum(wdp),"Removed weight:", w0, "Rest weight:",restw,"removed+rest=",w0+restw,"weight of the first 3 kept elements:", (datapoints[gt(datapoints, pivot)][:3] -pivot)[:3]*weights[gt(datapoints, pivot)][:3], file=sys.stderr)
    kept=sum(wdp)-w0
    
    print("W0=",w0, file=sys.stderr) #,"Entropy deleted:", w0*log(w0), "Entropy kept:", kept*log(kept), "Sum=", w0*log(w0)+kept*log(kept)
    curxi=ones(N_XI_PARAMS)       
    xip, negllk, _, _, _, _=fmin_powell(negll,curxi,args=(pivot, w0, datapoints, weights), maxiter=1000,maxfun=1000, full_output=True, disp=0)
    print("Best XI parameters=", xip, "Likelihood value=", -negllk, file=sys.stderr) #, "augmented entropy=", sum(weights*datapoints)*ent1(xip) #, "Sum LL+ENT=", sum(weights*datapoints)*ent(xip)+llk 
    curxip=copy(xip)
    
    if W0 is not None:
#        W0*=1.2
        pivot, _=w02pivot(W0, datapoints, weights)
#        w0=sum(wdp[less_eq(datapoints, pivot)]) +pivot*sum(weights[gt(datapoints, pivot)])
        w0=W0*sum(wdp)
        print("\nBased on MASS W0=",W0,"Estimated number=",W0*sum(wdp),"PIVOT=",pivot, "Small w0=", w0, file=sys.stderr)
        axip, negllk, _, _, _, _=fmin_powell(negll,curxi,args=(pivot, w0, datapoints, weights), maxiter=1000,maxfun=1000, full_output=True, disp=0)
        print("Best alternative XI parameters=", axip, "Likelihood value=", -negllk, file=sys.stderr) #
        
#    cweights=copy(weights)
#    for _ in range(5):
#        locxi=xif(datapoints**-1, weights, curxip)
#        dd=locxi*datapoints
##        pivot=sum((1-locxi)*wdp)/sum(weights)
##        print >>sys.stderr, "new pivot value=", pivot
#        
#        w0=sum(wdp)- sum(dd*weights)#datapoints[less_eq(datapoints, pivot)]*cweights[less_eq(datapoints, pivot)])+pivot*sum(cweights[gt(datapoints, pivot)])
#        print >>sys.stderr, "W0=",w0
#        
#        curxip, negllk, _, _, _, _=fmin_powell(negll_nop,curxi,args=(w0, dd, weights), maxiter=1000,maxfun=1000, full_output=True, disp=0)
#        print >>sys.stderr, "New XI params:", curxip, "W0=", w0
    print("", file=sys.stderr)
        
        
    return normalizer, x_weights, xip

    
PREC=1e-8

def tune_xi_w0(data,  xif, N_XI_PARAMS, indexes=None, W0=None, xi_arg="kn", xi_arg_is_count_dist=False):
    if W0 is None :
        raise
    print("Tuning XI function (association version)...", file=sys.stderr)
    if xi_arg=="wb" and N_XI_PARAMS > 1:
        print("############### WARNING ###############", file=sys.stderr)
        print("Using large number of parameters with", file=sys.stderr)
        print("Witten-Bell discounting may lead to", file=sys.stderr)
        print("unexpected results !!!", file=sys.stderr)
        print("#######################################", file=sys.stderr)
        
    def ll_wb_multi(p,  ddp, indexes, w0, mask, xi_args, xi_mask):
        if any(p<0) or any(p>1e100):
            return  -1e100        
        try:
#            pweights=weights[gt(datapoints, pivot)]
#            pdatapoints=datapoints[gt(datapoints, pivot)]
#            d_pdatapoints=pdatapoints-pivot
#            mask=gt(ddp, 0)
            locxi=xif(xi_args, None, p) #*sum(disc_w*datapoints)/sum(wdp)
            
            ll1_all=sum(data[mask])*log(locxi[0])
            
            
            ll1_parts=sum(add.reduceat(data, indexes)[xi_mask[1:]]*log(locxi[1:]))
            ll0=sum(w0[xi_mask]*log(1-locxi))
            
            return (ll1_all+ll1_parts+ll0) #+penal
        except FloatingPointError:
            
            return -1e100     
#    negll_wb_multi=lambda p, ddp, i, w0, m, a, m2: -ll_wb_multi(p, ddp, i, w0, m, a, m2)
    
    def ll_kn_multi(p,  ddp, indexes, w0, mask, xi_args):
        if any(p<0) or any(p>1e100):
            return  -1e100        
        try:
#            pweights=weights[gt(datapoints, pivot)]
#            pdatapoints=datapoints[gt(datapoints, pivot)]
#            d_pdatapoints=pdatapoints-pivot
#            mask=gt(ddp, 0)
            locxi=zeros_like(ddp, dtype=float)
            locxi[mask]=xif(xi_args[mask], None, p) #*sum(disc_w*datapoints)/sum(wdp)
            
            ll1=sum(data[mask]*log(locxi[mask]))
            
            ll0=w0[0]*log(sum((1-locxi[mask])*ddp[mask]))
            ll00=sum(w0[1:]*ma.log(add.reduceat((1-locxi)*ddp, indexes)))
            return (2*ll1+ll0+ll00) #+penal
        except FloatingPointError:
            return -1e100     
#    negll_kn_multi=lambda p, ddp, i, w0, m, a: -ll_kn_multi(p, ddp, i, w0, m, a)
    MAX_E=-1e70
    def ll(p,  dd,  w0,  xi_args):
#        if any(p<0) or any(p>1e100):
#            return  -1e100        
#        p=exp(p)
        try:
#            pweights=weights[gt(datapoints, pivot)]
#            pdatapoints=datapoints[gt(datapoints, pivot)]
#            d_pdatapoints=pdatapoints-pivot
#            mask=gt(ddp, 0)
            print("Evaluating p=", exp(p), file=sys.stderr)
            
            locxi=dd*(1-xif(xi_args, None, exp(p))) #*sum(disc_w*datapoints)/sum(wdp)
            print("First 10 Ds=", locxi[:10], file=sys.stderr)
            maximum.accumulate(locxi, out=locxi)
            
            print("After accumulation 10 Ds=", locxi[:10], file=sys.stderr)
            print("Problematic data ponts:", dd[dd<=locxi], file=sys.stderr)
            print("Problematic discounts:", locxi[dd<=locxi], file=sys.stderr)
            ll1=sum(log(dd-locxi))
            print("LL1:", ll1, file=sys.stderr)
            ll0=w0*log(sum(locxi))
            print("LL1:", ll1, "; LL0=", ll0, file=sys.stderr)
            print("Value of:", exp(p), "=", (ll1+ll0), file=sys.stderr) 
            return max(ll1+ll0, MAX_E) #+penal
        except FloatingPointError:
            print("Value of:", exp(p), "= -INF", file=sys.stderr) 
            return MAX_E     
            
    def ll1(p,  dd,  w0,  xi_args):
#        if any(p<0) or any(p>1e100):
#            return  -1e100        
#        p=exp(p)
        try:
#            pweights=weights[gt(datapoints, pivot)]
#            pdatapoints=datapoints[gt(datapoints, pivot)]
#            d_pdatapoints=pdatapoints-pivot
#            mask=gt(ddp, 0)
            
            locxi=xif(xi_args, None, exp(p)) #*sum(disc_w*datapoints)/sum(wdp)
            
            ll1=sum(dd*log(locxi))
#            ll1=sum(log(locxi))
            
            ll0=w0*log(sum((1-locxi)*dd))
            return max(ll1+ll0, MAX_E) #+penal
            return (ll1+ll0) #+penal
        except FloatingPointError:
            return MAX_E     
            return -inf #1e100     
    negll=lambda p, dd, w0,  a=1: -ll1(p, dd, w0, a)    
#    data.sort()

    i0=1
    if True:
        
        print("\nThe unseen data mass is fixed (w0=",W0,")", file=sys.stderr)
#        ddp=counts.astype(float)
#        w0=W0
#        i0=0
    

    curxi=ones(N_XI_PARAMS)    
    if  xi_arg.startswith( "wb"):
        print("Witten-Bell discouting strategy: XI depends on the ratio of unique over the total number of data points", file=sys.stderr)
        
        if indexes is None:
            xi_arg=ones(sum(mask), dtype=float)*(float(sum(mask))/sum(ddp)  )
        else:
#            xi_arg=empty_like(W0)
            m2=ones_like(W0, dtype=bool)
            p_sddp=add.reduceat(ddp , indexes)
            m2[1:]=p_sddp>0
            xi_arg=empty(sum(m2), dtype=float)
            xi_arg[0]=(float(sum(mask))/sum(ddp)  )
            xi_arg[1:]=add.reduceat(mask, indexes)[m2[1:]].astype(float)/p_sddp[m2[1:]]
            
        
            negll=lambda p, ddp, i, w0, m, a: -ll_wb_multi(p, ddp, i, w0, m, a, m2)
    elif xi_arg== "kn":
        print("Kneser-Ney/Good-Turing discouting strategy: XI depends on the counts of data points", file=sys.stderr)
        
        if indexes is not None:
            xi_arg=zeros_like(ddp, dtype=float)
            xi_arg[mask]=ddp[mask]**-1
            
            negll=lambda p, ddp, i, w0, m, a: -ll_kn_multi(p, ddp, i, w0, m, a)
        else:
            if not xi_arg_is_count_dist:
                xi_arg=data**-1
            else:
                slog=sum(count_counts[i0:]*(log(ddp)))
                scc=sum(count_counts[i0:])
#            cprobs=count_counts/float(scc)
#            def dist(p):
#                
#                pre=-(1+p[0]**2)*log(counts)+p[1]
#                return sum((pre-cprobs)**2)
#            def pll(p):
#                x0=1./(1+p**2)
#                
#                alpha=1+scc/(slog-scc*log(x0))
#                return -((log(alpha-1)+(alpha-1)*log(x0))*scc -alpha*slog)
#                
#            x0p, negllk, _, _, _, _=fmin_powell(pll, 1.0,args=( ), maxiter=1000,maxfun=1000, full_output=True, disp=0)
#            x0p, negllk, _, _, _, _=fmin_powell(dist, [1., 1.],args=( ), maxiter=1000,maxfun=1000, full_output=True, disp=0)
#            print >>sys.stderr, "X0P=", x0p
#                x0=0.5 #1./(1+x0p**2)
                alpha=1+scc/(slog-scc*log(x0))
                print("Alpha=", alpha, file=sys.stderr)
#            alpha=1+x0p[0]**2
#            x0=((alpha-1)*x0p[1])**(1./(1-alpha))
                xi_arg=2*(alpha-1)*(2*ddp)**-alpha
                print("First 10 XI-ARG:", xi_arg[:10], file=sys.stderr)
        
#    if xi_arg=="wb":         
#        #by LOO the unique will be all but those appearing once
#        du=float(sum(ddp>0))
#        ds=sum(ddp)        
#        f=w0/sum(data[ddp>0])
#        print >>sys.stderr, "XI depends on unique left context occurrence (Witten-Bell)"
#        print >>sys.stderr, "Closed form xi value=", 1/(1+f)
#        
#        xi_arg=ones(sum(ddp>0), dtype=float)*(du/ds) #data[:, 2]-data[:, 1]    
#        cfxi= 1/(1+f) #sum(ddp)/sum(data)
#        def err(p):
#            if any(p<0) or any(p>1e100):
#                return  1e100        
#            return (xif(du/ds, None, p) -cfxi)**2
#        xip, negllk, _, _, _, _=fmin_powell(err,curxi, maxiter=1000,maxfun=1000, full_output=True, disp=0)
#        print >>sys.stderr, "Closed form parameters=", xip, "corresponding likelihood:", ll(xip, ddp, w0, xi_arg)
#        print >>sys.stderr,"XI value for 1=", xif(1, None, xip), "D=", 1-xif(1, None, xip)
#    elif xi_arg=="occurrence":
#        #by LOO the total occurrence will be unique occurrnces        
#        print >>sys.stderr, "XI depends on the total number of occurrences of the left context"
#        xi_arg=data[:, 3]-data[:, 2]    
#    else:        
#        print >>sys.stderr, "XI depends on the count"
#        xi_arg=ddp[ddp>0]**-1
#    restw=sum((datapoints[gt(datapoints, pivot)] -pivot)*weights[gt(datapoints, pivot)])
#    print >>sys.stderr, "Sum weight=",sum(wdp),"Removed weight:", w0, "Rest weight:",restw,"removed+rest=",w0+restw,"weight of the first 3 kept elements:", (datapoints[gt(datapoints, pivot)][:3] -pivot)[:3]*weights[gt(datapoints, pivot)][:3]
#    kept=sum(wdp)-w0
    
    print("W0=",W0, file=sys.stderr) #,"Entropy deleted:", w0*log(w0), "Entrop
    xip, negllk, _, _, _, _=fmin_powell(negll,curxi,args=( data, W0, xi_arg), maxiter=1000,maxfun=1000, full_output=True, disp=0)
    xip=exp(xip)
        
    
    print("Best XI parameters=", xip, "Likelihood value=", -negllk, file=sys.stderr) #, "augmented entropy=", sum(weights*datapoints)*ent1(xip) #, "Sum LL+ENT=", sum(weights*datapoints)*ent(xip)+llk 
#    sys.exit(0)
    return xip
            



    

        
    print("Tuning XI function", file=sys.stderr)
    def pstd(x, p):
        m=sum(x*p)
        return sqrt(sum(p*(x-m)**2))
        
    N=len(data)
       
    if hasattr(dist, '__call__'):#dist is a fcunction?
        datapoints=unique(data) 
        weights=dist(datapoints)
    else:
        datapoints, ind=unique(data, return_inverse=True) 
        
        weights=bincount(ind)
        
    print("First 10 data points: %s"%(",".join("%g"%x for x in  datapoints[:10])), file=sys.stderr)
    print("First 10 weights: %s"%(",".join("%g"%x for x in  weights[:10])), file=sys.stderr)
    
    x_weights={"x":datapoints, "w":weights}
    normalizer=1.
    if contract:
        print("Number of unique data points before contracting:", len(datapoints), file=sys.stderr)
        distdata, datapoints=sample_data(datapoints, dist, N, 500, selection=MAX_SELECTION)    
        normalizer=sum(distdata)
        distdata/=normalizer
#        regularize_dist(datapoints, dist, N, 1000)    
#        print >>sys.stderr, "Data points=", datapoints
#        print >>sys.stderr, "Dist data points=", distdata
#    else:
#        distdata=dist(datapoints)
##       print >>sys.stderr, "Dist data before norm="%(distdata)
##       print >>sys.stderr, "Sum(distdata)= %g"%(sum(distdata))
#        distdata=distdata/sum(distdata)
#        print >>sys.stderr, "Keeping the data points unchanged. (no contraction)"
    print("Final number of unique data points:", len(datapoints), "Sum weights:", sum(weights), "Sum weighted data points:",  sum(weights*datapoints), file=sys.stderr)
#    
#    try:
#        tdp=transx(datapoints)
#    except TypeError:
#        tdp=array([f(datapoints) for f in transx]) 
#        
#    if contract:
#        tdp=tdp*distdata/dist(datapoints)
#        
#    print >>sys.stderr, "Transformed data points=", tdp
#    print >>sys.stderr, "Distribution of data points=", distdata    
#    print >>sys.stderr, "tdp shape:", tdp.shape
        
#    return normalizer, x_weights, ga_tune(datapoints, weights, xif, N_XI_PARAMS, del_rate=DEL_RATE, maxiter=10)
     
#    x=0.3*(datapoints)
    wdp=weights*datapoints
    K=3
#    gamma=empty((len(datapoints), K))
        
        
#    gamma/=sum(gamma, axis=1)[:, newaxis]
#    gamma[:,-1]=1-sum(gamma[:,:-1],axis=1) ##make sure
#    
#    w0=0.3*sum(weights*datapoints)
    curxi=ones(N_XI_PARAMS)       
    seterr(over="ignore", under="ignore")
    def w02pivot(mass, datapoints, weights):
        
        wdp=datapoints*weights
        if (0<mass<1):
            mass=mass*sum(wdp)
        
        def eval_pivot(pivot):
            if pivot <=0 or pivot >= max(datapoints):
                return 1e100
            return abs(sum(wdp[less_eq(datapoints, pivot)])+pivot*sum(weights[gt(datapoints, pivot)])-mass)
        p0=min(datapoints)
        pivot, dist, _, _, _, _=fmin_powell(eval_pivot,p0,args=( ), maxiter=1000,maxfun=1000, full_output=True, disp=0)
#        print >>sys.stderr, "MASS=", mass, "==> PIVOT=", pivot, "Distance=", dist
        return pivot, mass
    def pivot2w0(pivot, datapoints, weights):
        wdp=datapoints*weights
        return pivot, (sum(wdp[less_eq(datapoints, pivot)])+pivot*sum(weights[gt(datapoints, pivot)]))
           
    
    def ll(p, pivot, w0, datapoints, weights):
        if any(p<0) or any(p>1e100):
            return  -1e100        
        try:
            pweights=weights[gt(datapoints, pivot)]
            pdatapoints=datapoints[gt(datapoints, pivot)]
            d_pdatapoints=pdatapoints-pivot
            locxi=xif(d_pdatapoints**-1, pweights, p) #*sum(disc_w*datapoints)/sum(wdp)
            
            ll1=sum(pweights*pdatapoints*log(locxi*d_pdatapoints))
            
            ll0=w0*log(sum((1-locxi)*pweights*d_pdatapoints))
            return (ll1+ll0) #+penal
        except FloatingPointError:
            return -1e100     
#   
#    def ll(p, pivot, w0, datapoints, weights):
#        if any(p<0) or any(p>1e100):
#            return  -1e100        
##        try:
#        pweights=weights[gt(datapoints, pivot)]
#        pdatapoints=datapoints[gt(datapoints, pivot)]
#        d_pdatapoints=pdatapoints-pivot
##        print >>sys.stderr, "p=", p, "Discounted data:", d_pdatapoints
#        locxi=xif(d_pdatapoints**-1, pweights, p) #*sum(disc_w*datapoints)/sum(wdp)
#        logp=log(where(gt(locxi, 0), locxi*d_pdatapoints, 1)) #log(locxi*d_pdatapoints) #
#        ll1=sum(pweights*pdatapoints*logp)
#        
#        ll0=w0*log(sum((1-locxi)*pweights*d_pdatapoints))
#        return (ll1+ll0) #+penal
#        except FloatingPointError:
#            return -1e100     
   
    negll=lambda p, piv, w0, d, w: -ll(p, piv, w0, d, w)
    
    
    pivot=1. #datapoints[0]
#    print >>sys.stderr, "First term:", wdp[datapoints<=pivot], "sum:", sum(wdp[datapoints<=pivot])
    
    print("\nStarting at PIVOT=",pivot, file=sys.stderr)
    w0=sum(wdp[less_eq(datapoints, pivot)])  #+pivot*sum(weights[gt(datapoints, pivot)])
    restw=sum((datapoints[gt(datapoints, pivot)] -pivot)*weights[gt(datapoints, pivot)])
    print("Sum weight=",sum(wdp),"Removed weight:", w0, "Rest weight:",restw,"removed+rest=",w0+restw,"weight of the first 3 kept elements:", (datapoints[gt(datapoints, pivot)][:3] -pivot)[:3]*weights[gt(datapoints, pivot)][:3], file=sys.stderr)
    kept=sum(wdp)-w0
    
    print("W0=",w0, file=sys.stderr) #,"Entropy deleted:", w0*log(w0), "Entropy kept:", kept*log(kept), "Sum=", w0*log(w0)+kept*log(kept)
    curxi=ones(N_XI_PARAMS)       
    xip, negllk, _, _, _, _=fmin_powell(negll,curxi,args=(pivot, w0, datapoints, weights), maxiter=1000,maxfun=1000, full_output=True, disp=0)
    print("Best XI parameters=", xip, "Likelihood value=", -negllk, file=sys.stderr) #, "augmented entropy=", sum(weights*datapoints)*ent1(xip) #, "Sum LL+ENT=", sum(weights*datapoints)*ent(xip)+llk 
    curxip=copy(xip)
    
    if W0 is not None:
#        W0*=1.2
        pivot, _=w02pivot(W0, datapoints, weights)
#        w0=sum(wdp[less_eq(datapoints, pivot)]) +pivot*sum(weights[gt(datapoints, pivot)])
        w0=W0*sum(wdp)
        print("\nBased on MASS W0=",W0,"Estimated number=",W0*sum(wdp),"PIVOT=",pivot, "Small w0=", w0, file=sys.stderr)
        axip, negllk, _, _, _, _=fmin_powell(negll,curxi,args=(pivot, w0, datapoints, weights), maxiter=1000,maxfun=1000, full_output=True, disp=0)
        print("Best alternative XI parameters=", axip, "Likelihood value=", -negllk, file=sys.stderr) #
        
#    cweights=copy(weights)
#    for _ in range(5):
#        locxi=xif(datapoints**-1, weights, curxip)
#        dd=locxi*datapoints
##        pivot=sum((1-locxi)*wdp)/sum(weights)
##        print >>sys.stderr, "new pivot value=", pivot
#        
#        w0=sum(wdp)- sum(dd*weights)#datapoints[less_eq(datapoints, pivot)]*cweights[less_eq(datapoints, pivot)])+pivot*sum(cweights[gt(datapoints, pivot)])
#        print >>sys.stderr, "W0=",w0
#        
#        curxip, negllk, _, _, _, _=fmin_powell(negll_nop,curxi,args=(w0, dd, weights), maxiter=1000,maxfun=1000, full_output=True, disp=0)
#        print >>sys.stderr, "New XI params:", curxip, "W0=", w0
    print("", file=sys.stderr)
        
        
    return normalizer, x_weights, xip



def estimate_unseen_poisson_mix1(idata, indexes=None, mix_dist="gamma", args=None):
#    from scipy.optimize import broyden1, broyden2, newton_krylov, anderson
    
    olderr=seterr(all="raise")
    seterr(under="ignore")
    
    uv, ind=unique(idata, return_inverse=True)         
    ff=bincount(ind).astype(float)
    
    def gammap_ln(a, b, x):        
        return gammaln(x+b)-gammaln(b)-gammaln(x+1)-b*log(1+1./a)-x*log(a+1)
    def tgamma_ll(params, x, ff, sumff):
        if any(params<=0):
            return -1e100
        a, b=params
        try:
            lpis=gammap_ln(a, b, x)
            ll=sum(ff*lpis)-sumff*log(sum(exp(lpis)))
            return  ll        
        except FloatingPointError:            
            return -1e100     
    def gamma_p0(params, x):
        a, b=params
        p0=(1+1/a)**-b
        print("Untruncated P0=", p0, file=sys.stderr)
        p0=p0/(p0+sum(exp(gammap_ln(a, b, x))))
        print("Truncated P0=", p0, file=sys.stderr)
        return p0
        
        
    def lindleyp_ln(a, x):        
        return 2*log(a)+log(a+x+2)-(x+3)*log(a+1)
    def tlindley_ll(a, x, ff, sumff):
        if any(a<=0):
            return -1e100
#        a, b=params
        try:
            lpis=lindleyp_ln(a, x)
            ll=sum(ff*lpis)-sumff*log(sum(exp(lpis)))
            return  ll        
        except FloatingPointError:            
            return -1e100     
    def lindley_p0(a, x):
        
        p0=a**2*(a+2)*(a+1)**-3
        print("Untruncated P0=", p0, file=sys.stderr)
        p0=p0/(p0+sum(exp(lindleyp_ln(a, x))))
        print("Truncated P0=", p0, file=sys.stderr)
        return p0
        
    sff=sum(ff)
    l=len(idata)
    
    print("\nPoisson-Gamma\n", file=sys.stderr)
    negll=lambda p, x, lx, sx:-tgamma_ll(p, x, lx, sx)
    p0=[1., 1.]          
    p, negllk, _, _, _, _=fmin_powell(negll,p0,args=(uv, ff, sff), maxiter=1000,maxfun=1000, full_output=True, disp=0)
    pr0= gamma_p0(p, uv)
    print("MLE: params(alpha,m)=", p, "log-likelihood value:", -negllk, "P0=", pr0, "N0=", l*pr0/(1-pr0), file=sys.stderr)
    
    print("\nPoisson-Lindley\n", file=sys.stderr)
    negll=lambda p, x, lx, sx:-tlindley_ll(p, x, lx, sx)
    p0=1.
    p, negllk, _, _, _, _=fmin_powell(negll,p0,args=(uv, ff, sff), maxiter=1000,maxfun=1000, full_output=True, disp=0)
    pr0= lindley_p0(p, uv)
    print("MLE: params(alpha,m)=", p, "log-likelihood value:", -negllk, "P0=", pr0, "N0=", l*pr0/(1-pr0), file=sys.stderr)
    
#    sys.exit(0)
        
def estimate_unseen_poisson_mix_bounded(xi, nxi, mix_dist="mix-gamma-mle", npoints=-1):
    
    olderr=seterr(all="raise")
    seterr(under="ignore")

    def true_gammainc(a, x):
        if all(a)>0:
            return gamma(a)*gammainc(a, x)
        if any(a == a.astype(int)):
            raise FloatingPointError("Incomplete Gamma is not defined for negative integers")            
        if not isinstance(a,  ndarray):
            return (true_gammainc(1+a, x)+exp(a*log(x)-x))/a
        r=empty_like(a)
        m=a>0
        r[m]=gamma(a[m])*gammainc(a[m], x[m])
        r[~m]=(true_gammainc(1+a[~m], x[~m])+exp(a[~m]*log(x[~m])-x[~m]))/a[~m]
        return r
        if a == int(a):
            raise FloatingPointError("Incomplete Gamma is not defined for negative integers")
        return (true_gammainc(1+a, x)+exp(a*log(x)-x))/a
        
    pareto_p=lambda p, x: exp((log(p[0])+p[0]*log(p[1])-gammaln(x+1)-log(1-(p[1]/p[2])**p[0]))+log(gammainc(x-p[0], p[2])-gammainc(x-p[0], p[1]))+gammaln(x-p[0]))
    pareto_p0=lambda p: exp(p[0]*log(p[1])-log(1-(p[1]/p[2])**p[0]))*(true_gammainc(-p[0], p[2])-true_gammainc(-p[0], p[1]))
    def KS(p, xi, nxi, lenx, sumx):
        if any(p<=0):
            return 1e100
        p=1./(1+p[0]), p[1], p[2]
#        print >>sys.stderr, "Evaluating for", p
#        print >>sys.stderr, "first term =", exp(log(p[0])+p[0]*log(p[1])-gammaln(xi+1)-log(1-(p[1]/p[2])**p[0]))
#        print >>sys.stderr, "sec term =", gammainc(xi-p[0], p[2])-gammainc(xi-p[0], p[1])
#        print >>sys.stderr, "3rd term =", gamma(xi-p[0])
        
        
        try:
            probs=pareto_p(p, xi)
        except FloatingPointError:            
            return 1e100    
#        probs/=sum(probs)
        ex_cdf=cumsum(probs)
        ob_cdf=cumsum(nxi/float(lenx))
        
        return sum(abs(ex_cdf-ob_cdf))
    def chi2(p, xi, nxi, lenx, sumx):
        if any(p<=0):
            return 1e100
#        p=1./(1+p[0]), p[1], p[2]
#        print >>sys.stderr, "Evaluating for", p
#        print >>sys.stderr, "first term =", exp(log(p[0])+p[0]*log(p[1])-gammaln(xi+1)-log(1-(p[1]/p[2])**p[0]))
#        print >>sys.stderr, "sec term =", gammainc(xi-p[0], p[2])-gammainc(xi-p[0], p[1])
#        print >>sys.stderr, "3rd term =", gamma(xi-p[0])
        
        
        try:
            probs=pareto_p(p, xi)
            mp0=1-pareto_p0(p)
            if not (0<mp0<=1):
                return 1e100    
        except FloatingPointError:            
            return 1e100    
#        print >>sys.stderr, "ALPHA=",p[0],"mp0=", mp0
#        probs/=sum(probs)
        e=lenx*probs/mp0
        if any(isnan(e)) or any(isinf(e)):         
            return 1e100    
            
        return sum((e-nxi)**2/e)
    def tbpareto_ll(p, xi, nxi, lenx, sumx):
        if any(p<=0):
            return -1e100
        a, l0, l1=p
#        p=1./(1+p[0]), p[1], p[2]
#        a=1./(1+a)
        
        try:
            p0=pareto_p0(p)
            if not (0<p0<=1):
                return -1e100
            print("Evaluating LL for params:", a, l0, l1, "p_0=", p0, file=sys.stderr)
#            print >>sys.stderr, "first gamma inc term=", gammainc(xi[0]-p[0], p[2])
#            print >>sys.stderr, "second gamma inc term=", gammainc(xi[0]-p[0], p[2])
#            print >>sys.stderr, "Gamma ln term=", gammaln(xi-p[0])
            ll= lenx*(log(p[0])+p[0]*log(p[1])-log(1-(p[1]/p[2])**p[0]))+sum(nxi*(log(true_gammainc(xi-p[0], p[2])-true_gammainc(xi-p[0], p[1])) )) -lenx*log(1-p0) #+gammaln(xi-p[0])
            return ll
            
        except FloatingPointError:            
            return -1e100    
    
    def bpareto_moments_params(xi, nxi):
        mu1=average(xi, weights=nxi) #mean(data)
        mu2=average(xi*(xi-1), weights=nxi) #mean(data*(data-1))
        mu3=average(xi*(xi-1)*(xi-2), weights=nxi) 
        mu_r= lambda p, r: exp(log(p[0])+p[0]*log(p[1])-log(1-(p[1]/p[2])**p[0])-log(r-p[0]))*(p[2]**(r-p[0])-p[1]**(r-p[0]))
        def F(p):
            if any(p<=0):
                return 1e100
#            print >>sys.stderr, "Testing:", X
#            p[0]=1/(1+p[0])
            try:
                mp0=1-pareto_p0(p)
                if not (0<mp0<=1):
                    return 1e100
                
    #            try:
                return (mu_r(p, 1)/mp0-mu1)**2 + (mu_r(p, 2)/mp0-mu2)**2 +  (mu_r(p, 3)/mp0-mu3)**2 
                return sum((array([X[1]/(X[0]-1)/p0, 2*X[1]**2/(X[0]**2-3*X[0]+2)/p0])-[mu1, mu2])**2)
               
            except FloatingPointError:            
                return 1e100
                
        print("mu1=", mu1, file=sys.stderr)
        print("mu2=", mu2, file=sys.stderr)        
        print("mu3=", mu3, file=sys.stderr)
        
        param0=array([.5, amin(nxi), amax(nxi)])
        
#        print >>sys.stderr, "Parameters of the non-truncated =",param0
        (a, l0, l1), mine, _, _, _, _=fmin_powell(F,param0,args=(), maxiter=1000,maxfun=1000, full_output=True, disp=0)
#        c, b=broyden2(F, [c,b], maxiter=1000)
#        a=1./(1+a)
        print("Parameters of the zero-truncated=",(a, l0, l1) , "min-error=", mine, file=sys.stderr)
        return a, l0, l1
    
    s=sum(xi*nxi)
    l=sum(nxi) #len(da
    
    negll=lambda p, x, nx, lx, sx:-tbpareto_ll(p, x, nx, lx, sx)       
    param0=array([.5, amin(nxi), amax(nxi)])
    p, negllk, _, _, _, _=fmin_powell(negll,param0,args=(xi, nxi, l, s), maxiter=1000,maxfun=1000, full_output=True, disp=0)
#    p=1./(1+p[0]), p[1], p[2]
    pr0= pareto_p0(p)
    N0= l*pr0/(1-pr0)
    print("Poisson-Pareto MLE: parameters (alpha,lambda_0, lambda_1)=", p, "log-likelihood value:", -negllk, "P0=", pr0, "N0=",N0, file=sys.stderr)
#    return N0
    p=bpareto_moments_params(xi, nxi)
    pr0= pareto_p0(p)    
    N0= l*pr0/(1-pr0)
    print("Poisson-Pareto Method-of-Moments: parameters (alpha,m)=", p, "P0=", pr0, "N0=",N0, file=sys.stderr)
    param0=array([.5, amin(nxi), amax(nxi)])
    p, negllk, _, _, _, _=fmin_powell(chi2,param0,args=(xi, nxi, l, s), maxiter=1000,maxfun=1000, full_output=True, disp=0)
#    p=1./(1+p[0]), p[1], p[2]
    pr0= pareto_p0(p)
    N0= l*pr0/(1-pr0)
    print("Poisson-Pareto Xi2: parameters (alpha,lambda_0, lambda_1)=", p, "Xi2 fit value:", negllk, "P0=", pr0, "N0=",N0, file=sys.stderr)
#    return N0         
    param0=array([.5, amin(nxi), amax(nxi)])
    p, negllk, _, _, _, _=fmin_powell(KS,param0,args=(xi, nxi, l, s), maxiter=1000,maxfun=1000, full_output=True, disp=0)
#    p=1./(1+p[0]), p[1], p[2]
    pr0= pareto_p0(p)
    N0= l*pr0/(1-pr0)
    print("Poisson-Pareto KS: parameters (alpha,lambda_0, lambda_1)=", p, "D:", negllk, "P0=", pr0, "N0=",N0, file=sys.stderr)
    sys.exit(0)
    
    return N0         
    sys.exit(0)
    
SMALLEST_LOMAX_SHAPE=3.8
def estimate_unseen_poisson_mix(xi, nxi, mix_dist="mix-gamma-mle", npoints=-1):
#    from scipy.optimize import broyden1, broyden2, newton_krylov, anderson
    
    olderr=seterr(all="raise")
    seterr(under="ignore")
    
    gamma_p= lambda params, x:exp(gammaln(x+params[1])-gammaln(x+1)-gammaln(params[1])-(params[1]+x)*log(params[0]+1)+params[1]*log(params[0]))
    lindley_p=    lambda a, x: a**2*(a+2.+x)*((a+1.)**-(x+3.))
    lomax_p=    lambda params, x:  params[0]*exp(x*log(params[1])+log(hyperu(x+1, x+1-params[0], params[1])))
    blindley_p=    lambda p, x: exp(2*log(p[0])-(3+x)*log(p[0]+1))*((x+p[0]+2)*(gammainc(x+1, p[2])- gammainc(x+1, p[1])) -exp((x+1)*log(p[2])-p[2]) +exp((x+1)*log(p[1])-p[1]))
    
    gamma_p0= lambda params:(1.+1./params[0])**-params[1]
    lindley_p0=    lambda a:  a**2*(a+2.)*((a+1.)**-3.)
    blindley_p0=    lambda p:  p[0]**2/(p[0]+1)**3*((p[0]+2)*(gammainc(1, p[2])-gammainc(1, p[1])) -exp(log(p[2])-p[2]) + exp(log(p[1])-p[1]) )
    
    lomax_p0=    lambda params:  params[0]*hyperu(1., 1.-params[0], params[1])
    
    def tgamma_ll(params, xi, nxi, lenx, sumx):
        if any(params<=0):
            return -1e100
        a, b=params
        try:
#            print >> sys.stderr, "Testing params=", params, 
            ll=sum(nxi*gammaln(xi+b))-lenx*gammaln(b)+lenx*b*log(a)-(lenx*b+sumx)*log(a+1)-lenx*log(1-(1+1./a)**-b)
#            print >> sys.stderr, "llk=", ll
            return  ll
        
        except FloatingPointError:
            
            return -1e100     
    
    def tblindley_ll(p, xi, nxi, lenx, sumx):
        if any(p<=0):
            return -1e100
#        a, b=params
        try:
            p0=blindley_p0(p)
            ll=(2*lenx*log(p[0])-(3*lenx+sumx)*log(p[0]+1)) + sum(nxi*((xi+p[0]+2)*(gammainc(xi+1, p[2])- gammainc(xi+1, p[1])) -exp((xi+1)*log(p[2])-p[2]) +exp((xi+1)*log(p[1])-p[1])))-lenx*log(1-p0)
    #            print >> sys.stderr, "Testing params=", a, 
    #            print >> sys.stderr, "llk=", ll
            return  ll
        
        except FloatingPointError:
            
            return -1e100         
            
    def tlindley_ll(a, xi, nxi, lenx, sumx):
        if any(a<=0):
            return -1e100
#        a, b=params
        try:
#            print >> sys.stderr, "Testing params=", a, 
            ll= 2*lenx*log(a)+sum(nxi*log(a+2.+xi))-sumx*log(a+1)-lenx*log(a**2+3.*a+1.)
#            print >> sys.stderr, "llk=", ll
            return  ll
        
        except FloatingPointError:
            
            return -1e100         
            
     
    lowest_lomax_bound=array([SMALLEST_LOMAX_SHAPE, 0])
    def tlomax_ll(params, xi, nxi, lenx, sumx):
        
        if any(params-lowest_lomax_bound<=0):
            return -1e100
        c, b=params
#        if c<3:
#            return -1e100
            
        try:
#            print >> sys.stderr, "Testing params=", a, 
##            pr=lomax_p(params, xi)
##            return sum(nxi*log(pr/sum(pr)))
            ll= lenx*log(c)+sumx*log(b)+sum(nxi*log(hyperu(xi+1., xi+1.-c, b))) -lenx*log(1.-c*hyperu(1, 1-c, b))
#            print >> sys.stderr, "llk=", ll
            return  ll
        
        except FloatingPointError:
            
            return -1e100         
      
    def gamma_moments_params(xi, nxi):
        mu1=average(xi, weights=nxi) #mean(data)
        mu2=average(xi*(xi-1), weights=nxi) #mean(data*(data-1))
        def F(X):
            if any(X<=0):
                return 1e100
#            print >>sys.stderr, "Testing:", X
            try:
                p0=(1-(1+1/X[0])**-X[1])
    #            try:
                e1=(X[1]/(X[0])/p0/mu1-1)**2
                e2=((X[1]+X[1]**2)/(X[0]**2)/p0/mu2-1)**2
                return e1+e2
                return (X[1]/(X[0])/p0-mu1)**2 + ((X[1]+X[1]**2)/(X[0]**2)/p0-mu2)**2
                return sum((array([X[1]/(X[0]-1)/p0, 2*X[1]**2/(X[0]**2-3*X[0]+2)/p0])-[mu1, mu2])**2)
               
            except FloatingPointError:            
                return 1e100
                
        print("mu1=", mu1, file=sys.stderr)
        print("mu2=", mu2, file=sys.stderr)
        c=mu1/(mu2-mu1**2)
        b=mu1*c
        
        print("Parameters of the non-truncated c=", c, "b=", b, file=sys.stderr)
        (c, b), mine, _, _, _, _=fmin_powell(F,[c, b],args=(), maxiter=1000,maxfun=1000, full_output=True, disp=0)
#        c, b=broyden2(F, [c,b], maxiter=1000)
        print("Parameters of the zero-truncated c=", c, "b=", b, "min-error=", mine, file=sys.stderr)
        return c, b
    
    def lindley_moments_params(xi, nxi):
        mu1=average(xi, weights=nxi) #mean(data)
        
        def F(X):
            if any(X<=0):
                return 1e100
#            print >>sys.stderr, "Testing:", X
            try:
                p0=(1.-X**2*(X+2.)/(X+1.)**3)
    #            try:
                return ((X+2.)/(X**2+X)/p0/mu1-1)**2 
                return ((X+2.)/(X**2+X)/p0-mu1)**2 
                return sum((array([X[1]/(X[0]-1)/p0, 2*X[1]**2/(X[0]**2-3*X[0]+2)/p0])-[mu1, mu2])**2)
               
            except FloatingPointError:            
                return 1e100
                
        print("mu1=", mu1, file=sys.stderr)
        
        a=(1-mu1+sqrt(mu1**2+6*mu1+1))/mu1/2
        
        
        print("Parameters of the non-truncated a=", a, file=sys.stderr)
        a, mine, _, _, _, _=fmin_powell(F,a,args=(), maxiter=1000,maxfun=1000, full_output=True, disp=0)
#        c, b=broyden2(F, [c,b], maxiter=1000)
        print("Parameters of the zero-truncated a=", a, "min-error=", mine, file=sys.stderr)
        return a
            
    def lomax_moments_params(xi, nxi):
        mu1=average(xi, weights=nxi) #mean(data)
        mu2=average(xi*(xi-1), weights=nxi) #mean(data*(data-1))        
#        mu3=average(xi*(xi-1)*(xi-2), weights=nxi)
        N_MUS=2
        def F(X):
            if any(X<=0):
                return 1e100
            if X[0]<=N_MUS:
                return 1e100
#            print >>sys.stderr, "Testing:", X
            try:
                p0=(1-X[0]*hyperu(1, 1-X[0], X[1]))
                e1=(X[1]/(X[0]-1)/p0/mu1-1)**2
                e2=(2*X[1]**2/(X[0]**2-3*X[0]+2)/p0/mu2-1)**2
#                print >>sys.stderr, "Error1=", e1
#                print >>sys.stderr, "Error2=", e2
                return e1+e2
    #            try:
                return (X[1]/(X[0]-1)/p0-mu1)**2 + (2*X[1]**2/(X[0]**2-3*X[0]+2)/p0-mu2)**2 #+(6*X[1]**3/(X[0]-1)/(X[0]-2)/(X[0]-3)/p0-mu3)**2 
                return sum((array([X[1]/(X[0]-1)/p0, 2*X[1]**2/(X[0]**2-3*X[0]+2)/p0])-[mu1, mu2])**2)
               
            except FloatingPointError:            
                return 1e100
                
        print("mu1=", mu1, file=sys.stderr)
        print("mu2=", mu2, file=sys.stderr)
#        print >>sys.stderr, "mu3=", mu3
        c=1-mu2/(2*mu1**2-mu2)
        b=-mu1*mu2/(2*mu1**2-mu2)
        
        print("Parameters of the non-truncated c=", c, "b=", b, file=sys.stderr)
        (c, b), mine, _, _, _, _=fmin_powell(F,[c, b],args=(), maxiter=1000,maxfun=1000, full_output=True, disp=0)
#        c, b=broyden2(F, [c,b], maxiter=1000)
        print("Parameters of the zero-truncated c=", c, "b=", b, "min-error=", mine, file=sys.stderr)
        return c, b
    
    def chi2(p, prob_f, prob_p0, xi, nxi, lenx, sumx):
        if any(p<=0):
            return 1e100
#        p=1./(1+p[0]), p[1], p[2]
#        print >>sys.stderr, "Evaluating for", p
#        print >>sys.stderr, "first term =", exp(log(p[0])+p[0]*log(p[1])-gammaln(xi+1)-log(1-(p[1]/p[2])**p[0]))
#        print >>sys.stderr, "sec term =", gammainc(xi-p[0], p[2])-gammainc(xi-p[0], p[1])
#        print >>sys.stderr, "3rd term =", gamma(xi-p[0])
        
        
        try:
            probs=prob_f(p, xi)
            
#            mp0=1-prob_p0(p)
#            if not (0<mp0<=1):
#                return 1e100    
#        print >>sys.stderr, "ALPHA=",p[0],"mp0=", mp0
#            probs/=sum(probs)
#            print >>sys.stderr, "Probabilities=", probs
            e=lenx*probs  #/sum(probs[m])
            if any(isnan(e)) or any(isinf(e)):         
                return 1e100    
            m=probs>1e-50
            e=e[m]
            nis=nxi[m] #.astype(float)
#            e/=sum(e)
#            nis/=sum(nis)
#            print >>sys.stderr, "expected=", e
            return sum((e/nis-1)**2)
            return sum((e/nxi[m]-1)**2) #/e)    
        except FloatingPointError:            
            return 1e100    
#        except FloatingPointError:            
#            return 1e100    
    s=sum(xi*nxi)
    l=sum(nxi) #len(data)   
    
    if mix_dist=="mix-gamma-mle":
        negll=lambda p, x, nx, lx, sx:-tgamma_ll(p, x, nx, lx, sx)
        p0=array([1., 1.])
        p, negllk, _, _, _, _=fmin_powell(negll,p0,args=(xi, nxi, l, s), maxiter=1000,maxfun=1000, full_output=True, disp=0)
        pr0= gamma_p0(p)
        N0= l*pr0/(1-pr0)
        print("Poisson-Gamma MLE: parameters (alpha,m)=", p, "log-likelihood value:", -negllk, "P0=", pr0, "N0=",N0, file=sys.stderr)
        return N0     
        
    elif mix_dist=="mix-gamma-mom":
        p=gamma_moments_params(xi, nxi)
        pr0= gamma_p0(p)    
        
        N0= l*pr0/(1-pr0)
        print("Poisson-Gamma Method-of-Moments: parameters (alpha,m)=", p, "P0=", pr0, "N0=",N0, file=sys.stderr)
        return N0         
    
    elif mix_dist=="mix-lindley-mle":
        negll=lambda p, x, nx, lx, sx:-tlindley_ll(p, x, nx, lx, sx)        
        p0=1. 
        p, negllk, _, _, _, _=fmin_powell(negll,p0,args=(xi, nxi, l, s), maxiter=1000,maxfun=1000, full_output=True, disp=0)
        pr0= lindley_p0(p)
        N0= l*pr0/(1-pr0)
        print("Poisson-Lindley MLE: parameters (alpha)=", p, "log-likelihood value:", -negllk, "P0=", pr0, "N0=",N0, file=sys.stderr)
        return N0     
        
    elif mix_dist=="mix-lindley-mom":
        p=lindley_moments_params(xi, nxi)
        pr0= lindley_p0(p) 
        N0= l*pr0/(1-pr0)
        print("Poisson-Lindley Method-of-Moments: parameters (alpha)=", p, "P0=", pr0, "N0=",N0, file=sys.stderr)
        return N0     
        
    elif mix_dist=="mix-lindley-chi2":
        p0=1.
        p, negllk, _, _, _, _=fmin_powell(chi2,p0,args=(lindley_p, lindley_p0, xi, nxi, l, s), maxiter=1000,maxfun=1000, full_output=True, disp=0)
        pr0= lindley_p0(p)
        N0= l*pr0/(1-pr0)
        print("Poisson-Lindley Xi2 fit: parameters (alpha)=", p, "Fit score:",negllk,"P0=", pr0, "N0=",N0, file=sys.stderr)
        return N0     
        
    elif mix_dist=="mix-lomax-mle":
        negll=lambda p, x, nx, lx, sx:-tlomax_ll(p, x, nx, lx, sx)
        p0=array([SMALLEST_LOMAX_SHAPE, .5]  )
#    mask=data<=10
        if npoints <0:
            npoints=DEFAULT_NPOINTS
        ad=xi[:npoints]
        anxi=nxi[:npoints].astype(float)
        ## TEST ME!!!!
        anxi[-1]+=sum(nxi[npoints:])
#        anxi*=float(sum(nxi))/sum(anxi)
        ####################
        al=sum(anxi)
        a_s=sum(anxi*ad)
        p, negllk, _, _, _, _=fmin_powell(negll,p0,args=(ad, anxi, al, a_s), maxiter=1000,maxfun=1000, full_output=True, disp=0)   
        
        pr0= lomax_p0(p)     
        pr=lomax_p(p, ad)
        N0= l*pr0 /(1-pr0)
        print("Poisson-Lomax MLE: parameters (alpha,m)=", p, "log-likelihood value:", -negllk, "P0=", pr0, "N0=",N0, "sum_p=", sum(pr), file=sys.stderr)
 
#        p=lomax_moments_params(xi, nxi) 
#        pr=lomax_p(p, ad)
#        
#        print >>sys.stderr, "Poisson-Lomax MoM: parameters (alpha,m)=", p, "log-likelihood value:", -negll(p, ad, anxi, al, a_s), "sum_p=", sum(pr)
               
        return N0     
    elif mix_dist=="mix-lomax-mom":
        p=lomax_moments_params(xi, nxi)
        pr0= lomax_p0(p)
      
        N0= l*pr0/(1-pr0)
        print("Poisson-Lomax Method-of-moments: parameters (alpha,m)=", p, "P0=", pr0, "N0=",N0, file=sys.stderr)
        return N0     
    elif mix_dist=="mix-lomax-chi2":
        p0=[.5, .5]    
        if npoints <0:
            npoints=50 #DEFAULT_NPOINTS
        ad=xi[:npoints]
        anxi=nxi[:npoints] #.astype(float)
        ## TEST ME!!!!
#        anxi[-1]+=sum(nxi[npoints:])
#        anxi*=float(sum(nxi))/sum(anxi)
        ####################
        al=sum(anxi)
        a_s=sum(anxi*ad)
        p, negllk, _, _, _, _=fmin_powell(chi2,p0,args=(lomax_p, lomax_p0, ad, anxi, l, a_s), maxiter=1000,maxfun=1000, full_output=True, disp=0)
        pr0= lomax_p0(p)
        N0= l*pr0/(1-pr0)
        print("Poisson-Lomax Xi2 fit: parameters (alpha,m)=", p, "Fit score:",negllk,"P0=", pr0, "N0=",N0, file=sys.stderr)
        return N0     
    
    n1=sum(nxi[xi==1])
    n2=sum(nxi[xi==2]) #sum(data==2)   
    
    print("\nPoisson-Gamma\n", file=sys.stderr)
    negll=lambda p, x, nx, lx, sx:-tgamma_ll(p, x, nx, lx, sx)
    p0=[1., 1.]          
    p, negllk, _, _, _, _=fmin_powell(negll,p0,args=(xi, nxi, l, s), maxiter=1000,maxfun=1000, full_output=True, disp=0)
    pr0= gamma_p0(p)
    pr1=gamma_p(p, 1)    
    pr2=gamma_p(p, 2)    
    print("MLE: params(alpha,m)=", p, "log-likelihood value:", -negllk, "P0=", pr0, "N0=", l*pr0/(1-pr0), "n1'=", l*pr1/(1-pr0), "n1=", n1, "n1'/n1=", l*pr1/(1-pr0)/n1, "1'=", l*s*pr1/n1/(s*(1-pr0)+l*pr0), "n2'=", l*pr2/(1-pr0), "n2=", n2, "n2'/n2=", l*pr2/(1-pr0)/n2, "2'=", 2* l*s*pr2/n2/(s*(1-pr0)+l*pr0), file=sys.stderr)
    p=gamma_moments_params(xi, nxi)
    pr0= gamma_p0(p)    
    pr1=gamma_p(p, 1)   
    pr2=gamma_p(p, 2)     
    print("Moments method: params(alpha)=", p, "log-likelihood value:", -negllk, "P0=", pr0, "N0=", l*pr0/(1-pr0), "n1'=", l*pr1/(1-pr0), "n1=", n1, "n1'/n1=", l*pr1/(1-pr0)/n1, "1'=", l*s*pr1/n1/(s*(1-pr0)+l*pr0), "n2'=", l*pr2/(1-pr0), "n2=", n2, "n2'/n2=", l*pr2/(1-pr0)/n2, "2'=", 2* l*s*pr2/n2/(s*(1-pr0)+l*pr0), file=sys.stderr)
    
    p0=[1., 1.]          
    p, negllk, _, _, _, _=fmin_powell(chi2,p0,args=(gamma_p, gamma_p0, xi, nxi, l, s), maxiter=1000,maxfun=1000, full_output=True, disp=0)
    pr0= gamma_p0(p)
    pr1=gamma_p(p, 1)    
    pr2=gamma_p(p, 2)    
    print("Xi2: params(alpha,m)=", p, "Xi2 value:", negllk, "P0=", pr0, "N0=", l*pr0/(1-pr0), "n1'=", l*pr1/(1-pr0), "n1=", n1, "n1'/n1=", l*pr1/(1-pr0)/n1, "1'=", l*s*pr1/n1/(s*(1-pr0)+l*pr0), "n2'=", l*pr2/(1-pr0), "n2=", n2, "n2'/n2=", l*pr2/(1-pr0)/n2, "2'=", 2* l*s*pr2/n2/(s*(1-pr0)+l*pr0), file=sys.stderr)
    
    

    
    print("\nPoisson-Lindley\n", file=sys.stderr)
    negll=lambda p, x, nx, lx, sx:-tlindley_ll(p, x, nx, lx, sx)
    p0=1.
    p, negllk, _, _, _, _=fmin_powell(negll,p0,args=(xi, nxi, l, s), maxiter=1000,maxfun=1000, full_output=True, disp=0)
    pr0= lindley_p0(p)     
    pr1=lindley_p(p, 1)     
    pr2=lindley_p(p, 2)   
    print("MLE: params(alpha)=", p, "log-likelihood value:", -negllk, "P0=", pr0, "N0=", l*pr0/(1-pr0), "n1'=", l*pr1/(1-pr0), "n1=", n1, "n1'/n1=", l*pr1/(1-pr0)/n1, "1'=", l*s*pr1/n1/(s*(1-pr0)+l*pr0), "n2'=", l*pr2/(1-pr0), "n2=", n2, "n2'/n2=", l*pr2/(1-pr0)/n2, "2'=", 2* l*s*pr2/n2/(s*(1-pr0)+l*pr0), file=sys.stderr)
    if npoints <0:
        npoints=30 #DEFAULT_NPOINTS
    ad=xi[:npoints]
    anxi=nxi[:npoints] #.astype(float)
    ## TEST ME!!!!
#        anxi[-1]+=sum(nxi[npoints:])
#        anxi*=float(sum(nxi))/sum(anxi)
    ####################
    al=sum(anxi)
    a_s=sum(anxi*ad)
    negll=lambda p, x, nx, lx, sx:-tblindley_ll(p, x, nx, lx, sx)
    p0=array([1., amin(anxi), amax(anxi)])
    p, negllk, _, _, _, _=fmin_powell(negll,p0,args=(ad, anxi, al, a_s), maxiter=1000,maxfun=1000, full_output=True, disp=0)
    pr0= blindley_p0(p)     
    pr1=blindley_p(p, 1)     
    pr2=blindley_p(p, 2)   
    print("MLE bounded Lindley: params(alpha)=", p, "log-likelihood value:", -negllk, "P0=", pr0, "N0=", l*pr0/(1-pr0), "n1'=", l*pr1/(1-pr0), "n1=", n1, "n1'/n1=", l*pr1/(1-pr0)/n1, "1'=", l*s*pr1/n1/(s*(1-pr0)+l*pr0), "n2'=", l*pr2/(1-pr0), "n2=", n2, "n2'/n2=", l*pr2/(1-pr0)/n2, "2'=", 2* l*s*pr2/n2/(s*(1-pr0)+l*pr0), file=sys.stderr)
    
    
    
    p=lindley_moments_params(xi, nxi)
    pr0= lindley_p0(p)
    pr1=lindley_p(p, 1)    
    pr2=lindley_p(p, 2)    
    
    print("Moments method: params(alpha)=", p, "log-likelihood value:", -negllk, "P0=", pr0, "N0=", l*pr0/(1-pr0), "n1'=", l*pr1/(1-pr0), "n1=", n1, "n1'/n1=", l*pr1/(1-pr0)/n1, "1'=", l*s*pr1/n1/(s*(1-pr0)+l*pr0), "n2'=", l*pr2/(1-pr0), "n2=", n2, "n2'/n2=", l*pr2/(1-pr0)/n2, "2'=", 2* l*s*pr2/n2/(s*(1-pr0)+l*pr0), file=sys.stderr)
    
    
    
    p0= 1.
    p, negllk, _, _, _, _=fmin_powell(chi2,p0,args=(lindley_p, lindley_p0, xi, nxi, l, s), maxiter=1000,maxfun=1000, full_output=True, disp=0)
    pr0= lindley_p0(p)
    pr1=lindley_p(p, 1)    
    pr2=lindley_p(p, 2)    
    print("Xi2: params(alpha,m)=", p, "Xi2 value:", negllk, "P0=", pr0, "N0=", l*pr0/(1-pr0), "n1'=", l*pr1/(1-pr0), "n1=", n1, "n1'/n1=", l*pr1/(1-pr0)/n1, "1'=", l*s*pr1/n1/(s*(1-pr0)+l*pr0), "n2'=", l*pr2/(1-pr0), "n2=", n2, "n2'/n2=", l*pr2/(1-pr0)/n2, "2'=", 2* l*s*pr2/n2/(s*(1-pr0)+l*pr0), file=sys.stderr)
    
    sys.exit(0)
    print("\nPoisson-Lomax\n", file=sys.stderr)
    negll=lambda p, x, nx, lx, sx:-tlomax_ll(p, x, nx, lx, sx)
    p0=[.5, .5]    
#    mask=data<=10
    NPOINTS=90
    ad=xi[:NPOINTS]
    anxi=nxi[:NPOINTS].astype(float)
#    anxi*=float(sum(nxi))/sum(anxi)
#    anxi[-1]+=sum(nxi[NPOINTS:])
    al=sum(anxi)
    a_s=sum(anxi*ad)
#    p0=[1., 1.]
    p, negllk, _, _, _, _=fmin_powell(negll,p0,args=(ad, anxi, al, a_s), maxiter=1000,maxfun=1000, full_output=True, disp=0)   
    
    pr0= lomax_p0(p)     
    pr1=lomax_p(p, 1)     
    pr2=lomax_p(p, 2)   
    print("MLE: params(alpha)=", p, "log-likelihood value:", -negllk, "P0=", pr0, "N0=", l*pr0, "n1'=", l*pr1/(1-pr0), "n1=", n1, "n1'/n1=", l*pr1/(1-pr0)/n1, "1'=", l*s*pr1/n1/(s*(1-pr0)+l*pr0), "n2'=", l*pr2/(1-pr0), "n2=", n2, "n2'/n2=", l*pr2/(1-pr0)/n2, "2'=", 2* l*s*pr2/n2/(s*(1-pr0)+l*pr0), file=sys.stderr)

    
    ##
    p=lomax_moments_params(xi, nxi)

    pr0= lomax_p0(p)
    pr1=lomax_p(p, 1)    
    pr2=lomax_p(p, 2)   
    print("params(alpha)=", p, "log-likelihood value:", -negllk, "P0=", pr0, "N0=", l*pr0/(1-pr0), "n1'=", l*pr1/(1-pr0), "n1=", n1, "n1'/n1=", l*pr1/(1-pr0)/n1, "1'=", l*s*pr1/n1/(s*(1-pr0)+l*pr0), "n2'=", l*pr2/(1-pr0), "n2=", n2, "n2'/n2=", l*pr2/(1-pr0)/n2, "2'=", 2* l*s*pr2/n2/(s*(1-pr0)+l*pr0), file=sys.stderr)
    
    p0=[.5, .5]    
    p, negllk, _, _, _, _=fmin_powell(chi2,p0,args=(lomax_p, lomax_p0, ad, anxi, al, a_s), maxiter=1000,maxfun=1000, full_output=True, disp=0)
    pr0= lomax_p0(p)
    pr1=lomax_p(p, 1)    
    pr2=lomax_p(p, 2)    
    print("Xi2: params(alpha,m)=", p, "Xi2 value:", negllk, "P0=", pr0, "N0=", l*pr0/(1-pr0), "n1'=", l*pr1/(1-pr0), "n1=", n1, "n1'/n1=", l*pr1/(1-pr0)/n1, "1'=", l*s*pr1/n1/(s*(1-pr0)+l*pr0), "n2'=", l*pr2/(1-pr0), "n2=", n2, "n2'/n2=", l*pr2/(1-pr0)/n2, "2'=", 2* l*s*pr2/n2/(s*(1-pr0)+l*pr0), file=sys.stderr)
    
    sys.exit(0)
            
def estimate_unseen_poisson_pl(data, indexes=None, args=None):
    from scipy.special import gammaln, gamma, gammainc
    olderr=seterr(all="raise")
    seterr(under="ignore")
    power_by_fact_ln=lambda x, y: y*log(x)-gammaln(y+1)
    power_by_fact=lambda x, y: exp(y*log(x)-gammaln(y+1))    
    print("first 10 data points", data[:10], file=sys.stderr)

    def p0(alpha, lambda_0, lambda_1, n):
        l1_1_a=lambda_1**(1-alpha)
        l1_a=l1_1_a/lambda_1
        
        l0_1_a=lambda_0**(1-alpha)
        l0_a=l0_1_a/lambda_0
        
        if isfinite(lambda_1):
            ginc_0l1=gammainc(1-alpha, lambda_1)
            c=n*(1-alpha)/(l1_1_a-l0_1_a)
        else:
            ginc_0l1=1
            c=0
        ginc_0l0=gammainc(1-alpha, lambda_0)
        print("First=", (ginc_0l1-ginc_0l0)*gamma(1-alpha), "Second=", l1_a*exp(-lambda_1)-l0_a*exp(-lambda_0), file=sys.stderr)
        return c*((ginc_0l1-ginc_0l0)*gamma(1-alpha)+l1_a*exp(-lambda_1)-l0_a*exp(-lambda_0))/alpha
        
            
    def trunc_prob_ln(x, alpha, lambda_0, lambda_1, n):
        
        if isfinite(lambda_1):
            ginc_yl1=gammainc(x-alpha, lambda_1)
            ginc_0l1=gammainc(1-alpha, lambda_1)
        else:
            ginc_yl1=ginc_0l1=1
        
        ginc_yl0=gammainc(x-alpha, lambda_0)
        ginc_0l0=gammainc(1-alpha, lambda_0)
        l1_1_a=lambda_1**(1-alpha)
        l1_a=l1_1_a/lambda_1
        
        l0_1_a=lambda_0**(1-alpha)
        l0_a=l0_1_a/lambda_0
        
        
        if isfinite(lambda_1):
            c=n*(1-alpha)/(l1_1_a-l0_1_a)
        else:
            c=0
        print("C=", c, file=sys.stderr)
#        print >> sys.stderr, "Ginc_yl1=", ginc_yl1, "ginc_0l1=", ginc_0l1, "Ginc_yl0=", ginc_yl0, "ginc_0l0=", ginc_0l0
        return gammaln(x-alpha)+log(ginc_yl1-ginc_yl0)-log(1./c-((ginc_0l1-ginc_0l0)*gamma(1-alpha)+l1_a*exp(-lambda_1)-l0_a*exp(-lambda_0))/alpha)
#        return  gammaln(1-alpha)+log(ginc_1-(lambda_0/lambda_1)**x*ginc_0)\
#                    -log((lambda_1**(1-alpha)-lambda_0**(1-alpha))/(1-alpha)+ginc_0-ginc_1)
#        return x*log(lambda_1)-gammaln(x+1) + gammaln(1-alpha)+log(ginc_1-(lambda_0/lambda_1)**x*ginc_0)\
#                    -log((lambda_1**(1-alpha)-lambda_0**(1-alpha))/(1-alpha)+ginc_0-ginc_1)
    def ll(alpha, d, lower, upper, n):
        if alpha <=0 or alpha >=1:
            return -1e100
        try:
            print("Testing alpha=", alpha, end=' ', file=sys.stderr) #"First 10 log-probs=", trunc_prob_ln(d, alpha, lower, upper, n)[:10]
            ll=sum(trunc_prob_ln(d, alpha, lower, upper, n))
            print("Corresponding ll=", ll, file=sys.stderr)
            return ll
        except FloatingPointError:
            
            return -1e100     
    negll=lambda a, x, l0, l1, n: -ll(a, x, l0, l1, n)
    
    a0=.5       
    n=sum(data)
    l0=1.
    l1=amax(data)
    alpha, negllk, _, _, _, _=fmin_powell(negll,a0,args=(data, l0, l1, n), maxiter=1000,maxfun=1000, full_output=True, disp=0)
    print("Alpha=", alpha, "log-likelihood value:", -negllk, "P0=", p0(alpha,  l0, l1, n), file=sys.stderr)
    
    
    sys.exit(0)
    seterr(**olderr)
def estimate_unseen(data, indexes=None, method="efron-thisted", args=None):    
    print("Estimating the number of unseen ngrams; using '%s' method"%(method), file=sys.stderr)
    idata=data
    if not issubclass(idata.dtype.type,integer):
        idata=idata.astype(int)
    
    
    uv, ind=unique(idata, return_inverse=True)         
    ff=bincount(ind).astype(float)
    print("The first 5 frequencies of frequencies:", ff[:5], file=sys.stderr)
    print("Their cumulative sum:", cumsum(ff[:5]), file=sys.stderr)
#    return estimate_unseen_poisson_mix_bounded(uv, ff)
    
    n=sum(idata)
    uniq=float(len(idata))
    TAU=3
    if method=="efron-thisted":
        if args is None:
            args=HIGHEST_COUNT
        elif args<0:
            args=len(ff)
        coef=-hstack((ff[:args][::-1], [0]))
       
        p=polyval(coef, -1./n) #sum(ff[:args]/(-power(-n, arange(1, args+1))))
        
        if p<=0:
            raise TypeError("Estimated Efron-Thisted probability not appropriate: %g"%(p))
        estimated=len(idata)*p/(1-p)
        print("Estimated unseen probability:", p, file=sys.stderr)    
    elif method=="boneh-boneh-caron":
        old=seterr(under="ignore")
        estimated=sum(exp(log(ff)-uv))
        if estimated <= ff[0]:
            Ui_1=estimated
            while True:
                Ui=estimated + Ui_1*exp(-ff[0]/Ui_1)
                if abs(Ui-Ui_1)<=PREC:
                    estimated=Ui
                    break
            
                Ui_1=Ui
        seterr(**old)

    elif method=="bhat-sproat":
        estimated=sum(ff[:5]*[5./4, -5./4, 15./16, -15./32, 15./128])    
    elif method=="chao":
        estimated=(ff[0]/ff[1])*(ff[0]/2)
    elif method=="bcorrected-chao":
        estimated=(ff[0]/(ff[1]+1))*((ff[0]-1)/2)
    elif method=="extended-chao":
        estimated=3*(ff[0]/ff[1])**3*ff[2]/4
    
    elif method=="ace":
        s_rare=sum(ff[:TAU])
        n_rare=sum(uv[:TAU]*ff[:TAU])
        c_ace=1-1.*ff[0]/n_rare
        cv=max(0, (s_rare/c_ace)*(sum(uv[:TAU]*(uv[:TAU]-1)*ff[:TAU], dtype=float))/n_rare/(n_rare-1)-1)
        estimated=s_rare/c_ace + (ff[0]/c_ace)*cv - s_rare
        
    elif method=="nonparametric-lomax":        
        lmbd=(-4*ff[3]*ff[1]/ff[2]+3*ff[2]+ff[1])/(ff[0]-ff[1]**2/ff[2])
        a=lmbd*ff[0]/ff[1]-3*ff[2]/ff[1]+2
        estimated=(2*ff[1]+(a-1)*ff[0])/lmbd
    elif method=="":        
        a=(ff[1]+ff[0]/3-ff[2])/(ff[2]+ff[1]/3)        
        estimated=2*(a+1)*ff[1]-(2-a)*ff[0]
        
#        a=(ff[0]+uniq/3-ff[1])/(ff[1]+ff[0]/3)        
#        estimated=2*(a+1)*ff[0]-(2-a)*uniq
    elif method=="nonparametric-gp":        
        l0=ff[1]/ff[0]
        l1=ff[2]/ff[1]
        a=(20*ff[4]-24*l1*ff[3]-16*ff[3]+18*ff[2]*l1+6*ff[2]**2/ff[0]-4*ff[2]*l0)/(6*ff[2]*l1-4*ff[2]-2*ff[2]*l0)
        mu=(12*ff[3]-9*ff[2]-6*ff[2]*l0+4*ff[1]*l0+a*(3*ff[2]-2*ff[1]*l0))/ff[1]
        beta=(6*ff[2]-2*(2-a)*ff[1])/ff[0]-mu
        estimated=(2*ff[1]-(1-a)*ff[0])/beta
        
    elif method=="nonparametric-gamma":       
        l= 3*ff[0]*ff[2]
        m=(l-4*ff[1]**2)/(2*ff[1]**2-l)
        estimated=0.5*(1+1./m)*ff[0]**2/ff[1]
        
    elif method=="nonparametric-invgamma":        
        l=ff[0]/ff[1]
        a=(6*ff[2]-12*ff[3]*l-4*ff[1]+9*ff[2]*l)/(3*ff[2]*l-2*ff[1])
        mu=(6*ff[2]+(2*a-4)*ff[1])/(ff[0])

        estimated=(2*ff[1]-(1-a)*ff[0])/mu
        
    elif method=="nonparametric-geometric":        
        estimated=(ff[0]/ff[1])*(ff[0])
    
        
    elif method=="chao-bunge":
        mu=1-ff[0]*sum(uv[:TAU]**2*ff[:TAU])/sum(uv[:TAU]*ff[:TAU])**2
        estimated=sum(ff[1:TAU])/mu
    
    elif method=="gandolfi-sastri":
        u=len(idata)
        estimated=n*(u+ff[0]*( (ff[0] -n-u + sqrt(5*n**2 +2*n*(u-3*ff[0])+(u-ff[0])**2) )/(2*n) ))/(n-ff[0])-u
    
    elif method=="mix-exp":
        estimated=len(idata)/(mean(idata)-1)
    elif method=="good-turing":
#        estimated1=uniq/(n/ff[0]-1)
        estimated=(1./ff[0]-1./len(idata))**-1        
#        print >>sys.stderr, "Estimated 1=", estimated1, "; Estimated2=", estimated
#        estimated=len(idata)*float(ff[0])/(sum(data)-ff[0])
#        print >>sys.stderr, "Estimated 2=", estimated
#    elif method=="mix-gamma-mle":
#        estimate_unseen_poisson_mix(uv, ff)

#    elif method=="mix-gamma-mom":
#    elif method=="mix-lindley-mle":
#    elif method=="mix-lindley-mom":
#    elif method=="mix-lomax-mle":
#    elif method=="mix-lomax-mom":
    else:
        if method in MIX_METHODS:
            estimated=estimate_unseen_poisson_mix(uv, ff, method)
        else:
            raise UnknownEstimatorError("Unknown unseen estimator: '%s'. Should be one of %s"%(method, UNSEEN_ESTIMATORS))
        
    print("Estimated unseen number of ngrams:", estimated, file=sys.stderr)
    if indexes is not None:
        pest=empty(len(indexes)+1,dtype=float)
        pest[1:-1]=indexes[1:]-indexes[:-1]
        pest[-1]=len(idata)-indexes[-1]
        pest[1:]*=estimated/len(idata)
        pest[0]=estimated
        return pest
    return estimated
    
    
def mkdata_array(od,o):
    
    if o == 1:
#        print >>sys.stderr,"Order 1.. returning directly"
        return fromiter((v for v in list(od[o].values()) if v>0), dtype=float), [0]
    ng_dict=od[o]

    print("Sorting ngrams...", file=sys.stderr)
    st=time.time()
    sorted_ng=sorted(ng_dict)
    et=time.time()
    print("Done sorting in:", formatted_span(et-st), file=sys.stderr)
    d_arr=[] #empty(sum(1 for v in ng_dict.values() if v >0), dtype=int)
    d_ind=[0]
    
    cur_ctxt, _=sorted_ng[0].rsplit(None, 1)
    
#    ctxti=0
#    j=0
    for  k in sorted_ng:
        v=ng_dict[k]
#        print >>sys.stderr,"K=",k,"\tV=",v
        if v<=0:
            continue
        
        if not k.startswith(cur_ctxt+WORD_SEP):
#            print >>sys.stderr,"Context switch found.. K=",k,"\tCur-ctxt=",cur_ctxt,"Saving Index=",len(d_arr)
            d_ind.append(len(d_arr))
            
            cur_ctxt, _=k.rsplit(None, 1)
            
#        print >>sys.stderr,"Adding values to array:",v
        d_arr.append(v)
    print("", file=sys.stderr)
    return fromiter(d_arr, dtype=float), d_ind
    
gparams["xi_func"]=analyse_arg(options.xi_func , default="xi_hyp")
if options.smoothing =="wb" and False:
    for i in assigns[rank][0]:  
        if gparams["xi_func"][i] != "xi_hyp1":
            print("\n*********************************", file=sys.stderr)
            print("**           WARINIG           **", file=sys.stderr)
            print("*********************************\n", file=sys.stderr)
            print("Using discounting function '%s' with wb discounting is meaningless using 'xi_hyp1' instead\nYou can still use 'wb-mle' discounting with the discounting function you specified\n\n"%( gparams["xi_func"] ), file=sys.stderr)
            gparams["xi_func"]=ddict("xi_hyp1" )
            break
    
    
print("XI function form:", gparams["xi_func"], file=sys.stderr) # xi_func



def multilog(x, N):
    res=log10(1+x)
    for _ in range(1, max(1, int(N))):
        res=log10(1+res)        
    return res
    
SMALLEST_DISC=0.0001
LARGEST_DISC=.999
N_XI_PARAMS_hyp=1
XI_hyp=lambda x,w,p: (1+x)**-p
gparams["N_XI_PARAMS"]=ddict(N_XI_PARAMS_hyp)
gparams["XI"]=ddict(XI_hyp) #lambda x,w,p: (1.)/(1+x)**p)
comb_tune={}
#gparams["DXI"]=ddict(None)
for i in range(1, options.order+1):   
    if gparams["xi_func"][i]=="xi_hyp":
#        gparams["DXI"][i]=lambda x,w,p: -log(1+x)*(1+x)**-p
        pass
        N_XI_PARAMS=1
        XI=lambda x,w,p: (1+x)**-p
     
    elif gparams["xi_func"][i]=="xi_hyp1":
        gparams["N_XI_PARAMS"][i]=1
        gparams["XI"][i]=lambda x,w,p: 1./(1+p*x)
#        N_XI_PARAMS=2
#        XI=lambda x,w,p: (1+p[1]/(1+x)**p[0])/(1+p[1])

    elif gparams["xi_func"][i] =="xi_kn":
        gparams["N_XI_PARAMS"][i] =NBR_KN_CONSTS
        gparams["XI"][i] =lambda x,w, p: 1-p[clip(atleast_1d(x), 1, len(p)).astype(int)-1]/clip(atleast_1d(x), 1, None)
        
    elif gparams["xi_func"][i]=="xi_hyp2":
        gparams["N_XI_PARAMS"][i]=3
        gparams["XI"][i]=lambda x,w,p: (1)/(1+p[0]*x**p[1]+p[2])  
        
#        N_XI_PARAMS=2
#        XI=lambda x,w,p: (1.)/(1+(p[0]*x)**p[1])  

    elif gparams["xi_func"][i]=="xi_hyp3":
        gparams["N_XI_PARAMS"][i]=3
        gparams["XI"][i]=lambda x,w,p: (1+(p[1])/(1+p[0]*x)**(p[2]))/(1+(p[1]))
        
#        N_XI_PARAMS=3
#        XI=lambda x,w,p: (1+(p[1])/(1+p[0]*x)**(p[2]))/(1+(p[1]))
     
    elif gparams["xi_func"][i]=="xi_hyp4":
        gparams["N_XI_PARAMS"][i]=2
        gparams["XI"][i]=lambda x,w,p: (1.)/(1+p[0]*x**p[1])
        
#        N_XI_PARAMS=2
#        XI=lambda x,w,p: (1.)/(1+p[0]*x**p[1])


    elif gparams["xi_func"][i]=="xi_hyp5":
        gparams["N_XI_PARAMS"][i]=2
        gparams["XI"][i]=lambda x,w,p: (1.)/(1+p[0]*x**p[1])
        

#        N_XI_PARAMS=2
#        XI=lambda x,w,p: (1.)/(1+p[0]*x**p[1])
#        lambda x,w,p
    elif gparams["xi_func"][i]=="xi_hyp6":
        gparams["N_XI_PARAMS"][i]=3
        gparams["XI"][i]=lambda x,w,p: (1.)/(1+p[0]*x**p[2])**p[1]
        

#        N_XI_PARAMS=2
#        XI=lambda x,w,p: (1.)/(1+p[0]*x)**p[1]
#        
        
    elif gparams["xi_func"][i]=="xi_hyp7":
        gparams["N_XI_PARAMS"][i]=2
        gparams["XI"][i]=lambda x,w,p: (1.)/(1+p[0]*x+p[1]*x**2)
        

#        N_XI_PARAMS=1
#        XI=lambda x,w,p: (1.)/(1+p*x)
         
    elif gparams["xi_func"][i]=="xi_hyp8":
        gparams["N_XI_PARAMS"][i]=2
        gparams["XI"][i]=lambda x,w,p:(1+ (p[0])/(1+p[1]*x))/(1+p[0])
        

         
    elif gparams["xi_func"][i]=="xi_hyp9":
        gparams["N_XI_PARAMS"][i]=2
        gparams["XI"][i]=lambda x,w,p:((1+p[0]*x)**-p[1])
        
#        N_XI_PARAMS=2
#        XI=lambda x,w,p:(1+ (p[0])/(1+p[1]*x))/(1+p[0])
         
    elif gparams["xi_func"][i]=="xi_hyp_log":
        gparams["N_XI_PARAMS"][i]=2
        gparams["XI"][i]=lambda x,w,p: (1+p[0]*log1p(x))**-p[1]  
        
    elif gparams["xi_func"][i]=="xi_hyp_hyp_log":
        gparams["N_XI_PARAMS"][i]=3
        gparams["XI"][i]=lambda x,w,p: (1+p[0]*log1p(x)+p[1]*x)**-p[2]   
        
    elif gparams["xi_func"][i]=="xi_hyp_hyp_log2":
        gparams["N_XI_PARAMS"][i]=3
        gparams["XI"][i]=lambda x,w,p: ((1+p[0]*log10(x+1))/(1+p[1]*x))**-p[2]          
        
    elif gparams["xi_func"][i]=="xi_hyp_hyp_log3":
        gparams["N_XI_PARAMS"][i]=2
        gparams["XI"][i]=lambda x,w,p: ((1+p[0]*log10(x+1))/(1+p[0]*x))**-p[1]   
        
        
    elif gparams["xi_func"][i]=="xi_hyp_hyp_log4":
        gparams["N_XI_PARAMS"][i]=4
        gparams["XI"][i]=lambda x,w,p: (1+p[0]*log10(x+1))**-p[2]*(1+p[1]*x)**-p[3]   
        
    
    elif gparams["xi_func"][i]=="xi_hyp_hyp_log5":
        gparams["N_XI_PARAMS"][i]=2
        gparams["XI"][i]=lambda x,w,p: (1.)/(1+p[0]*log10(x+1)+p[1]*x)   

    elif gparams["xi_func"][i]=="xi_hyp_hyp_log6":
        gparams["N_XI_PARAMS"][i]=2
        gparams["XI"][i]=lambda x,w,p: (1.)/(1+p[0]*log10(x+1))**p[1]/(1+p[0]*x)**p[1]   
        
    elif gparams["xi_func"][i]=="xi_hyp_hyp_log7":
        gparams["N_XI_PARAMS"][i]=4
        gparams["XI"][i]=lambda x,w,p: (1.)/(1+p[0]*log10(x+1)+p[1]*x+p[3]*x**2)**p[2]  
        
    elif gparams["xi_func"][i]=="xi_hyp_hyp_log8":
        gparams["N_XI_PARAMS"][i]=4
        gparams["XI"][i]=lambda x,w,p: (1.)/(1+p[0]*log10(x+1)+p[1]*x+p[3]*log10(x+1)**2)**p[2]  
        
    elif gparams["xi_func"][i]=="xi_hyp_hyp_log9":
        gparams["N_XI_PARAMS"][i]=4
        gparams["XI"][i]=lambda x,w,p: (1.)/(1+p[0]*log10(x+1)**p[1]+p[2]*x**p[3])
        
    elif gparams["xi_func"][i]=="xi_hyp_hyp_log10":
        gparams["N_XI_PARAMS"][i]=3
        gparams["XI"][i]=lambda x,w,p: (1.)/(1+(p[0]*log10(x+1)+p[1]*x)**p[2])
        
    elif gparams["xi_func"][i]=="xi_hyp_hyp_log1":
        gparams["N_XI_PARAMS"][i]=4
        gparams["XI"][i]=lambda x,w,p: (1.)/(1+(p[0]*log10(x+1)+p[1]*x)**p[2])**p[3]
        

#        N_XI_PARAMS=2
#        XI=lambda x,w,p: (1.)/(1+p[0]*log10(x+1))**p[1]   
       
    elif gparams["xi_func"][i]=="xi_hyp_log1":
        gparams["N_XI_PARAMS"][i]=2
        gparams["XI"][i]=lambda x,w,p: 1./(1+p[0])+(p[0])/(1+p[0])/(1+p[1]*log10(x+1))
        

#        N_XI_PARAMS=2
#        XI=lambda x,w,p: 1./(1+p[0])+(p[0])/(1+p[0])/(1+p[1]*log10(x+1))

    elif gparams["xi_func"][i]=="xi_hyp_log2":
        gparams["N_XI_PARAMS"][i]=3
        gparams["XI"][i]=lambda x,w,p: (1+p[0]*log10(p[2]*x+1))**-p[1]   
        
    elif gparams["xi_func"][i]=="xi_comb_hyp":
        
        gparams["N_XI_PARAMS"][i]=2
        gparams["XI"][i]=lambda x,y,p: (1+x)**-(p[0]*(1+p[1]*y))
        comb_tune[i]=lambda x,w, p: (1+x)**-p
        
     
    elif gparams["xi_func"][i]=="xi_comb_hyp1":
        gparams["N_XI_PARAMS"][i]=2
        gparams["XI"][i]=lambda x,y,p: 1/(1+p[0]*x*(1+p[1]*y))
        comb_tune[i]=lambda x,w, p: 1/(1+p*x)
        
    elif gparams["xi_func"][i]=="xi_comb_hyp_sep":
        
        gparams["N_XI_PARAMS"][i]=2
#        gparams["XI"][i]=lambda x,y,p: (1+x**(1/(1+p[1]))*y**(p[1]/(1+p[1])))**-p[0]
        gparams["XI"][i]=lambda x,y,p: (1+x*(1/(1+p[1]))+y*(p[1]/(1+p[1])))**-p[0]
        
     
    elif gparams["xi_func"][i]=="xi_comb_hyp1_sep":
        gparams["N_XI_PARAMS"][i]=2
#        gparams["XI"][i]=lambda x,y,p: 1/(1+p[0]*x**(1/(1+p[1]))*y**(p[1]/(1+p[1])))
        gparams["XI"][i]=lambda x,y,p: 1/(1+p[0]*(x*(1/(1+p[1]))+y*(p[1]/(1+p[1]))))
        
##        gparams["XI"][i]=lambda x,y,p: 1./(1+p[0]*x) #+(1-p[2])/(1+p[1]*y)
#    elif gparams["xi_func"][i]=="xi_comb_loglin_hyp":        
#        gparams["N_XI_PARAMS"][i]=3
#        gparams["XI"][i]=lambda x,y,p: (1+x)**(-p[0]*p[2])*(1+y)**(-p[1]*(1-p[2]))
#     
#    elif gparams["xi_func"][i]=="xi_comb_loglin_hyp1":
#        gparams["N_XI_PARAMS"][i]=3
#        gparams["XI"][i]=lambda x,y,p: (1+p[0]*x)**-p[2]*(1+p[1]*y)**(-1+p[2])
    
        
    elif gparams["xi_func"][i]=="xi_comb_exp":
        gparams["N_XI_PARAMS"][i]=2
        gparams["XI"][i]=lambda x,y,p: (1+p[0]*(1+p[1]*y))**(-x*y)
        comb_tune[i]=lambda x,w, p: (1+p)**(-x)
#        gparams["DXI"][i]=lambda x,w,p: -x*(1+p)**(-x-1)
#        
#    elif gparams["xi_func"][i]=="xi_comb_loglin_exp":
#        gparams["N_XI_PARAMS"][i]=3
#        gparams["XI"][i]=lambda x,y,p: (1+p[0])**(-x*p[2])* (1+p[1])**(-(1-p[2])*y)
##        gparams["DXI"][i]=lambda x,w,p: -x*(1+p)**(-x-1)
        
#        N_XI_PARAMS=2
#        XI=lambda x,w,p: (1+p[1]/(1+x)**p[0])/(1+p[1])
    
#        N_XI_PARAMS=3
#        XI=lambda x,w,p: (1.)/(1+p[0]*log10(p[2]*x+1))**p[1]   
       
#
#    elif gparams["xi_func"][i]=="xi_hyp_log3":
#        gparams["N_XI_PARAMS"][i]=3
#        gparams["XI"][i]=lambda x,w,p: (1+p[0]*log10(p[2]+x+1))**-p[1]   
#        
#    elif gparams["xi_func"][i]=="xi_hyp_log4":
#        gparams["N_XI_PARAMS"][i]=3
#        gparams["XI"][i]=lambda x,w,p:  ( (1.)/(1+p[0]*x)+p[2]/(1+p[1]*log10(1+x)))/(1+p[2])
#    
    elif gparams["xi_func"][i]=="xi_hyp1_log5":
        gparams["N_XI_PARAMS"][i]=1
        gparams["XI"][i]=lambda x,w,p: (1.)/(1+p*log1p(x))  
    elif gparams["xi_func"][i]=="xi_hyp_log5":
        gparams["N_XI_PARAMS"][i]=1
        gparams["XI"][i]=lambda x,w,p: (1+log1p(x))**-p
    elif gparams["xi_func"][i]=="xi_hyp_log5_2":
        gparams["N_XI_PARAMS"][i]=2
        gparams["XI"][i]=lambda x,w,p: (1+log1p(x))**-p[0]* (1+log1p(x)**2)**-p[1]
    elif gparams["xi_func"][i]=="xi_hyp1_log5_2":
        gparams["N_XI_PARAMS"][i]=2
        gparams["XI"][i]=lambda x,w,p: (1.)/(1+p[0]*log1p(x)+p[1]*log1p(x)**2)  
#    
#    elif gparams["xi_func"][i]=="xi_power8":
#        gparams["N_XI_PARAMS"][i]=1
#        gparams["XI"][i]=lambda x,w,p: (1+x**-p)**(-1./p)/x        
#          
#    elif gparams["xi_func"][i]=="xi_power9":
#        gparams["N_XI_PARAMS"][i]=2
#        gparams["XI"][i]=lambda x,w,p: 1-(p[0]+x**-p[1])**(-1./p[1])        
#          
#    elif gparams["xi_func"][i]=="xi_power10":
#        gparams["N_XI_PARAMS"][i]=1
#        gparams["XI"][i]=lambda x,w,p: 1-exp(1-p*log(x))**(-1./p) 
          
    elif gparams["xi_func"][i]=="xi_linear":
        gparams["N_XI_PARAMS"][i]=1
        gparams["XI"][i]=lambda x,w,p: 1-x/(1+p) 
#        
    elif gparams["xi_func"][i]=="xi_comb_linear":
        gparams["N_XI_PARAMS"][i]=2
        gparams["XI"][i]=lambda x,y,p: 1-x/(1+p[0]*(1+p[1]*y)) 
        comb_tune[i]=lambda x,w, p: 1-x/(1+p) 
    elif gparams["xi_func"][i]=="xi_linear2":
        gparams["N_XI_PARAMS"][i]=2
        gparams["XI"][i]=lambda x,w,p: 1-x**(1./(p[0]+1))/(1+p[1]) 
#          
    elif gparams["xi_func"][i]=="xi_linear3":
        gparams["N_XI_PARAMS"][i]=1
        gparams["XI"][i]=lambda x,w,p: 1-x*log(1/x+1)/(1+p) 
    elif gparams["xi_func"][i]=="xi_linear4":
        gparams["N_XI_PARAMS"][i]=1
        gparams["XI"][i]=lambda x,w,p: 1-x/log(x+1)/(1+p) 
#    elif gparams["xi_func"][i]=="xi_power13":
#        gparams["N_XI_PARAMS"][i]=1
#        gparams["XI"][i]=lambda x,w,p: tanh(x/(1+p))
    
    elif gparams["xi_func"][i]=="xi_power6":
        gparams["N_XI_PARAMS"][i]=1
        gparams["XI"][i]=lambda x,w,p: 1-(1+x**-p)**(-1./p)        
        
    elif gparams["xi_func"][i]=="xi_comb_power6":
        gparams["N_XI_PARAMS"][i]=2
        gparams["XI"][i]=lambda x,y,p: 1-(1+(x)**-(p[0]*(1+p[1]*y)))**(-1./(p[0]*(1+p[1]*y)))   
        comb_tune[i]=     lambda x,p: 1-(1+(x)**-p)**(-1./p)   
#          
#    elif gparams["xi_func"][i]=="xi_power1":
#        gparams["N_XI_PARAMS"][i]=2
#        gparams["XI"][i]=lambda x,w,p: 1-p[0]*(1+x**-p[1])**(-1./p[1])
#        
#    elif gparams["xi_func"][i]=="xi_power2":
#        gparams["N_XI_PARAMS"][i]=1
#        gparams["XI"][i]=lambda x,w,p: 1-x*(1+p)**(-x)        
#          
#    elif gparams["xi_func"][i]=="xi_power3":
#        gparams["N_XI_PARAMS"][i]=2
#        gparams["XI"][i]=lambda x,w,p: 1-p[0]*x*(1+p[1])**(-x)  
#        
#         
#    if gparams["xi_func"][i]=="xi_power4":
#        gparams["N_XI_PARAMS"][i]=1
#        gparams["XI"][i]=lambda x,w,p: 1-x*(1+x)**-p
#     
#    if gparams["xi_func"][i]=="xi_power5":
#        gparams["N_XI_PARAMS"][i]=2
#        gparams["XI"][i]=lambda x,w,p: 1-p[0]*x*(1+x)**-p[1]
#        
    elif gparams["xi_func"][i]=="xi_power":
        gparams["N_XI_PARAMS"][i]=1
        gparams["XI"][i]=lambda x,w,p: 1-x/(1+p*x)
#        
#    elif gparams["xi_func"][i]=="xi_power7":
#        gparams["N_XI_PARAMS"][i]=2
#        gparams["XI"][i]=lambda x,w,p: 1-p[0]*x/(1+p[1]*x)
        
        
    elif gparams["xi_func"][i]=="xi_exp":
        gparams["N_XI_PARAMS"][i]=1
        gparams["XI"][i]=lambda x,w,p: (1+p)**(-x)
#        gparams["DXI"][i]=lambda x,w,p: -x*(1+p)**(-x-1)
        

#        N_XI_PARAMS=1
#        XI=lambda x,w,p: (1.)/(1+p)**(x)
       
    elif gparams["xi_func"][i]=="xi_exp1":
        gparams["N_XI_PARAMS"][i]=2
        gparams["XI"][i]=lambda x,w,p: exp(0.5*(-x*log(1+p[0]) -p[1]*log(1+x)))
        
#
#        N_XI_PARAMS=2
#        XI=lambda x,w,p:  1./(1+p[0])+(p[0])/(1+p[0])/(1+p[1])**(x)
       
    elif gparams["xi_func"][i]=="xi_exp2":
        gparams["N_XI_PARAMS"][i]=1
        gparams["XI"][i]=lambda x,w,p:  (1+p)**(-x**2)
        
    elif gparams["xi_func"][i]=="xi_exp3":
        gparams["N_XI_PARAMS"][i]=1
        gparams["XI"][i]=lambda x,w,p:  (1+p)**(-x**3)
        
        
    elif gparams["xi_func"][i]=="xi_exp4":
        gparams["N_XI_PARAMS"][i]=2
        gparams["XI"][i]=lambda x,w,p: (1+p[0]*x**p[1])**-(x)
        
    elif gparams["xi_func"][i]=="xi_exp5":
        gparams["N_XI_PARAMS"][i]=3
        gparams["XI"][i]=lambda x,w,p: (1+p[0])**-(p[1]*x**p[2])
        
    elif gparams["xi_func"][i]=="xi_exp6":
        gparams["N_XI_PARAMS"][i]=4
        gparams["XI"][i]=lambda x,w,p: (1+p[0])**-(p[1]*x**p[2]+p[3])
     
    elif gparams["xi_func"][i]=="xi_exp7":
        gparams["N_XI_PARAMS"][i]=2   
        gparams["XI"][i]=lambda x,w,p: 1./(1+p[0]*exp(p[1]*x))
        
    elif gparams["xi_func"][i]=="xi_exp8":
        gparams["N_XI_PARAMS"][i]=3   
        gparams["XI"][i]=lambda x,w,p: 1./(1+p[0]*exp(p[1]*x+p[2]))
        
#        N_XI_PARAMS=2
#        XI=lambda x,w,p: (1.)/(1+p[0]*x)**(p[1]*x)
      
    elif gparams["xi_func"][i]=="xi_exp_log":
        gparams["N_XI_PARAMS"][i]=2
        gparams["XI"][i]=lambda x,w,p: (1+p[0])**-(p[1]*log10(x+1))
        

#        N_XI_PARAMS=2
#        XI=lambda x,w,p: (1.)/(1+p[0])**(p[1]*log10(x+1))
     
    elif gparams["xi_func"][i]=="xi_exp_log1":
        gparams["N_XI_PARAMS"][i]=3
        gparams["XI"][i]=lambda x,w,p: (1+p[0])**-(p[1]*log10(x+1)**p[2])
        

#        N_XI_PARAMS=3
#        XI=lambda x,w,p: (1.)/(1+p[0])**(p[1]*log10(x+1)**p[2])
        
    elif gparams["xi_func"][i]=="xi_exp_log2":
        gparams["N_XI_PARAMS"][i]=2
        gparams["XI"][i]=lambda x,w,p: (1+x)**(-p[0])
        

#        N_XI_PARAMS=2
#        XI=lambda x,w,p: (1+x)**(-p[0])
            
    elif gparams["xi_func"][i]=="xi_exp_log3":
        gparams["N_XI_PARAMS"][i]=2
        gparams["XI"][i]=lambda x,w,p: (1+x)**(-p[0]*log(1+p[1]))
        

#        N_XI_PARAMS=2
#        XI=lambda x,w,p: (1+x)**(-p[0]*log(1+p[1]))
      
     
    elif gparams["xi_func"][i]=="xi_exp_log4":
        gparams["N_XI_PARAMS"][i]=3
        gparams["XI"][i]=lambda x,w,p: (1+p[0])**-(p[1]*log10(x+1)+p[2])
        
    elif gparams["xi_func"][i]=="xi_exp_log5":
        gparams["N_XI_PARAMS"][i]=1
        gparams["XI"][i]=lambda x,w,p: (1+p)**(-log10(x+1))
                                   
    elif gparams["xi_func"][i]=="xi_exp_log6":
        gparams["N_XI_PARAMS"][i]=3
        gparams["XI"][i]=lambda x,w,p: (1+p[0])**-(p[1]*log10(x+1)**p[2])
        
    elif gparams["xi_func"][i]=="xi_exp_log7":
        gparams["N_XI_PARAMS"][i]=4
        gparams["XI"][i]=lambda x,w,p: (1+p[0])**-(p[1]*log10(x+1)**p[2]+p[3])
        
        
#
#        N_XI_PARAMS=3
#        XI=lambda x,w,p: (1.)/(1+p[0])**(p[1]*log10(x+1)+p[2])
                           

#
#        N_XI_PARAMS=1
#        XI=lambda x,w,p: (1.)/(1+p)**(log10(x+1))
        
    elif gparams["xi_func"][i]=="xi_multilog":
        gparams["N_XI_PARAMS"][i]=2
        gparams["XI"][i]=lambda x,w,p:  1./(1+p[0])+(p[0])/(1+p[0])/(1+p[1]*(multilog(x, 2)))
        

#        N_XI_PARAMS=2
#        XI=lambda x,w,p:  1./(1+p[0])+(p[0])/(1+p[0])/(1+p[1]*(multilog(x, 2)))

     
    elif gparams["xi_func"][i]=="xi_log":
        gparams["N_XI_PARAMS"][i]=2
        gparams["XI"][i]=lambda x,w,p: (1.)/(1+p[0]*(log10(x+1)**p[1]))
        

#        N_XI_PARAMS=2
#        XI=lambda x,w,p: (1.)/(1+p[0]*(log10(x+1)**p[1]))

    elif gparams["xi_func"][i]=="xi_log1":
        gparams["N_XI_PARAMS"][i]=1
        gparams["XI"][i]=lambda x,w,p: (1.)/(1+p*(log10(x+1)))
        

#        N_XI_PARAMS=1
#        XI=lambda x,w,p: (1.)/(1+p*(log10(x+1)))

    elif gparams["xi_func"][i]=="xi_log6":
        gparams["N_XI_PARAMS"][i]=2
        gparams["XI"][i]=lambda x,w,p: (1+p[0])/(1+p[0]+p[1]*(log10(x+1)))
        

#        N_XI_PARAMS=2
#        XI=lambda x,w,p: (1+p[0])/(1+p[0]+p[1]*(log10(x+1)))

        
    elif gparams["xi_func"][i]=="xi_log2":
        gparams["N_XI_PARAMS"][i]=2
        gparams["XI"][i]=lambda x,w,p: (1.)/(1+p[0]*(log10(p[1]+x+1)))
        

#        N_XI_PARAMS=2
#        XI=lambda x,w,p: (1.)/(1+p[0]*(log10(p[1]+x+1)))
        
    elif gparams["xi_func"][i]=="xi_log3":
        gparams["N_XI_PARAMS"][i]=2
        gparams["XI"][i]=lambda x,w,p: (1.)/(1+p[0]*(log10(p[1]*x+1)))
      
        

#        N_XI_PARAMS=2
#        XI=lambda x,w,p: (1.)/(1+p[0]*(log10(p[1]*x+1)))
      
    elif gparams["xi_func"][i]=="xi_log4":
        gparams["N_XI_PARAMS"][i]=3
        gparams["XI"][i]=lambda x,w,p: (1.)/(1+p[2]+p[0]*(log10(p[1]*x+1)))
        

#        N_XI_PARAMS=3
#        XI=lambda x,w,p: (1.)/(1+p[2]+p[0]*(log10(p[1]*x+1)))
     
    elif gparams["xi_func"][i]=="xi_log5":
        gparams["N_XI_PARAMS"][i]=3
        gparams["XI"][i]=lambda x,w,p: (1+p[1]+p[0]*(log10(x+1))) **-p[2]
        

    elif gparams["xi_func"][i]=="xi_hyp_exp":
        gparams["N_XI_PARAMS"][i]=3
        gparams["XI"][i]=lambda x,w,p: ( (1.)/(1+p[0]*x)+p[2]*(1+p[1])**-x)/(1+p[2])
        
#        N_XI_PARAMS=3
#        XI=lambda x,w,p: (1.)/(1+p[1]+p[0]*(log10(x+1))) **p[2]

    elif gparams["xi_func"][i]=="xi_mix_hyp":
        gparams["N_XI_PARAMS"][i]=3
        gparams["XI"][i]=lambda x,w,p: 1./(1+p[0])/(1+x[0])**p[1]+ p[0]/(1+p[0])/(1+x[1])**p[2]
        

#        N_XI_PARAMS=3
#        XI=lambda x,w,p: 1./(1+p[0])/(1+x[0])**p[1]+ p[0]/(1+p[0])/(1+x[1])**p[2]
       
    elif gparams["xi_func"][i]=="xi_mix_hyp1":
        gparams["N_XI_PARAMS"][i]=3
        gparams["XI"][i]=lambda x,w,p: 1./(1+p[0]*x[0]+p[1]*x[1])**p[2]
#        
#        N_XI_PARAMS=3
#        XI=lambda x,w,p: 1./(1+p[0]*x[0]+p[1]*x[1])**p[2]

    elif gparams["xi_func"][i]=="xi_mix_hyp2":
        gparams["N_XI_PARAMS"][i]=2
        gparams["XI"][i]=lambda x,w,p: 1./(1+p[0]*x[0])/(1+p[1]*x[1])
        

#        N_XI_PARAMS=2
#        XI=lambda x,w,p: 1./(1+p[0]*x[0])/(1+p[1]*x[1])

    elif gparams["xi_func"][i]=="xi_mix_hyp_log":
        gparams["N_XI_PARAMS"][i]=3
        gparams["XI"][i]=lambda x,w,p: 1./(1+p[0]*log10(x[0]+1)+p[1]*log10(x[1]+1))**p[2]
        
#
#        N_XI_PARAMS=3
#        XI=lambda x,w,p: 1./(1+p[0]*log10(x[0]+1)+p[1]*log10(x[1]+1))**p[2]

    elif gparams["xi_func"][i]=="xi_mix_hyp_log1":
        gparams["N_XI_PARAMS"][i]=3
        gparams["XI"][i]=lambda x,w,p: 1./(1+p[0]*(x[0])+p[1]*log10(x[1]+1))**p[2]
        

#        N_XI_PARAMS=3
#        XI=lambda x,w,p: 1./(1+p[0]*(x[0])+p[1]*log10(x[1]+1))**p[2]
#      
    elif gparams["xi_func"][i]=="xi_mix_hyp_log2":
        gparams["N_XI_PARAMS"][i]=3
        gparams["XI"][i]=lambda x,w,p: 1./(1+p[0]*log10(x[0]+1)+p[1]*(x[1]))**p[2]
        
#        N_XI_PARAMS=3
#        XI=lambda x,w,p: 1./(1+p[0]*log10(x[0]+1)+p[1]*(x[1]))**p[2]
      
        
    elif gparams["xi_func"][i]=="xi_mix_hyp_log3":
        gparams["N_XI_PARAMS"][i]=2
        gparams["XI"][i]=lambda x,w,p: 1./(1+p[0]*log10(x[0]+1)+p[1]*log10(x[1]+1))
        

#        N_XI_PARAMS=2
#        XI=lambda x,w,p: 1./(1+p[0]*log10(x[0]+1)+p[1]*log10(x[1]+1))

    elif gparams["xi_func"][i]=="xi_mix_hyp_log4":
        gparams["N_XI_PARAMS"][i]=2
        gparams["XI"][i]=lambda x,w,p: 1./(1+p[0]*log10(x[0]+1))/(1+p[1]*log10(x[1]+1))
        
    print("XI Scheme for order:", i, ":", gparams["xi_func"][i], file=sys.stderr)
#        N_XI_PARAMS=2
#        XI=lambda x,w,p: 1./(1+p[0]*log10(x[0]+1))/(1+p[1]*log10(x[1]+1))
#XI=lambda x,w,p: (1+p[0]+p[1]*x)**-p[2]  
#XI=lambda x,w,p: (1+p[0]*x)**-p[1] #p[2]
#XI=lambda x,w,p: ((1+p[0])/(1+p[0]+p[1]*x**p[2]))#**p[3] 
#XI=lambda x,w,p: ((1+p[0])/(1+p[0]+p[1]*x**p[2]))



#XI=lambda x,w,p: (1.)/(1+p[0])**x*p[1]
#XI=lambda x,w,p: (1.)/(1+(1+p[1])**(p[0]*x+p[2]))

#XI=lambda x,w,p: (1+p[0])**(-(p[1]*x**p[3]+p[2]))
#XI=lambda x,w,p: exp(-(p[0]*x**p[3]+p[1]))/(1+p[2])
#XI=lambda x,w,p: (1./(1+p[0]))*(1./(1+p[1]))**(p[2]*x**p[3])
#XI=lambda x,w,p: ((1+p[0])/(1+p[0]+p[1]*exp(x)))**p[2] 
#XI=lambda x,w,p: ((1+p[0]+p[1]*exp(-x))/(1+p[0]+p[1]))**p[2] 
#XI=lambda x,w,p: (1-tanh(p[0]*x**p[3]+p[1]))**p[2] 
#sech=lambda x: 2/(exp(x)+exp(-x))
#XI=lambda x,w,p: ((1+p[0]+p[1]*sech(x))/(1+p[0]+p[1]))**p[2]

#P=.001

def split_ngram(ng):
    try:
        s, t=ng.rsplit(None, 1)
    except ValueError:
        return "", ng
    return s, t
    
def get_data_weights(data, weights):
    if not isinstance(data, ndarray) and not  isinstance(data, list):
        return weights["w"][weights["x"]==data]
    res=empty_like(data)
    for i, d in enumerate(data):
        res[i]=weights["w"][weights["x"]==d]
    return res

#options.regress_unseen=True
#if True: #not options.print_ranges_only:
def mk_data_xy(ddict, order, use_loo):
    if order==1:
        if use_loo:
            dx=[c-1 for c in ddict[order].values() if c>1]
        else:
            dx=[c for c in ddict[order].values()  if c>0]
        return dx, [float(len(dx))/sum(dx)]
    ctxt_u={}
    dx, dy=[], []
    for ng, c in ddict[order].items():
        if use_loo and c<=1:
            continue
        if use_loo:
            c-=1
        dx.append(c)
        ctxt, _=ng.rsplit(None, 1)
        try:
            item=ctxt_u[ctxt]
            item[0]+=1
            item[1]+=c
        except KeyError:
            ctxt_u[ctxt]=[1, c]
    
    for ng, c in ddict[order].items():
        if use_loo and c<=1:
            continue        
        ctxt, _=ng.rsplit(None, 1)
        item=ctxt_u[ctxt]
        dy.append(float(item[0])/item[1])
    return dx, dy
            
seterr(all="raise")

for i in range(1, options.order+1):
    if i in options.nodiscount:
        continue
    print("\n", file=sys.stderr)
#        dfuns[i], Dd[i]=mk_discount_fun(od[i].values(), i, options.discount_func, params=options.rpfile)
    ind=None
    if i>1 and options.mle_ctxt is not None and options.mle_ctxt[i]:
        print("Likelihood value for different contexts will be maximized together with the data, for order", i, file=sys.stderr)
        data,ind=mkdata_array(od,i)
    else:
        print("Likelihood value will be maximized for all data points only (no context values are involved), for order", i, file=sys.stderr)
        data=fromiter(iter_nz_counts(od[i]), dtype=float)

    counts, indexes=unique(data, return_inverse=True)     
    counts_counts=bincount(indexes)
    del indexes
    if options.n_tune_counts is not None and options.n_tune_counts[i]:
        counts=counts[:options.n_tune_counts[i]]
        counts_counts=counts_counts[:options.n_tune_counts[i]]
#    highest_count=5
#    counts=counts[:highest_count]
##    counts_counts[highest_count-1]+=sum(counts_counts[highest_count:])
#    counts_counts=counts_counts[:highest_count]
    
    minval, maxval=amin(data), amax(data)
    dfuns[i]={}
    if ind is not None:
        ind=array(ind)
#        for m in UNSEEN_ESTIMATORS:
#            estimate_unseen(data, method=m)

    unseen_mass=None
    if options.estimate_unseen is not None:
        if options.estimate_unseen[i] is not None:
            try:
                unseen_mass=estimate_unseen(data, indexes=ind , method=options.estimate_unseen[i])
                print("Estimated unseen number of ngrams for order",i,":", unseen_mass, file=sys.stderr)
                p0=unseen_mass/(unseen_mass+sum(counts*counts_counts))
            except UnknownEstimatorError:
                print("Unkown estimation method,  falling back to leave-one-out", file=sys.stderr)
#                raise
                unseen_mass=None
    if unseen_mass is None:
        p0=counts_counts[0]*counts[0]*1.0/(counts[0]*counts_counts[0]+sum((counts[1:]-counts[0])*counts_counts[1:]))
    
    if False: #i==2: #>1:
        print("Unseen proportion:", p0, file=sys.stderr)
        print("Computing associations...", file=sys.stderr)
        
        tune_xi_local=lambda data,  xif, N_XI_PARAMS, indexes=None, W0=None, xi_arg="kn", xi_arg_is_count_dist=False:tune_xi_w0(data[0],   xif, N_XI_PARAMS, indexes, W0, xi_arg, xi_arg_is_count_dist)

#        ########
#        if options.penalty [i] is not None:
##            st=time.time()
#            print("\nLoading %d-grams penalties from '%s'..."%(i, options.penalty [i]), file=sys.stderr)
#            n_pen=0;
#            for line in open_infile(options.penalty [i]):
#                ng, s=line.rsplit(None, 1)
#                
#                s=float(s)
#                if s>=0:
#                    continue
#                    
#                c=od[i][ng]
#                if c<=2:
#                    continue
#                
#                n_pen+=1             
#                od[i][ng]=c/(1-s)  
#            print("Penalized n_grams:", n_pen, file=sys.stderr)
#                
#        else:
#            out_d, _, _, smax=compute_assoc(od[i],gparams["method"][i], split_ngram, out={}) #, scale=options.assoc_scales[i])
#            if True:
#                n_pen=0;
#                print("Penalizing counts...", file=sys.stderr)
#                for ng, c in od[i].items():
#                    s=out_d[ng]
#                    if s>=0 or c<=2:
#                        continue
#                    n_pen+=1
#                    if s<0:
#                        s=1./(1-s)
#                    else:
#                        s=s/(1+s)
#                        
#                    od[i][ng]*=s
#                print("Penalized n_grams:", n_pen, file=sys.stderr)
#            else:
#                print("Assiging associations...", file=sys.stderr)
#                for ng, c in od[i].items():
#                    s=out_d[ng]
#                    if s<0:
#                        s=-1./(s)
#                    od[i][ng]=s
        unseen_mass=__builtins__.sum(od[i].values())*p0/(1-p0)
        print("New unseen estimation:", unseen_mass, file=sys.stderr)
        data=fromiter(iter_nz_counts(od[i]), dtype=float)
        tune_data=(data, )
    else:
        tune_data=(counts, counts_counts)
        
        tune_xi_local=lambda data,  xif, N_XI_PARAMS, indexes=None, W0=None, xi_arg="kn", xi_arg_is_count_dist=False:tune_xi(data[0], data[1],   xif, N_XI_PARAMS, indexes, W0, xi_arg, xi_arg_is_count_dist)
##        if options.regress_unseen:
##            print >>sys.stderr, "Estimating the unseen mass through regression with repeated leaving-one-out"            
##                        
##            NTRIALS=20
##            idata=data.astype(int)
##            MAX_COUNTS=4
##            N_ARGS=5
##            par=ones((NTRIALS, N_ARGS+1))
##            
##            print >>sys.stderr,"Number of trials:", NTRIALS
##            
##        
##            def gety(p,  args):
##                return sum(p*args, axis=1)
##                return -p[0]*log(abs(log(su)-p[2])) #-p[1]
##                    
##            def err(p, args, y):
###                if any(p<0):                
###                    return 1e100
##                return sum( (gety(p, args)-(y))**2)
##                return sum((getu(p, par[:,1])-log(par[:,0]))**2)                
##                
###            prev=   sum(data>=1)
##            mask=idata>0
##            for l in range(NTRIALS):      
##                mask=idata-l>0         
##                
##                if not any(idata[mask]-l==1):
##                    continue
###                print >>sys.stderr,"Computing trial %d"%(l)
##                _, ind=unique(idata[mask]-l-1, return_inverse=True)                 
##                par[l, :MAX_COUNTS+1]=bincount(ind)[:MAX_COUNTS+1]
###                
###                par[l, 1]=sum(data-l-1>=1)
###                par[l, 0]=float(par[l, 1])/prev
###                prev=par[l, 1]
##            savetxt("reg.%d"%(i), par, fmt="%d")
##            
##            p0=ones(N_ARGS)
##            bp, mine, _, _, _, _=fmin_powell(err, p0, args=(par[:, 1:], par[:, 0 ]), maxiter=1000,maxfun=1000, full_output=True, disp=0)
##            print >>sys.stderr,"Best regressed parameters:", bp, "Error=", mine   
##            
##           
##            _, ind=unique(idata, return_inverse=True)           
##            arg4all= ones(N_ARGS)
##            arg4all[:MAX_COUNTS]=bincount(ind)[:MAX_COUNTS]
###            useen=len(data)
###            unseen_mass=(1/exp(getu(bp, useen )) -1 )* useen
##            
##            unseen_mass=gety(bp, arg4all.reshape((-1, N_ARGS)))
##            print >>sys.stderr,"Estimated unseen number of ngrams for order",i,":", unseen_mass, "("+("; ".join(("N%d = %d"%(n+1, c) for n, c in enumerate(arg4all))))+")" #, "Unique seen=", useen
    
#    options.xi_arg_is_count_dist=True
    if gparams["xi_func"][i] .startswith("xi_comb"):
        print("Tuning combined XI...", file=sys.stderr)
        
#        dx, dy=mk_data_xy(od, i, unseen_mass is None)
#        p_wb=((float(len(data))/sum(data.astype(int)==int(minval))-1)**-1) 
        olderr=seterr(all="raise") 
        stune_t=time.time()
#        if gparams["xi_func"][i] .endswith("_sep"):
#        dfuns[i]["xi_p"]=tune_comb_xi_sep(dx, dy, gparams["XI"][i], gparams["N_XI_PARAMS"][i], len(od[i])-len(dx) if unseen_mass is None else unseen_mass, unseen_mass is None) 
#        else:
#            dfuns[i]["xi_p"]=tune_comb_xi(dx, dy, comb_tune[i], gparams["N_XI_PARAMS"][i], len(od[i])-len(dx) if unseen_mass is None else unseen_mass, unseen_mass is None) 
        ################
        
        p_kn=tune_xi(counts, counts_counts, comb_tune[i], gparams["N_XI_PARAMS"][i]/2, indexes=ind, W0=unseen_mass, xi_arg=options.smoothing, xi_arg_is_count_dist=(options.xi_arg_is_count_dist is not None and options.xi_arg_is_count_dist[i]) )
        print("Determining the paramter of the XI function", file=sys.stderr)
        if options.original_wb is not None and options.original_wb[i]:
            print("Witten-Bell default parameter", file=sys.stderr)
            p_wb=1.
        elif unseen_mass is None:
            print("Paramter estimated through Leaving-One-Out", file=sys.stderr)
            p_wb=((float(len(data))/sum(data.astype(int)==int(minval))-1)**-1) #(float(len(data))/sum(data.astype(int)==int(minval)))**-1
        else:
            p_wb=unseen_mass/(len(data))
        dfuns[i]["xi_p"]=[p_kn/(1+(p_wb*len(data))/sum(data) ), p_wb]
        ###############
        etune_t=time.time()
        seterr(**olderr)
        print("Tuning XI took:", formatted_span(etune_t-stune_t), file=sys.stderr)
        
    elif options.smoothing=="kn" or options.smoothing=="wb-mle":
        olderr=seterr(all="raise") 
        stune_t=time.time()
            
        if gparams["xi_func"][i] != "xi_kn":
#            dfuns[i]["xi_p"]=tune_xi(counts, counts_counts, gparams["XI"][i], gparams["N_XI_PARAMS"][i], indexes=ind, W0=unseen_mass, xi_arg=options.smoothing, xi_arg_is_count_dist=(options.xi_arg_is_count_dist is not None and options.xi_arg_is_count_dist[i]) )
            dfuns[i]["xi_p"]=tune_xi_local(tune_data, gparams["XI"][i], gparams["N_XI_PARAMS"][i], indexes=ind, W0=unseen_mass, xi_arg=options.smoothing, xi_arg_is_count_dist=(options.xi_arg_is_count_dist is not None and options.xi_arg_is_count_dist[i]) )
        else:
#            print >>sys.stderr, "Using KN discounts... Number of constants:", gparams["N_XI_PARAMS"][i]
            dfuns[i]["xi_p"]=tune_xi_kn(counts_counts, gparams["N_XI_PARAMS"][i])     
            
        etune_t=time.time()
        seterr(**olderr)
        print("Tuning XI took:", formatted_span(etune_t-stune_t), file=sys.stderr)
    elif options.smoothing=="wb":
        print("Determining the paramter of the XI function", file=sys.stderr)
        if options.original_wb is not None and options.original_wb[i]:
            print("Witten-Bell default parameter", file=sys.stderr)
            dfuns[i]["xi_p"]=1.
        elif unseen_mass is None:
            print("Paramter estimated through Leaving-One-Out", file=sys.stderr)
            dfuns[i]["xi_p"]=((float(len(data))/sum(data.astype(int)==int(minval))-1)**-1) #(float(len(data))/sum(data.astype(int)==int(minval)))**-1
        else:
            dfuns[i]["xi_p"]=unseen_mass/(len(data))
    
        
    
    print("Best XI params for order %d:"%(i), dfuns[i]["xi_p"], file=sys.stderr)
    if gparams["xi_func"][i].startswith("xi_comb"):
        dfuns[i]["xi"]=lambda x, y: gparams["XI"][i]((1.*x)**-1, y, dfuns[i]["xi_p"])
    elif options.smoothing=="kn" :
           
        if gparams["xi_func"][i] != "xi_kn":
            
            dfuns[i]["tr"]=lambda x: (1.*x)**-1
            if options.xi_arg_is_count_dist is not None and options.xi_arg_is_count_dist[i]:            
#                x0=0.5
                slog=sum(counts_counts*(log(counts)))
                scc=sum(counts_counts)
                dfuns[i]["tr_p"]=1+scc/(slog-scc*log(x0))
                dfuns[i]["tr"]=lambda x: (dfuns[i]["tr_p"]-1)*(x/x0)**-dfuns[i]["tr_p"]/x0
                print("First 10 XI values:", gparams["XI"][i](dfuns[i]["tr"](counts), None, dfuns[i]["xi_p"]) [:10], file=sys.stderr)
            dfuns[i]["xi"]=lambda x: gparams["XI"][i](dfuns[i]["tr"](x), None, dfuns[i]["xi_p"]) #, 1-HIGHEST_XI, HIGHEST_XI)
#            dfuns[i]["xi"]=lambda x: gparams["XI"][i]((1.*clip(x, 1, None))**-1, None, dfuns[i]["xi_p"]) #, 1-HIGHEST_XI, HIGHEST_XI)
        else:
            
            dfuns[i]["xi"]=lambda x:  gparams["XI"][i]((x), None, dfuns[i]["xi_p"])
#            dfuns[i]["xi"]=lambda x:  gparams["XI"][i]((clip(x, 1, None)), None, dfuns[i]["xi_p"])
            
        
        dfuns[i]["zeromass"]=sum((1-dfuns[i]["xi"](data))*data) 

            
        dfuns[i]["func"]=lambda x: x*(1-dfuns[i]["xi"](x))  if x>0 else 0
        
#        estimate_multiplicative_discount(dfuns, i, minval, xi_depends_on_x=False)(float(x)) if x>0 else 0
#    acounts=array(od[i].values())
#    Dd[i]= compute_smoothing_attributes(acounts)
#    dfuns[i]=discount_params(array(Dd[i]), arange(1,HIGHEST_POSITIVE_BIN,1.))
        
        
        print("Mass collected for 0:%g"%(dfuns[i]["zeromass"]), file=sys.stderr)         
#        print >>sys.stderr, "Process",rank,"Value of the new distribution at %g:%g"%(minval,dfuns[i]["xi"](minval)*dfuns[i]["tail_dist"](minval))        
#        print >>sys.stderr, "Weibull parameters for order %d: %s"%(i, fitweibull4lower(xi(minval)*dfuns[i]["tail_dist"](minval), zeromass, minval))
        
#        print >>sys.stderr,"Alpha= %g, XMIN=%g, C=%g"%(alpha, xmin, (alpha-1)*xmin**(alpha-1))
#        print >>sys.stderr,"Process",rank,"DIST(%g)= %g, DIST(%g)= %g"%(minval, dfuns[i]["tail_dist"](minval), minval+(maxval-minval)/10, dfuns[i]["tail_dist"](minval+(maxval-minval)/10))
        print("XI(%g)= %g (^-1= %g)"%(minval, dfuns[i]["xi"](minval), 1./dfuns[i]["xi"](minval)), "XI(%g)= %g (^-1= %g)"%(maxval, dfuns[i]["xi"](maxval), 1./dfuns[i]["xi"](maxval)), file=sys.stderr)
#        print >>sys.stderr,"Process",rank,"New densitiy value: %g"%(dfuns[i]["tail_dist"](minval)* dfuns[i]["xi"](minval))
        
        print("Smallest X*(%g)= %g (D= %g)"%(minval, minval-dfuns[i]["func"](minval), dfuns[i]["func"](minval)), end=' ', file=sys.stderr)
        print("Largest X*(%g)= %g (D= %g)"%(maxval, maxval-dfuns[i]["func"](maxval), dfuns[i]["func"](maxval)), file=sys.stderr)
    
    elif options.smoothing.startswith("wb"):
            N_RND_POINTS=1000
#            dfuns[i]["xi"]=lambda x: clip(gparams["XI"][i]((1.*x), None, dfuns[i]["xi_p"]), 1-HIGHEST_XI, HIGHEST_XI)
            if gparams["xi_func"][i] !="xi_hyp1":
                print("Re-estimating the parameters for the function '%s' through regression from %d randomly generated points"%(gparams["xi_func"][i], N_RND_POINTS), file=sys.stderr)
                r=random.random_sample(N_RND_POINTS)
                d=1./(1.+dfuns[i]["xi_p"]*r)
#                print >>sys.stderr, "Values generated:", r[:10]
#                print >>sys.stderr, "Their corresponding xi vals:", d[:10]
                def f(p):
#                    if any(p<=0):
#                        return inf
                    return 1-d/gparams["XI"][i](r, None, exp(p))
#                    return 1-d/gparams["XI"][i](r, None, (p**2))
#                    return sum((1-d/gparams["XI"][i](r, None, p))**2)
#                df=None
#                if gparams["DXI"][i] is not None:
#                    def df(p):
#                        if any(p<=0):
#                            return inf
#                        return -gparams["DXI"][i](r, None, p)/d
#                
                seterr(under="ignore")
#                dfuns[i]["xi_p"], _=leastsq(f, ones(gparams["N_XI_PARAMS"][i])) #, Dfun=df)
#                pp, _=leastsq(f, ones(gparams["N_XI_PARAMS"][i]))
                dfuns[i]["xi_p"],  mine, _, _, _, _= fmin_powell(lambda p: sum(absolute(f(p))),ones(gparams["N_XI_PARAMS"][i]), maxiter=1000,maxfun=1000, full_output=True, disp=0)
                dfuns[i]["xi_p"]=exp(dfuns[i]["xi_p"])
#                dfuns[i]["xi_p"]=(dfuns[i]["xi_p"])**2
                print("Best re-esimated XI params for order %d:"%(i), dfuns[i]["xi_p"], file=sys.stderr) #, "e^x=", exp(dfuns[i]["xi_p"])
#                print >>sys.stderr, "Error:",  sum(f(dfuns[i]["xi_p"])**2)
                print("Error:",  mine, file=sys.stderr)
#                print >>sys.stderr, "Best XI params for order %d (Without Jacobian):"%(i), pp
#                print >>sys.stderr, "Value:",  sum(f(pp)**2)
                
#                print >>sys.stderr, "Estimated d=", gparams["XI"][i](r, None, dfuns[i]["xi_p"])[:10]
            dfuns[i]["xi"]=lambda x: gparams["XI"][i]((1.*x), None, dfuns[i]["xi_p"])
#                print >>sys.stderr, "Witten-Bell for functions other than Hyp1 is not yet implemented"
#                raise
        
#        dfuns[i]["xi"]=lambda x: gparams["XI"][i]((1.*x), None, dfuns[i]["xi_p"]) 
        #        out=file("dist."+str(i), "wb")
#        print >>out, "0.0\t%g\t0.0"%(zeromass)
#        for v in ud:
#            print >>out, "%g\t%g\t%g"%(v, dfuns[i]["tail_dist"](v)*xi(v), dfuns[i]["tail_dist"](v))
#        out.close()
print("D=",Dd, file=sys.stderr)
#print >>sys.stderr, "DFUNS=", dfuns
#sys.exit(0)

print("Original number of n-grams:", file=sys.stderr)
#probs=compute_probs(od, 1)
for i in od: #range(1, options.order+1):
   
    print("\tAttributes of %d-grams:"%(i), file=sys.stderr)
    print("\t\t%d-grams: %d"%(i, len(od[i])), file=sys.stderr) #cd[i+1])
#    print >>sys.stderr,"\t\t", 
#
#for ng in probs:
#    print log10(probs[ng]) if probs[ng] else -99, ng

#acounts=array(od[ord].values())
#D= compute_smoothing_attributes(acounts, ord)
for i in range(1, options.order+1):
    if i in options.nodiscount  or options.smoothing.startswith("wb") or gparams["xi_func"][i] .startswith("xi_comb"):
        continue
#    s, b=min(x for x in od[i].values() if x>0), max(od[i].values())
    first10= array([x for x in sorted(set(iter_nz_counts(od[i])))[:20]])
    print("\n", file=sys.stderr)
    print("First 10 associations for order %i: [%s]" %(i,", ".join("%g"%(x) for x in first10)), file=sys.stderr)
    firstdis=list(map(dfuns[i]["func"], first10))
#    savetxt(gparams["xi_func"][i]+".%d.first20"%(i), firstdis)

#    print >>sys.stderr,"==>Their distribution values: [%s]" %(", ".join("%g"%(x) for x in dfuns[i]["tail_dist"]( first10)))        
#    print >>sys.stderr,"==>Their discounts: [%s]" %(", ".join("%g"%(x) for x in dfuns[i]["xi"](first10)))
    print("==>Their corresponding discounted associations: [%s]" %(", ".join("%g"%(x) for x in firstdis)), file=sys.stderr)
    
    print("==>Their final discounted associations: [%s]" %(", ".join("%g"%(x-y) for x, y in zip(first10, firstdis))), file=sys.stderr)
                                                                                      
#    print >>sys.stderr,"Largest value for order %d: %g" %(i, b)
#sys.exit(0)

#if options.print_ranges_only:
#    sys.exit(0)
## If we start the program for testing different bandwidths, we just stay here waiting for inputs

#RATE=MIN_MAXVAL/min(max(od[o].values()) for o in od) #assigns[rank][0])
#bis={1:46700.1579428, 2:861.306419154, 3:103.364689755, 4:39.7022722343}
SIGMAP={1:7, 2:1., 3:1,4:1}
#RATE=0.009
def ll_ll2(p, df, dfn): #, rego=None, regp=None):
    try:
        wf=sum(p*df,axis=1)
#        ewf=exp(wf)

        wfn=sum(p*dfn,axis=1)
        
        pr=wf/(wfn)
        return -sum(pr*log(pr))
#        dp=df[:, 0]
        ll0=sum(dp*(log(wf)-log(wfn)))
        return ll0
    except FloatingPointError:
        return -1e100
 
negll_ll2=lambda p, d, dn:-ll_ll2(p, d, dn)   

def ent1(p, df):
    try:
        if any(p<0) or any(p>1):
            return -1e100
        p/=sum(p)
        pr=sum(p*df,axis=1) #/sum(p)
        pr0=1-sum(pr)
        if pr0>0:
            e0=-(pr0*log(pr0))
        else:
            e0=0
            
        er=-sum(pr*log(pr)) 
#        print >> sys.stderr, "p=",p,"Normalized p=",p/sum(p),"p0=",pr0,"e_0=", e0, "e_r=", er
        return  e0+er

    except FloatingPointError:
        return -1e100
 
ALPH=2 

def ent(p0, df):
    try:
        p=p0
        if any(p<0) or any(p>1):
            return -1e100
#        print >> sys.stderr, "p=",p0
#        p=array((1-p0, p0)).reshape((2, ))
        p/=sum(p)
        probs=sum(p*df, axis=1)
        e=-sum(probs*ma.log(probs))
#        e=1./(1-ALPH)*log(sum(probs**ALPH))
#        print >> sys.stderr, "p=",p,"Normalized p=",p/sum(p),"p0=",pr0,"e_0=", e0, "e_r=", er
        return  e

    except FloatingPointError:
        return -1e100
        
negent=lambda p, d:-ent(p, d)

def ent2(p0, df):
    try:
        p=p0
        if any(p<0) or any(p>1):
            return -1e100
#        print >> sys.stderr, "p=",p0
#        p=array((1-p0, p0)).reshape((2, ))
        p/=sum(p)
        
        q=df*p #/sum_qdf
        q/=sum(q, axis=1)[:, newaxis]
        
        qdf=clip(q*df, 1e-30, inf) #q*nz_df 
        qdf/=sum(qdf, axis=0)
        
        probs=sum(p*qdf, axis=1)
        e=-sum(probs*ma.log(probs))
#        e=1./(1-ALPH)*log(sum(probs**ALPH))
#        print >> sys.stderr, "p=",p,"Normalized p=",p/sum(p),"p0=",pr0,"e_0=", e0, "e_r=", er
        return  e

    except FloatingPointError:
        return -1e100
        
negent2=lambda p, d:-ent(p, d)
def get_zmass(probs, xi):
    locxi=xi
    if xi.shape != probs.shape:
        locxi=xi[:, newaxis]
    return 1-sum(probs*locxi, axis=0)
    
def entropy(probs):
    
    return -sum(probs*ma.log(probs), axis=0)
    
    
def dentropy(probs, xi):
    locxi=xi
    if xi.shape != probs.shape:
        locxi=xi[:, newaxis]
    
    dprobs=probs*locxi
    p0=1-sum(dprobs, axis=0)
    
    return -sum(dprobs*ma.log(dprobs), axis=0)-p0*ma.log(p0)

def pdentropy(p, probs, xi):
    
    if any(p<0) or any(p>1):
        return -1e100
        
    p/=sum(p)
    pbs=sum(probs*p, axis=1)
   
    
    dprobs=pbs*xi
    p0=1-sum(dprobs, axis=0)
    
    return -sum(dprobs*ma.log(dprobs), axis=0)-p0*ma.log(p0)

def d_ll(probs, weights):
    w=weights
    if probs.shape!=weights.shape:
        w=weights[:, newaxis]
    return sum(w*ma.log(probs), axis=0)
    
    return -sum(dprobs*ma.log(dprobs), axis=0)-p0*ma.log(p0)

def comp_entropy(p, probs, xi):
    
    if any(p<0) or any(p>1):
        return -1e100
        
    p/=sum(p)
    q=probs*p #/sum_qdf
    q/=sum(q, axis=1)[:, newaxis]
    pp= mean(q, axis=0)
    qdf=q*probs #nz_df  #clip(q*nz_df , 1e-70, inf) #q*nz_df 
    qdf/=sum(qdf, axis=0)
    return pdentropy(pp,qdf,xi)
                
negpdent=lambda p, pr, xi:-pdentropy(p, pr, xi)
negcent=lambda p, pr, xi:-comp_entropy(p, pr, xi)

def ll_ll(p, df, zf=None): #, rego=None, regp=None):
    try:
        if any(p<0) :
            return -1e100
        p/=sum(p)
#        wf
        pr=sum(p*df,axis=1)
        if zf is not None:
            wzf=sum(p*zf,axis=1)
#        ewf=exp(wf)
#        z=sum((wf))
#        z=sum((p))
        
#        pr=wf #/z #sum(ewf)
        if zf is not None:
            pr0=1-wzf #/z
           
            e0=-sum(pr0*log(pr0))
            er=-sum(pr*log(pr)) 
            print("Normalized p=",p,"e_0=", e0, "e_r=", er, "Ent=", e0+er, file=sys.stderr)
            return  e0+er
        
        return -sum(pr*log(pr))
        dp=df[:, 0]
#        dp=wf #sum(df, axis=1)
        
        ll0=sum(dp*log(wf))
        norm=-sum(dp)*log(z)
        
#        print >> sys.stderr, "Features x ",p,"=", p*sum(df,axis=0), "Likelihood=",  ll0+norm
#        reg=0
#        if (rego and regp) is not None:
#            reg=-regp*sum(abs(p)**rego)
        return ll0+norm #+reg
    except FloatingPointError:
        return -1e100
 
negll_ll=lambda p, d, c=None:-ll_ll(p, d, c)   

freqd=None

def obj_grad_e(lmbd,X,sig=-1.0):    
    Q=exp(clip(lmbd, -30, 30)).reshape(X.shape)
    NQ=Q/sum(Q,axis=1).reshape((-1,1)) #[:,newaxis]
    s=sum(NQ*X)
    sw=sum(NQ*X,axis=1)[:,newaxis] #.reshape((-1,1))
    
    obj_val=sig*(-sum(sw*log(sw))/s + log(s))
    grad_val=sig*(NQ*(X-sw)/s)*(sum(sw*log(sw))/s - log(sw))
    return obj_val,grad_val.reshape(lmbd.shape)

def llk_obj_grad(q, X, sign=-1.0):
    w=q.reshape(X.shape)
    sw=sum(w, axis=1)[:, newaxis]
    w/=sw
    probs=sum(w*X, axis=1)[:, newaxis]
    s=sum(probs)
    obj_val=sum(log(probs/s))
    
    grad_val=(X-probs)*(1/probs-len(X)/s)/sw
    return obj_val, grad_val.reshape(q.shape)


def meanstd(seq):
#    from math import sqrt
    sumx, sumx2, lseq=reduce(lambda x,y:(x[0]+y,x[1]+y*y, x[2]+1),seq,(0,0,0))
    mean=sumx*1./lseq
    return mean, sqrt((sumx2 / lseq) - (mean * mean)) 
    

penalties=None
if False:#options.penalty is not None:
    penalties={}
    if True: #for i in assigns[rank][0]:
        if True : #options.penalty [i] is not None:
            if True:
                for i in assigns[rank][0]:
                    if options.penalty [i] is not None:
                        penalties[i]={}
                        st=time.time()
                        print("\nLoading %d-grams penalties from '%s'..."%(i, options.penalty [i]), file=sys.stderr)
                        for line in open_infile(options.penalty [i]):
                            k, p=line.rsplit(None, 1)
                            penalties[i][k]=float(p)
                         
                        
                        print("Loading %d-penalties took", formatted_span(time.time()-st), file=sys.stderr) 
            elif True:                
                for i in assigns[rank][0]:
                    if options.penalty [i] is not None:
#                        penalties[i]={}
                        st=time.time()
                        print("\nPenalizing %d-grams from '%s'..."%(i, options.penalty [i]), file=sys.stderr)
                        for line in open_infile(options.penalty [i]):
                            k, p=line.rsplit(None, 1)
#                            penalties[i][k]=float(p)
                            try:
                                od[i][k]*=float(p)
                            except KeyError:
                                pass
                         
            else:
#                if freqd is None:
#                    freqd=ddict(None, {}  )
#                freqd[i]=fromiter(od[i].values(),dtype=int)       
                                
                def max_mix(seq, weights):
                    return __builtins__.max(seq)
                    
                def min_mix(seq, weights):
                    return __builtins__.min(seq)

                def loglin_mix(seq, weights):
                    return exp(__builtins__.sum(w*log(p) for p, w in zip(seq, weights)))
                    
                def lin_mix(seq, weights):
                    return sum(w*p for p, w in zip(seq, weights))
                  
                def harmlin_mix(seq, weights):
                    return 1./__builtins__.sum(w/p for p, w in zip(seq, weights))
                    
                if options.mix_func is None:
                    options.mix_func=ddict("linear",{})
                if options.mix_weights is None:
                    options.mix_weights=ddict([0.5]* 2, {})
                print("MIXING WEIGHTS=", options.mix_weights, file=sys.stderr)
#                print >>sys.stderr, "MIXING PARAMS=", gparams["mix-weights"]
                
                print("", file=sys.stderr)
                READAPT_XI=False
                for i in assigns[rank][0]:
                    st=time.time()

                    if options.penalty [i] is not None:
                        print("\nMixing distributions for order: %d;"%(i), end=' ', file=sys.stderr) 
                        
                        print("using",options.mix_func[i],"mixing, with weights:", options.mix_weights[i], file=sys.stderr)
                        
                        if  options.mix_func[i]== "linear":
                            options.mix_func[i] =lin_mix
                        elif  options.mix_func[i]== "log-linear":
                            options.mix_func[i] =loglin_mix
                        elif options.mix_func[i] == "harmonic-linear":
                            options.mix_func[i] =harmlin_mix
                        elif options.mix_func[i] == "max":
                            options.mix_func[i] =max_mix
                        elif options.mix_func[i] == "min":
                            options.mix_func[i]=min_mix
                        else:
                            raise NameError("Unknown mixing function: '%s'"%(options.mix_func[i]))
                        
                        st=time.time()
                        ctxt={}                
                        print("\nMixing %d-grams using '%s'..."%(i, options.penalty [i]), file=sys.stderr)
                        print("Collecting contextual counts...", file=sys.stderr)
                        for ng, s in od[i].items():
                            ctx=split_ngram(ng)[0]                    
                            try:
                                ctxt[ctx][0]+=s
                            except KeyError:
                                ctxt[ctx]=[s, 0]
                            
                        
#                        save_scores=open_outfile("scores.before.3.%d"%(i))
        #                ctxt2={}
                        print("Computing mix scores...", file=sys.stderr)
                        
                        if READAPT_XI:
                            xi_vals=dfuns[i]["xi"](fromiter(iter(od[i].values()), dtype=float)) #, None, dfuns[i]["xi_p"])
                        for k in od[i]:       
                            ctx=split_ngram(k)[0]           
                            od[i][k]/=float(ctxt[ctx][0])
                        for line in open_infile(options.penalty [i]):
                            k, p=line.rsplit(None, 1)
                            try:
                                ctx=split_ngram(k)[0]
                                olds=od[i][k]
                                news=options.mix_func[i]((float(p), olds), options.mix_weights[i])
                                ctxt[ctx][1]+=news-olds
        #                        ctxt2[ctx]=ctxt2.get(ctx, 0)+news     
#                                print >>save_scores, k, od[i][k], float(od[i][k])/ctxt[ctx][0], p, news  
                                od[i][k]=news
                            except KeyError:
                                
                                print("Missing %d-gram '%s' in the model"%(i, k), file=sys.stderr)
                                
#                        for k, s in od[i].iteritems():       
#                            ctx=split_ngram(k)[0]   
#                            ctxt[ctx][1]+=s
                        print("Renormalizing...", file=sys.stderr)
                        for ctx, (csum, ssum) in ctxt.items():
                            ctxt[ctx]=float(csum)/(1+ssum)
                        
#                        save_scores=open_outfile("scores.after.3.%d"%(i))
                        for k, s in od[i].items():                    
                            ctx=split_ngram(k)[0]           
#                            print >>save_scores, k, od[i][k], od[i][k]*ctxt[ctx]
                            od[i][k]*=ctxt[ctx]
                        if READAPT_XI:
                            print("Readapting XI parameters to the new scores...", file=sys.stderr)
                            #################################
                            scores=fromiter(iter(od[i].values()), dtype=float)
                            
                            def f(p):
                                dfuns[i]["xi_p"]=p**2
#                                print >>sys.stderr, "first 10 scores", scores[:10]
#                                print >>sys.stderr, "Their corresponding xi values:", dfuns[i]["xi"](scores)[:10]
                                
#                                print >>sys.stderr, "Evaluating p=", p, "Return value:", sum(absolute(xi_vals-dfuns[i]["xi"](scores)))
#                                return xi_vals-dfuns[i]["xi"](scores) #gparams["XI"][i](scores, None, )
                                return 1-dfuns[i]["xi"](scores)/xi_vals #gparams["XI"][i](scores, None, )
#                    return sum((1-d/gparams["XI"][i](r, None, p))**2)
#                df=None
#                if gparams["DXI"][i] is not None:
#                    def df(p):
#                        if any(p<=0):
#                            return inf
#                        return -gparams["DXI"][i](r, None, p)/d
#                
                            seterr(under="ignore")
#                dfuns[i]["xi_p"], _=leastsq(f, ones(gparams["N_XI_PARAMS"][i])) #, Dfun=df)
#                pp, _=leastsq(f, ones(gparams["N_XI_PARAMS"][i]))
                            print("XI parameter(s) changed from:", dfuns[i]["xi_p"], end=' ', file=sys.stderr) 
                            dfuns[i]["xi_p"],  mine, _, _, _, _= fmin_powell(lambda p: sum(absolute(f(p))),ones(gparams["N_XI_PARAMS"][i]), maxiter=1000,maxfun=1000, full_output=True, disp=0)
                            dfuns[i]["xi_p"]=(dfuns[i]["xi_p"])**2
                            print("to", dfuns[i]["xi_p"], "Error=", mine, file=sys.stderr)
                            ################################
                        del ctxt    
                        print("Mixing order %d took: %s"%(i,formatted_span(time.time()-st)  ), file=sys.stderr)
#                MIX1, MIX2=0.1, 1.5
#                MIX1, MIX2=MIX1/(MIX1+MIX2), MIX2/(MIX1+MIX2)
##                cmax=__builtins__.max(od[i].itervalues())
#                st=time.time()
#                ctxt={}                
#                for ng, s in od[i].iteritems():
#                    ctx=split_ngram(ng)[0]                    
#                    try:
#                        ctxt[ctx][0]+=s
#                    except KeyError:
#                        ctxt[ctx]=[s, 0]
#                    
#                print >>sys.stderr, "\nMixing %d-grams using '%s'..."%(i, options.penalty [i])
#                save_scores=open_outfile("scores.before.2.%d"%(i))
##                ctxt2={}
#                for line in open_infile(options.penalty [i]):
#                    k, p=line.rsplit(None, 1)
#                    try:
#                        ctx=split_ngram(k)[0]
#                        
#                        news=exp(MIX1*log(float(p))+MIX2*log(float(od[i][k])/ctxt[ctx][0]))
#                        ctxt[ctx][1]+=news
##                        ctxt2[ctx]=ctxt2.get(ctx, 0)+news     
#                        print >>save_scores, k, od[i][k], float(od[i][k])/ctxt[ctx][0], p, news  
#                        od[i][k]=news
#                    except KeyError:
#                        print >>sys.stderr,"Missing %d-gram '%s' in the model"%(i, k)
#                for ctx, (csum, ssum) in ctxt.iteritems():
#                    ctxt[ctx]=float(csum)/ssum
#                
#                save_scores=open_outfile("scores.after.2.%d"%(i))
#                for k, s in od[i].iteritems():                    
#                    ctx=split_ngram(k)[0]           
#                    print >>save_scores, k, od[i][k], od[i][k]*ctxt[ctx]
#                    od[i][k]*=ctxt[ctx]
#                del ctxt
#                
                
                
    print("", file=sys.stderr)
#exit(0)
if False: #for i in assigns[rank][0]: #range(1, options.order+1):
    olderr=seterr(all="raise")
    seterr(under="ignore")
    
    if (options.n_scores <= 1 ) and (gparams["method"][i] != "cooc" ): #options.method != "cooc" or options.normalize:
        st=time.time()
        if freqd is None:
            freqd={}
        print("\nComputing/Loading associations for order:", i, file=sys.stderr)
#        print >>sys.stderr,"First 10 elements before :",od[i].items()[:10]
        # try finding the scores  in a file
        try:
            raise IOError
            print("Trying to load existent associations", file=sys.stderr)
            od[i]=pickle.load(file(("STDIN" if infile == sys.stdin else args[0])+"."+str(i)+"."+options.method+".pkl", "rb"))
            print("Loaded", file=sys.stderr)
        except IOError:
            print("Associations have not yet been created. Creating...", file=sys.stderr)
#            assocs=None
#            alpha=(RATE*max(od[i].values())-1)/(max(od[i].values())-min(v for v in od[i].values() if v >0))
#            beta=(max(od[i].values())-min(v for v in od[i].values() if v >0)*RATE*max(od[i].values()))/(max(od[i].values())-min(v for v in od[i].values() if v >0))
#            print >>sys.stderr, "Alpha=", alpha, "Beta=", beta, "Beta/Alpha=", beta/alpha
            try:
                freqd[i]=fromiter(list(od[i].values()),dtype=int)
                
                
                smax=amax(freqd[i])
#                for k,c in od[i].iteritems():
#                    freqd[i][k]=c
#                f=file("freqd-%d"%(i),"wb")
#                for k,c in freqd[i].iteritems():
#                    print >>f,"%s\t%s"%(k,c)
#                f.close()
#                f=file("od-%d"%(i),"wb")
#                for k,c in od[i].iteritems():
#                    print >>f,"%s\t%s"%(k,c)
#                f.close()
                if i !=1 : #or options.method not in ["chi-square", "log-likelihood-ratio"]:
                    
#                    values_vec=fromiter((v for v in od[i].values() if v >0), dtype=float)
#                    sigmap=std(values_vec)#od[i].values())
#                    mup=mean(values_vec)
#                    minb4, maxb4=min(values_vec), max(values_vec)
#                    print >>sys.stderr, "Mean data before scoring:", mean(od[i].values()), "STDDEV:", std(od[i].values()), "Range by stddev:", (max(od[i].values())-1)/std(od[i].values()), "Mean/STD=", mean(od[i].values())/std(od[i].values()), "STD/Mean=", std(od[i].values())/mean(od[i].values())
#                    uniq_vals=fromiter(set(v for v in od[i].values() if v >0), dtype=float)
#                    print >>sys.stderr, "****Number of unique values:", len(uniq_vals), "Mean unique values:", mean(uniq_vals), "STDDEV:", std(uniq_vals), "Range by stddev:", (max(uniq_vals)-1.)/std(uniq_vals), "Range/N_unique=", (max(uniq_vals)-1.)/len(uniq_vals)
#                    "Skew:", skew(od[i].values()), "Kurtosis:", kurtosis(od[i].values())
#                    od[i]=freqd[i].copy()
                    _, _, _, smax=compute_assoc(od[i],gparams["method"][i], split_ngram) #, scale=options.assoc_scales[i])
#                    compute_assoc(od[i],method=gparams["method"][i],separate=False,keepneg=False,normalize=False,bothdir=False,linearizedicts=True, between=None,  #(1.,  bis[i]), #RATE*max(od[i].values())), #
#                                                            transform=None, lower_bound=None, out_f2np=od[i])#, use_ranks=True)#, epsillon=options.epsillon)
#                    od[i],_,_,_=compute_assoc(od[i],method=options.method,separate=False,keepneg=True,normalize=options.normalize,bothdir=False,linearizedicts=True, between=None,  #(1.,  10*max(od[i].values())), #
#                                                            transform=None, lower_bound=None)#, use_ranks=True)#, epsillon=options.epsillon)
#                    print >>sys.stderr, "Mean data after scoring:", mean(od[i].values()), "STDDEV:", std(od[i].values()), "Range by stddev:", (max(od[i].values())-min(v for v in od[i].values() if v>0))/std(od[i].values()), "Mean/STD=", mean(od[i].values())/std(od[i].values()), "STD/Mean=", std(od[i].values())/mean(od[i].values())
#                    uniq_vals=fromiter(set(v for v in od[i].values() if v >0), dtype=float)
#                    print >>sys.stderr, "****After scoring: numberr of unique values:", len(uniq_vals), "Mean unique values:", mean(uniq_vals), "STDDEV:", std(uniq_vals), "Range by stddev:", (max(uniq_vals)-min(uniq_vals))/std(uniq_vals), "Range/N_unique=", (max(uniq_vals)-min(uniq_vals))/len(uniq_vals)

#                    print >>sys.stderr, "b1=", 1+(sigmap/std(od[i].values()))*(max(od[i].values())-min(od[i].values()))
#                    "Skew:", skew(od[i].values()), "Kurtosis:", kurtosis(od[i].values())
#                    print >>sys.stderr, "Alpha*std=", alpha*std(od[i].values()), "Alpha/std=",alpha/std(od[i].values()),"Beta*std=", beta*std(od[i].values()), "Beta/std",beta/std(od[i].values()),"std*Beta/Alpha=", std(od[i].values())*beta/alpha
                else:
                    print("'%s' is not suitable for the first order, falling back to 'cooccurrence'"%(gparams["method"][i]), file=sys.stderr) #options.method)
#                    continue
            except ZeroDivisionError:
                print("Zero division while evaluating associations. Probably '%s' association score is not suitable for order %d. Falling back to 'cooccurrence'"%(options.method, i), file=sys.stderr)
                if options.normalize:
#                    od[i],_,_,_=compute_assoc(od[i],method="cooc",separate=False,keepneg=False,normalize=options.normalize,bothdir=False,linearizedicts=True)#, epsillon=options.epsillon)
                    pass
#            if assocs is not None:
#                if options.assocs_are_weights:                    
#                    for ng in od[i]:
#                        od[i][ng]*=assocs[ng]
#                    assocs=None
#                else:
#                    od[i]=assocs
            ### In case we use backward scores, we need to flip the keys
#            tmp={}
#            for l,r in od[i]:
#                tmp[r,l]=od[i][l,r]
#            od[i]=tmp
            #Correct the case of BOS
            if i==1:
                if od[i][BOS]>0:
                    print("Correcting the score for <s>. Score before:",od[i][BOS], file=sys.stderr)
                    od[i][BOS]=0
                    print("Score after:",od[i][BOS], file=sys.stderr)
                    
        et=time.time()
        print("Done computing/loading the associations. It took", formatted_span(et-st), file=sys.stderr) 
#        
#        def comp_ll(p, scores, coocs, xivals, mask, n0):
#            if any(p<=0):
#                return 1e100
#            comp=(p[0]*coocs[mask] *exp(scores[mask]/amax(scores[mask]))/(coocs[mask]+1))*coocs[mask] # (scores[mask]+1)*coocs[mask]
#            comp/=sum(comp)
#            ll1=sum(coocs[mask]*log(comp*xivals))
#            ll0=n0*log(sum(comp*(1-xivals)))
#            return -ll1-ll0
#        print >>sys.stderr, "Tuning combination params..."
#        scores=fromiter(od[i].values(),dtype=float)
#        sc=scores.astype(float) #*freqd[i]
#        sc/=sum(sc)
#        factor=sum(freqd[i], dtype=float)/sum(od[i].values())
        
#        def dist(p, sc, scores, coocs):
#            if p<=0:
#                return 1e100
#            comp=(p*scores+1)*coocs
#            comp/=sum(comp)
#            return sum(comp*ma.log(comp/sc))
#            return -sum(ma.log(comp/sum(comp)))
#        
#        llo_c=freqd[i]-1
#        mask=llo_c>0
#        n0=sum(freqd[i]==1)
#        xiv=dfuns[i]["xi"](llo_c[mask])
        
#        mask=(sc>0)#*(co>0)
        
#        for params in arange(1, 10, 1):
#            comp=(params*scores+1)*freqd[i]
#            comp/=sum(comp)
#            d1=sum()
#            d3=.5*sum(comp[mask]*log(2./(1+sc[mask]/comp[mask]))+sc[mask]*log(2./(1+comp[mask]/sc[mask])))
#            d2=sum()           
#            d4=.5*sum(comp[mask]*log(2./(1+co[mask]/comp[mask]))+co[mask]*log(2./(1+comp[mask]/co[mask])))
#            d5=sum(comp[mask]*ma.log(comp[mask]/(co[mask]*sc[mask])) )
#            print >>sys.stderr, "Parameter:", params, "D1=", d1, "D2=", d2, "D3=", d3, "D4=", d4, "D5=", d5
#        params, negllk, _, _, _, _=fmin_powell(comp_ll,[1], args=(scores, llo_c, xiv, mask, n0), maxiter=1000,maxfun=1000, full_output=True, disp=0)
#        params, negllk, _, _, _, _=fmin_powell(dist,1., args=(sc[mask], scores[mask], freqd[i][mask]), maxiter=1000,maxfun=1000, full_output=True, disp=0)
#        print >>sys.stderr, "Best parameters:", params, "Likelihood:", -negllk
#        sc=(params*scores+1)*freqd[i]
#        reliable=1-sum(freqd[i]==1, dtype=float)/len(freqd[i])
#        maxc=amax(freqd[i])
#        print >>sys.stderr, "Reliability index:", reliable
        if options.assoc_scales is None:
            scale=log(2) #1. #exp(7.28*(reliable)**0.64 -2.16)
        else:
            try:
                scale=options.assoc_scales[i]
            except KeyError:
                scale=log(2)
            
#        if not 0<=scale<=1:
#            print >>sys.stderr,"Scale for order %d=%g is not suitble; using 0.1 instead"%(i,scale)
#            scale=1.
            
#        smax=max(od[i].itervalues())#
        if options.mix_weights is None:
            weight=.1
        
        else:
            try:
                weight=options.mix_weights[i]
            except KeyError:
                weight=.1
            
#        weight=options.mix_weights [i]
        if False and i != 1: #True:
            if options.mix_weights is None:
                weight=.1
            
            else:
                try:
                    weight=options.mix_weights[i]
                except KeyError:
                    weight=.1
                
            ctxt={}
            for x, (ng, s) in enumerate(od[i].items()):
                l, _=split_ngram(ng)
                s=exp(scale*s/smax )
                od[i][ng]=s
                ss, sc=ctxt.get(l, (0, 0))
                ctxt[l]=(ss+s, sc+freqd[i][x])
            
            for x, (ng, s) in enumerate(od[i].items()):
                l, _=split_ngram(ng)
                ss, sc=ctxt[l]
                od[i][ng]=weight*s/ss+(1-weight)*freqd[i][x]/float(sc)
            del ctxt
        if True and i != 1: #False:
#            out_cmp=open_outfile("compare-modif.%d"%(i))
#            all=sum(freqd[i])
#            mu, sig=meanstd(od[i].itervalues())
#            maxc=float(amax(freqd[i]))
#            temp_out=file("scores.%d"%(i), "wb")
            N_WORST=100
            if options.nworst is not None:
                N_WORST=options.nworst[i]
            print("Number of worst scores for order %i: %i" %(i, N_WORST), file=sys.stderr)
            
            minworst, maxworst=sorted(od[i].values())[:N_WORST][::N_WORST-1]
#            tmp_v=fromiter(od[i].itervalues(), dtype=float)
#            print >>sys.stderr, "minworst=", minworst, "maxworst=", maxworst, "Median of the lowest half=", median(tmp_v[tmp_v<median(tmp_v)])
            scale=1
            scale=1
            
            MIN_PEN, MAX_PEN=0.1, 1.0
            if options.highest_pen is not None:
                MAX_PEN=options.highest_pen[i]            
            if options.lowest_pen is not None:
                MIN_PEN=options.lowest_pen[i]
            print("Min penalty:", MIN_PEN, "Max penalty:", MAX_PEN, file=sys.stderr)
#            sys.exit(0)
            save_scores=open_outfile("scores.llr.%d"%(i))
            for x, (ng, s) in enumerate(od[i].items()):
                
#                cooc=float(freqd[i][x])
                od[i][ng]=float(freqd[i][x])
#                print >>sys.stderr, ng, s, cooc/all
                
            
                print("%s %g"%(ng, s), file=save_scores)
#                od[i][ng]=exp((cooc/(scale+cooc))* s/smax if cooc >3 else 0 )*cooc
                if s <= maxworst:
                    pen=MIN_PEN+(MAX_PEN-MIN_PEN)*(s-minworst)/(maxworst-minworst) #.1/(1+exp(-scale*s/smax ))
#                    print >>sys.stderr, "'%s' Count= %d; Score= %g; Penalty= %g; New count= %g"%(ng, float(freqd[i][x]), s, pen, float(freqd[i][x])*pen)
                    od[i][ng]*=pen
#                od[i][ng]=exp(scale*s/smax )*cooc
#                od[i][ng]=cooc**exp(scale*s/smax )
                
#                print >>temp_out, "%s\t%s\t%s\t%s"%(ng, cooc, s, od[i][ng])
#            temp_out.close()
#                od[i][ng]=(0.5+1/(exp( -(cooc/(scale+cooc))*s/smax )+1))*cooc
#                od[i][ng]=(0.6+s/smax/2 )*cooc
#                print >>out_cmp, "%s\t%d\t%g"%(ng, cooc, od[i][ng])
            
#            out_cmp.close()
#                od[i][ng]=exp(scale*s/smax+ (1-scale)*cooc/maxc)
            
#        factor=sum(freqd[i], dtype=float)/sum(od[i].values())        
#        for x, (ng, s) in enumerate(od[i].iteritems()):
#            od[i][ng]=s*factor
#        sys.exit(0)
        if  options.renorm2cooc is not None and options.renorm2cooc[i]:            
            st=time.time()
            print("Normalizing scores...", file=sys.stderr) 
            old_sum=sum(freqd[i])
            new_sum=sum(od[i].values())
            factor=float(old_sum)/new_sum
#            for k, (ng, sc) in enumerate(od[i].iteritems()):
#                od[i][ng]=freqd[i][k]/(1+log(1+sc)**-1)
            for ng in od[i]:
                od[i][ng]*=factor
                  
            et=time.time()
            print("Done normalizing association scores. It took", formatted_span(et-st), file=sys.stderr) 

sys.stderr.flush()         
if (options.n_scores > 1 ) :
    if freqd is None:
        freqd={}
    for i in assigns[rank][0]:
#        print >>sys.stderr,"First val:", od[i].iteritems().next()
        freqd[i]=fromiter((v[0] for v in od[i].values()),dtype=int)
#        freqd[i]=dict((k,int(v[0])) for k,v in od[i].iteritems())
                    

def max_mix(prob, weights):
    newp=(amax((prob), axis=1))
    return newp/sum(newp, axis=0)

def min_mix(prob, weights):
    newp=(amin((prob), axis=1))
    return newp/sum(newp, axis=0)

def loglin_mix(prob, weights):
    newp=exp(sum(weights*ma.log(prob), axis=1))
    return newp/sum(newp, axis=0)

def lin_mix(prob, weights):
    return (sum(weights*(prob), axis=1))
  
def harmlin_mix(prob, weights):
    newp=ma.power(sum(weights*ma.power(prob, -1), axis=1), -1)
    return newp/sum(newp, axis=0)  
if options.mix_dists or (options.n_scores > 1):
    
    if options.mix_func is not None:
        gparams["mix-func"]=options.mix_func 
    else:
        gparams["mix-func"]=ddict("linear",{})
    if options.mix_weights is not None:
        gparams["mix-weights"]=options.mix_weights 
    else:
        gparams["mix-weights"]=ddict([1.0/max(options.n_scores, 2)]*max(options.n_scores, 2),{})
    print("MIXING WEIGHTS=", options.mix_weights, file=sys.stderr)
    print("MIXING PARAMS=", gparams["mix-weights"], file=sys.stderr)
    
    print("", file=sys.stderr)
    for i in assigns[rank][0]:
        if (gparams["method"][i] != "cooc") or (options.n_scores > 1):

            print("\nMixing distributions for order: %d;"%(i), end=' ', file=sys.stderr) 
            
            print("using",gparams["mix-func"][i],"mixing, with weights:", gparams["mix-weights"][i], file=sys.stderr)
            
            if gparams["mix-func"][i] == "linear":
                gparams["mix-func"][i] =lin_mix
            elif gparams["mix-func"][i] == "log-linear":
                gparams["mix-func"][i] =loglin_mix
            elif gparams["mix-func"][i] == "harmonic-linear":
                gparams["mix-func"][i] =harmlin_mix
            elif gparams["mix-func"][i] == "max":
                gparams["mix-func"][i] =max_mix
            elif gparams["mix-func"][i] == "min":
                gparams["mix-func"][i] =min_mix
            else:
                raise NameError("Unknown mixing function: '%s'"%(gparams["mix-func"][i]))
                
            
#            print >>sys.stderr, "Sorting ngrams..."
            st=time.time()
#            sorted_ng=sorted(od[i])
##            n_ctxts=1 if i==1 else len(set(x.rsplit(None, 1)[0]  for x in sorted_ng))
#            et=time.time()
#            print >>sys.stderr, "Done sorting. It took", et-st, "s"
            try:
                NF=len(next(iter(od[i].values())))
            except TypeError:
                NF= 2
            print("Number of features=",NF, file=sys.stderr)
#            df=empty((n_ctxts+sum(1 for v in freqd[i].values() if v >0), NF))#c_[(dp), fromiter((v for v in od[i].values() if v >0), dtype=float)]
            if options.n_scores > 1:
                df=fromiter(( tuple(t) for j,(k,t) in enumerate(od[i].items()) if freqd[i][j]>0), dtype=",".join(["float"]*NF)).view(float).reshape((-1, NF))
            else:
                df=fromiter(((freqd[i][j], v) for j,(k,v) in  enumerate(od[i].items()) if freqd[i][j]>0), dtype=",".join(["float"]*NF)).view(float).reshape((-1, NF))
            
            df= df/sum(df, axis=0)
#            print >>sys.stderr, "Initial df=", df
#            mask=ones(len(df), dtype=bool)
#            mask[dind-1]=False
            ######## <=======>
  
            j=0
            print("Setting the mixed associations...", file=sys.stderr)
#            st=time.time()
    #        sco=sum(bp*qdf[:-1], axis=1)/dfxi #np[:-1]/dfxi  #dfuns[i]["xi"](df[mask][:,0])

            sco=gparams["mix-func"][i](df,gparams["mix-weights"][i])
            for l,k in enumerate(od[i]):
    #        for  j, g in enumerate(ng for ng in od[i] if od[i][ng]>0): #g in od[i]:
                if freqd[i][l]>0:
                    try:
    #                    print >>sys.stderr, "j=", j, "k=", k, "Old feature", od[i][k], "OR", df[mask][j]
                        od[i][k]=sco[j] #sum(bp*qdf[j])/xi[j] 
                        
                    except FloatingPointError:
                        print("Problem while computing exp for: NGRAM=",g,"Freq=", freqd[i][l], "Measure=", od[i][g], "Sum=", bp[0]*freqd[i][g]+bp[1]*od[i][g], file=sys.stderr)
                        raise
                    j+=1
                else:
                    od[i][k]=0
                    
                    
            et=time.time()
            print("Done mixing distributions. It took", formatted_span(et-st), file=sys.stderr) #, "s"
            
#        for  j, g in enumerate(ng for ng in od[i] if od[i][ng]>0): #g in od[i]:
##            if od[i][g]>0:
#                try:
#                    
#                    od[i][g]=sum(bp*qdf[j]) / dfuns[i]["xi"](freqd[i][g]) #(bp[0]*qdf[j][0]+bp[1]*qdf[j][1])#+bp[2])                    
##                    od[i][g]=(bp[0]*freqd[i][g]/sums[0]+bp[1]*od[i][g]/sums[1] )#+bp[2])
#                    continue
#                    if i >1:
#                        l, _=g.rsplit(None, 1)
#                    else:
#                        l=""
#                    od[i][g]=(bp[0]*freqd[i][g]/ctxt[l][0]+bp[1]*od[i][g]/ctxt[l][1] )#+bp[2])
##                    od[i][g]=(0.861196*freqd[i][g]/ctxt[l][0]+0.138804*od[i][g]/ctxt[l][1] )#
#                except FloatingPointError:
#                    print >>sys.stderr, "Problem while computing exp for: NGRAM=",g,"Freq=", freqd[i][g], "Measure=", od[i][g], "Sum=", bp[0]*freqd[i][g]+bp[1]*od[i][g]
#                    raise
    seterr(**olderr)
  
for i in range(1, options.order+1):
    if options.assoc_xi_param is not None and options.assoc_xi_param[i]:
        freqd[i]=None
                      
#
#for i in assigns[rank][0]: 
#    s=float(sum(od[i].values()))
#    for ng in od[i]:
#        od[i][ng]/=s
#sys.exit(0)

if freqd is not None:
    if __builtins__.all(x is None for x in list(freqd.values())):
        freqd=None

#print >>sys.stderr, "Freqd0=", freqd
if options.save_assocs:
    print("Saving associations to text files", file=sys.stderr)
    for i in assigns[rank][0]:
        fname=("STDIN" if infile == sys.stdin else args[0])+"."+str(i)+"."+options.method+".scores"
        f=file(fname, "wb")
        for ng in od[i]:
            print("%s\t%g"%( ng, od[i][ng]), file=f)
        f.close()
        print("Order",i,"Saved to file:",fname, file=sys.stderr)

if options.compute_assocs_only:
    sys.exit(0)
if options.astester:
    print("Starting in testing mode", file=sys.stderr)
    start4testing(od)

    ###END of smoothing parameter computation
#######
## Discarding low frequency ngrams if necessary
#######
#freqd=None

od[1][BOS]=0

sys.stderr.flush()
print("\n#############################################\n", file=sys.stderr)

#ng2keep={}
#                RECEIVE
#                
#                #send the prefixes of the kept ones to the lower order if 
#                    allngs=od[o].iterkeys() if freqd is None else freqd[o].iterkeys()
#                    kept=must_keep_for_lower(allngs, ng2rem[o] if o in ng2rem else set())
#                    comm.isend(kept, dest=modelat[o-1], tag=o)
#                    ### Unfortunately, irecv is not supported on objects :-(
#                    ng2rem[o-1]=comm.recv(source=modelat[o-1], tag=10*o)
#            

#print >>sys.stderr, "ng2rem=", ng2rem
#comm.barrier
#for i in assigns[rank][0]:   
#    print >>sys.stderr, "Number of ngrams for order %d=%d"%(i, len(od[i])-(len(ng2rem[i]) if i in ng2rem else 0))
##sys.exit(0)
#f=file("prefixes.2", "wb")
#for pref in prefixes[2]:
#    print >>f, pref
#f.close()
#sys.exit(0)
#if prefixes is not None:
#    del prefixes



maxkeep=lambda x: 1 if x <options.order else 0

try:
    outfile=open_outfile(args[1], "w")  
except IndexError:
    outfile=sys.stdout

    #print >>sys.stderr,"First ten values in unigrams after adjusting:"
    #print >>sys.stderr,od[1].items()[:10]
print(file=outfile)
print("\\data\\", file=outfile)
    

for i in range(1, options.order+1):    
    n=sum(ng2rem[i]<= (i <options.order )) if i in ng2rem else len(od[i]) #-(sum(ng2keep[i]) if i in ng2keep else 0)
            
    print("Number of ngrams for order %d=%d"%(i, n), file=sys.stderr)
    print("ngram %d=%d"%(i, n), file=outfile)
outfile.flush()

probs=None
#### VErification
#if options.sumprobs_prefix is not None:   
#    allprobs={}
#    allweights={}
#    
#### end Verif
seterr(all="raise")
#print >>sys.stderr, "Process", rank, "My assignments:", assigns[rank]
bow=None
psend_r=None
for i in range(1, options.order+1):
    print("\nComputing probabilities for order:", i, file=sys.stderr)
    print("=====================================", file=sys.stderr)
    start=time.time()
#    acounts=array(od[i].values())
#    if freqd is not None:
#        freqs=array(freqd[i].values())
    print("Final number of ngrams:",len(od[i]), file=sys.stderr)
#    print >>sys.stderr,  "Discounting constants:",Dd[i]
#    if i not in options.nodiscount:
    #        continue
    #    if probs is not None:
    #        print >>sys.stderr,"First ten values of the previous probabilities:", probs.items()[:10]
    #    else: 
    #        print >>sys.stderr,"Previous probabilities are still empty"
    #    first10= array([x for x in sorted(set(v for v  in  od[i].values() if v>0))[:10]])
    #    print >>sys.stderr, "\n"
    #    print >>sys.stderr,"AGAIN First 10 associations for order %i: [%s]" %(i,", ".join("%g"%(x) for x in first10))
    #    for x in first10:
    #        t1=dfuns[i]["body_params"][2]*(dfuns[i]["body_params"][0]/dfuns[i]["body_params"][1])
    #        print >>sys.stderr,"T1=", t1
    #        
    #        t2=(x/dfuns[i]["body_params"][1])**(dfuns[i]["body_params"][0]-1)
    #        print >>sys.stderr,"T2=", t2
    #        t3=exp(-(x/dfuns[i]["body_params"][1])**dfuns[i]["body_params"][0])
    #        print >>sys.stderr,"T3=", t3
    #        t4=(1-exp(-(x/dfuns[i]["body_params"][1])**dfuns[i]["body_params"][0]))**(dfuns[i]["body_params"][2]-1)
    #        print >>sys.stderr,"T4=", t4
    #        
    #        print >>sys.stderr, "Score (%g)= %g" %(x, t1*t2*t3*t4)
    #    print >>sys.stderr, "Func eval:"
    #    for x in first10:
    #        print >>sys.stderr, "Score (%g)= %g" %(x, dfuns[i]["body_params"][2]*(dfuns[i]["body_params"][0]/dfuns[i]["body_params"][1])*(x/dfuns[i]["body_params"][1])**(dfuns[i]["body_params"][0]-1)*exp(-(x/dfuns[i]["body_params"][1])**dfuns[i]["body_params"][0])*(1-exp(-(x/dfuns[i]["body_params"][1])**dfuns[i]["body_params"][0]))**(dfuns[i]["body_params"][2]-1)) #dfuns[i]["func"](x))
    #    firstdis=map(dfuns[i]["func"], first10)
    #    print >>sys.stderr,"==>AGAIN Their corresponding discounts: [%s]" %(", ".join("%g"%(x) for x in firstdis))
    #    

    #        print >>sys.stderr, kvals
    #            except FloatingPointError:
    #                print >>sys.stderr, "Problem in order %d for count value "%(i), v
    #                print >>sys.stderr, "Lower dist=", dfuns[i]["body_dist"](float(v))
    #                print >>sys.stderr, "Parametrs for this order:", dfuns[i]["body_params"]
    #                raise
    #        print >>sys.stderr,"Number of unique values:",len(ui),"==",len(kvals)
    #        Di, pv, gm,weights= order_vectors(od, i, D=Dd[i], lprobs=probs,discount=options.smoothing in DISCOUNT_METHODS, e=hd[i]["e"]) 
    #        Di, pv, gm,weights= order_vectors(od, i, D=[0], lprobs=probs,discount=options.smoothing in DISCOUNT_METHODS, e=[1],dfun=dfuns[i]["func"])
    #    for m in od:
#        print >>sys.stderr, "Process", rank, "has dictionary:", m
    sys.stderr.flush()
    if options.kern_smooth:
#        if freqd is None:
#            discounter=lambda x: kvals[x]
#        else:
#            discounter=lambda x, freq: x*(1-kvals[freq]) 
        if gparams["xi_func"][i].startswith("xi_comb"):
            p=compute_probs_comb(od[i], od[i-1] if i-1 in od else None, dfuns[i]["xi"], xi_arg=freqd and freqd[i], mass_from_arg=options.reuse_unseen, smoothing_dist=options.smoothing_uni_dist)
        elif options.smoothing.startswith("wb"):
#            Di, pv, gm= compute_prob_args_wb(od, i, D=[0], lprobs=od[i-1] if i-1 in od else None,discount=options.smoothing in DISCOUNT_METHODS, e=[1], xi_fun=dfuns[i]["xi"], freqd=freqd)
            p=compute_probs_wb(od[i], od[i-1] if i-1 in od else None, dfuns[i]["xi"], xi_arg=freqd and freqd[i], mass_from_arg=options.reuse_unseen, smoothing_dist=options.smoothing_uni_dist)
        else:
            p=compute_probs_kn(od[i], od[i-1] if i-1 in od else None, dfuns[i]["xi"], xi_arg=freqd and freqd[i], mass_from_arg=options.reuse_unseen, smoothing_dist=options.smoothing_uni_dist)
#            Di, pv, gm= compute_prob_args_kn(od, i, D=[0], lprobs=od[i-1] if i-1 in od else None,discount=options.smoothing in DISCOUNT_METHODS, e=[1],xi_fun=dfuns[i]["xi"], freqd=freqd)
    
#        Di, pv, gm= compute_prob_args_kn(od, i, D=[0], lprobs=od[i-1] if i-1 in od else None,discount=options.smoothing in DISCOUNT_METHODS, e=[1],xi_fun=dfuns[i]["xi"], freqd=freqd)
#        del kvals
    else:
        
#        Di, pv, gm, weights= order_vectors(od, i, D=Dd[i], lprobs=probs,discount=options.smoothing in DISCOUNT_METHODS)    
        Di, pv, gm= order_vectors(od, i, D=Dd[i], lprobs=od[i-1] if i-1 in od else None,discount=options.smoothing in DISCOUNT_METHODS)
    if freqd is not None:
        if freqd[i] is not None:
            del freqd[i]
#    savetxt("sd-all%d"%(i),c_[acounts,Di,acounts-Di])

    #### have a look at the weights
#    f=file("weights."+str(i), "w")
#    for w in weights:
#        print >> f, w, weights[w]
#    f.close()
    #####
    #    continue
    #if probs is None:
        #pv-=od[1][BOS]
#        Di=0.0
#        pv=norm_1g(od)
#    print >>sys.stderr, "Normalizing factor in order %d: D=%s, COUNTS=%s, DENOM=%s, INTERP_FACTOR=%s"%(i,Di, acounts,  pv, gm)
    
    if not options.interpolate and i >1:
        gm=0
        
#    if i in  options.nodiscount:
#        p=compute_kn_probs(0,  acounts, pv, 0)
#    else:
#        p=compute_kn_probs(Di, acounts, pv, gm)
#    savetxt("P.%s"%(time.time()), p)
#    sc_out=file(str(os.getpid())+"."+str(i), "wb")
#    if i<=1:        
#        for j, ng in enumerate(od[i]):
#            print >>sc_out, "%s\t%g\t%g\t%g"%(ng, acounts[j], pv, p[j])
#    
#    else:
#        for j, ng in enumerate(od[i]):
#            print >>sc_out, "%s\t%g\t%g\t%g"%(ng, acounts[j], pv[j], p[j])
#    
#    sc_out.close()
    
##    f=file("pp."+str(i)+".rec_at."+str(i),"wb")
##    for j,ng in enumerate(od[i]):
##        print >>f,ng,p[j]
##    f.close()  el", i+1

#    savetxt("cprobs.%d"%(i),c_[p,Di,acounts, ones(len(p))*pv, ones(len(p))*gm])
#    del Di
#    del acounts
#    del pv
#    del gm
    
    if any(p>1):
        print("***************************Some probabilities are larger than 1,  in order:", i, file=sys.stderr)
        if False:
            for x, ng in enumerate(od[i]):
                if p[x]>1:
                    print("\t\t", ng, p[x], file=sys.stderr)
#    print >>sys.stderr,"Sum(Probs)=",sum(p)
#    print >>sys.stderr, "Number of probabilities: %d"%(len(p) )
#    probs={}
#    for j, ng in enumerate(od[i]):
#        print log10(p[j]) if p[j]>0 else -99, ng
#        probs[ng]=p[j]
    end=time.time()
    print("Computing probabilities for order",i,"took:", formatted_span(end-start), file=sys.stderr) # , "s"
    start=time.time()
    ###Backoffs of previous order
    
    P_NG_BO_FORMAT="%.8g\t%s\t%.8g"
    P_NG_FORMAT="%.8g\t%s"
    if options.sb:
        
    
        print("Using stupid backoff, no need to compute the backoff weights for order:", i, file=sys.stderr)

        if i>1 and size>1 and rank != modelat[i-1]:
            notif=empty(1, dtype=int)
            print("Process",rank, "waiting for notification from process", modelat[i-1], "(taking care of model", i-1, ")", file=sys.stderr)
            
            comm.Recv(notif,modelat[i-1], 111)# notif, source=modelat[i-2], tag=i-2)
            print("Process",rank, "notification received from process", modelat[i-1], file=sys.stderr)
            outfile.seek(0, 2)
            
        print("\n\\%d-grams:"%(i), file=outfile)
#        bf=file("pbows.%d"%(i), "wb")
#        hbow=lambda o, p:(0.4+.45*log(o+1))/(1+log(o+1))
#            0.5/(1+.2*log(1+p))
        
        
        toout=(lambda x:ng2rem[i-1][x]==0) if i-1 in ng2rem else (lambda x: True) #logical_not(ng2keep[i]) if i in ng2keep else set()
        if i<options.order:
            for j, ng in enumerate(od[i]):
                if not toout(j):
                    continue
            
    #            ngram=ng #WORD_SEP.join(g for g in ng if g)

                
                print(P_NG_BO_FORMAT%(log10(p[j]) if p[j] else -99, ng, log10(DEFAULT_BOW[i])), file=outfile)#log10(hbow(i, p[j]))) #
    #                print >>bout, "%s\t%.15f\t%.15f\t%.15f\t%.15f"%(ng, probs[ng], bow[ng][0], bow[ng][1],bow[ng][0]/bow[ng][1] )
#                print >>bf, "%.15f\t%.15f"%(p[j], hbow(i, p[j]))
        else:
            for j, ng in enumerate(od[i]):
                if not toout(j): #ng in noout:
                    continue
            
    #            ngram=ng #WORD_SEP.join(g for g in ng if g)

                
                print(P_NG_FORMAT %(log10(p[j]) if p[j] else -99, ng), file=outfile)
            print("\n\\end\\", file=outfile)
    elif 1<i<=options.order:
        print("\nComputing backoff weights for order:",i, file=sys.stderr)
        print("======================================", file=sys.stderr)
#        print >>sys.stderr, "SB=", options.sb
        if bow is not None:
            if i-1 in ng2rem:
                print_err( "Computing the mass freed by the excluded ngrams for order:", i-1)
                for j, ng in enumerate(od[i-1]):
                    if ng2rem[i-1][j]>=1:
                        if i-1>1:
                            l,_=ng.rsplit(None,1)
                            _,right=ng.split(None, 1) #ng.split(None,1)
                        else:
                            l=""
                            right = ng
                        if l in bow:
                            od[i-1][ng]=(bow[l][0]/bow[l][1])*od[i-2][right]
#                    del ng2rem[i-1]
#        print >>sys.stderr,i-1,"in NG2REM:",i-1 in ng2rem,"---",i,"inNG2REM:",i in ng2rem
#        print >>sys.stderr,"First 10 probs:"
#        for n,(k,v) in enumerate(od[i-1].iteritems()):
#            if i>10: break
#            print >>sys.stderr,"P(%s)= %g"%(k,v)
         
#        print >>sys.stderr,"Number of ngs in order",i,"=",len(od[i])
#        if i in ng2rem:
#            print >>sys.stderr,"Number of entries in ng2rem in order",i,"=",len(ng2rem[i])
#        
#        print >>sys.stderr,"Number of ngs in order",i-1,"=",len(od[i-1])
#        if i-1 in ng2rem:
#            print >>sys.stderr,"Number of entries in ng2rem in order",i-1,"=",len(ng2rem[i-1])
            
##        f=file("pp."+str(i-1)+".rec_at."+str(i),"wb")
##        for ng in (od[i-1]):
##            print >>f,ng,od[i-1][ng]
##        f.close()
        print("Computing backoffs", file=sys.stderr)
        bow=compute_bow(od,i,p,od[i-1],ng2rem)
        
    
        
#        NPARTS=4
#        bows=dict()
#        for part in range(NPARTS):
#            bowp=compute_partial_bow(od,i,p,probs,ng2rem, NPARTS, part)
#            print >>sys.stderr, "Comparison:"
#            for ng in bowp:
#                if abs(bow[ng][0]- bowp[ng][0])>1e-10 or abs(bow[ng][1]- bowp[ng][1])>1e-10:
#                    print >>sys.stderr,"'%s': Correct BOW: %s... Partial BOW: %s"%(ng, bow[ng], bowp[ng])
#            bows.update(bowp)
#        if bows.keys() == bow.keys():
#            for ng in bows:
#                if abs(bow[ng][0]- bows[ng][0])>1e-10 or abs(bow[ng][1]- bows[ng][1])>1e-10:
#                    print >>sys.stderr,"'%s': Correct BOW: %s... Partial BOW: %s"%(ng, bow[ng], bows[ng])
#            print >>sys.stderr,"Comparison done"
#        else:
#            kbows, kbow=set(bows.keys() ), set(bow.keys() )
#            print >>sys.stderr,"Comparison failure: dictionaries have different keys, len(Bows)= %d, len(bow)= %d"%(len(bows), len(bow))
#            print >>sys.stderr,"ngrams in BOWS but not in BOW:", kbows- kbow
#            print >>sys.stderr,"ngrams in BOW but not in BOWS:", kbow- kbows
#            
            
#            print >>sys.stderr,"Different keys=", [(k1, k2) for k1, k2 in zip( bows.keys() ,  bow.keys()) if k1 != k2]
    
#            sys.exit(0)
    
#        print >>sys.stderr,"Computing backoff weights for order:",i-1
#        bow={}
#        for j, ng in enumerate(od[i]):
#            try:
#                left,_=ng.rsplit(None,1)
#                _,right=ng.split(None,1)
#                bow[left][0]  -=p[j]
#                bow[left][1]  -=probs[right]
#            except KeyError:
#                bow[left]  = [1-p[j],1-probs[right]]
        #outputting order i-1

        end=time.time()
        print("Computing backoff weights for order", i, "took:", formatted_span(end-start), file=sys.stderr) #, "s"
        
        print("Writing model", i-1, file=sys.stderr)
        start=time.time()
        print("\n\\%d-grams:"%(i-1), file=outfile)
        
        toout=(lambda x:ng2rem[i-1][x]<=1) if i-1 in ng2rem else (lambda x: True) #set()
    #        bout=file("bows."+str(i), "wb")
#            bf=file("pbow.%d"%(i-1), "wb")
        for j,ng in enumerate(od[i-1]):

            if (not toout(j)) : #and (ng not in bow): #ng in noout:
                continue
#            ngram=ng #WORD_SEP.join(g for g in ng if g)
            
            try:
                bo=bow[ng][0]/bow[ng][1]
                try:
                    bo=log10(bo)
                except (FloatingPointError,ZeroDivisionError) :
#                    print >>sys.stderr,"NGRAM=",ng,"BOW num=",bow[ng][0],"BOW den=",bow[ng][1],"Value of ng2rem=",ng2rem[i-1][j] if i-1 in ng2rem else 100
                    bo=LOG10_ZERO
                    print("%g/%g=BO=%g"% (bow[ng][0], bow[ng][1], bo), file=sys.stderr)
                    
                print(P_NG_BO_FORMAT%(log10(od[i-1][ng]) if od[i-1][ng] else -99, ng, bo), file=outfile)
#                    print >>bf, "%.15f\t%.15f"%(probs[ng], bow[ng][0]/bow[ng][1])
#                print >>bout, "%s\t%.15f\t%.15f\t%.15f\t%.15f"%(ng, probs[ng], bow[ng][0], bow[ng][1],bow[ng][0]/bow[ng][1] )
            except KeyError:
#                if i-1 in ng2keep and ng2keep[i-1][j] ==1:
#                    print >>outfile,"%.15g\t%s\t0"%(log10(od[i-1][ng]) if od[i-1][ng] else -99, ng)
#                else:
                print(P_NG_FORMAT %(log10(od[i-1][ng]) if od[i-1][ng] else -99, ng), file=outfile)
            except FloatingPointError:
                print("'%s' ----> BOW=%g/%g;;; Prob=%.15g"% (ng, bow[ng][0], bow[ng][1], od[i-1][ng]), file=sys.stderr)
                
                print(P_NG_FORMAT+"\t0"%(log10(od[i-1][ng]) if od[i-1][ng] else -99, ng), file=outfile)
        end=time.time()
        print("Writing model", i-1, "took:", formatted_span(end-start), file=sys.stderr) #, "s"
        
    outfile.flush()
    
    if i-1 in ng2rem:
        del ng2rem[i-1]
    
    if not options.sb:
        
#        if i<options.order:
#            if  size<=1 or rank == modelat[i+1]:
#                newprobs=dict.fromkeys(od[i])
        ##for verification
    #    if options.sumprobs_prefix is not None:   
    #        print >>sys.stderr,"Extracting all contexts"
    #        contexts=set(left for left, _ in od[i] )
        ##end verification
        if i==options.order:
            start=time.time()
            print("Writing model", i, file=sys.stderr)
            toout=(lambda x:ng2rem[i][x]==0) if i in ng2rem else (lambda x: True) #set()
            print("\n\\%d-grams:"%(i), file=outfile)
            for j, ng in enumerate(od[i]):
                if not toout(j) : #ng in noout:
                    continue
                print(P_NG_FORMAT %(log10(p[j]) if p[j] else -99, ng), file=outfile) #WORD_SEP.join(g for g in ng if g))
            print("\n\\end\\", file=outfile)
            end=time.time()
            print("Writing model", i, "took:", formatted_span(end-start), file=sys.stderr) #, "s"
            
            if ind_send is not None:
                ind_send.Wait()
            if i in ng2rem:
                del ng2rem[i]
    ### VErfication
    #            if options.sumprobs_prefix is not None:
    #                newprobs[ng]=p[j]   
    #### end verif
    #####
            #####
        else:
            if i<options.order:
                print("Storing probabilities of order", i, "for the next run...", file=sys.stderr)
                for j, ng in enumerate(od[i]):        
#                        newprobs[ng]=p[j]   
                            
                    od[i][ng]=p[j] 
                    
                if i>1:
                    if bow is not None:
                        if i in ng2rem:
                            print("Computing the mass freed by the excluded ngrams for order:", i-1, file=sys.stderr)
                            for j, ng in enumerate(od[i]):                         
                                if ng2rem[i][j]>=1:                        
                                    if i>1:
                                        l,_=ng.rsplit(None,1)
                                        _,right=ng.split(None, 1) #ng.split(None,1)
                                    else:
                                        l=""
                                        right = ng 
                                    if l in bow:
                                        p[j]=(bow[l][0]/bow[l][1])*od[i-1][right]
                                        
                                            
                            
                   
#        if bow is not None:               
#            del bow
            
        sys.stderr.flush()
        if i-2 in od:
            del od[i-2]             
#        del p
         
        ## JUST FOR VERIFICATION     
    #    def getprob(t, probs, weights, ord):
    #        c, w=t
    #        if t in probs[ord]:
    ##            print >>sys.stderr,"P[order=%d]('%s', '%s')=%f"%(ord, c, w, probs[ord][t])
    #            return probs[ord][t]
    ##        print >>sys.stderr,"\t\t'%s' not found in order %d,  interpolating"%(t, ord)
    #        if c=="" or ord==1:
    ##            print >>sys.stderr,"Returning 0 prob (could not find an entry)"
    #            return 0.0
    #        try:
    #            _, cr=c.split(None, 1)
    #        except ValueError:
    #            cr=""
    ##        print >>sys.stderr,"Recursively looking for '%s' (Weight('%s')=%f)"%((cr, w), c, weights[ord][c])
    #        p=weights[ord][c]*getprob((cr, w), probs, weights, ord-1)
    #
    #        return p
    #    
    #    if options.sumprobs_prefix   :
    #        allprobs[i]=newprobs
    #        allweights[i]=weights
    #    if options.sumprobs_prefix is not None  :   
    #        print >>sys.stderr,"\nComputing sum of probabilities for order:",i
    #        print >>sys.stderr,    "==========================================="
    #        sums={}
    #        print >>sys.stderr,"Processing %d contexts" %(len(contexts))
    #        
    #        for ic, c in enumerate(contexts):
    #            print >>sys.stderr,"Now processing context No. %d out of %d: " %(ic, len(contexts)), 
    #            for w in voc:
    #                
    #                sums[c]=sums.get(c, 0)+getprob((c, w), allprobs, allweights, i)
    ### 
    #            print >>sys.stderr,"P('%s'*)= %f"%(c,sums.get(c, 0))
    #                    
    #        sump=file(options.sumprobs_prefix+".%d-grams"%(i), "wb")
    #        for ng in sums:
    #            print >>sump,"%s\t%.15f"%(ng, sums[ng])
    #        sump.close()       
    ##    
        ##end verification
#        if i<options.order:
#            if  size<=1 or rank == modelat[i+1]:
#                probs=newprobs
#    
#if
#    print >>outfile,"\n\\end\\"
if outfile is not None:
    outfile.close()        
#    
#if options.wpfile is not None:
#    options.wpfile.close()
#    compute_smoothing_attributes(od, i+1)
#    if i>0:
#        sum_counts(od, i+1)
    
    
sys.stderr.flush()
##+free resources while waiting for others
global_end=time.time()
print("\nDone @", time.ctime(), file=sys.stderr)
print("\n\n**It took %s to complete the whole training**"%(formatted_span(global_end-global_start)), file=sys.stderr)


#
#acounts=array(od[ord].values())
#D= compute_smoothing_attributes(acounts, ord)
#print "D=", D
#Di, pv, gm= order_vectors(od, ord, D)
#print "Di=", Di, "PV=", pv, "Gamma=", gm
#compute_kn_probs(Di, acounts, pv, gm)
