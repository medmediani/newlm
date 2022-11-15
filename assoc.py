#!/usr/bin/env python
# -*- coding: utf-8 -*-


import sys, os
from scipy.special import gammaln as lgamma
from scipy.stats import hypergeom as hg

import numpy as np
from optparse import OptionParser
#import __builtin__

from scipy.optimize import fmin_powell

WORD_SEP=" "

NBR_KN_CONSTS=3

PENALIZE=1
BIG=1e100
ALPHA=0.5


os.system("taskset -p 0xFFFFFFFF %d >/dev/stderr" % os.getpid())


def open_infile(f, mode="rb"):
    return file(f,mode) if f !="-" else sys.stdin
  
def open_outfile(f, mode="wb"):
    return file(f,mode) if f !="-" else sys.stdout
    
 
def lfact(n):
    return lgamma(n+1)


def xlogx(x):
    return x*log(x) if x>0 else 0
    
def xlogy(x, y):
    return x*log(max(y, 1e-50))
    
def lcomb(n, r):
    if r>n:
        return 0
    return lfact(n)-lfact(r)-lfact(n-r)
    
    
def comb(n, r):
    if r>n:
        return 1
    lval=lfact(n)-lfact(r)-lfact(n-r)
    return exp(lval)
    
def cooccurrence(cooc, socc, tocc, all):
    return float(cooc)

def multinomial_likelihood(cooc, socc, tocc, all):
    lval=lfact(all)-lfact(cooc)-lfact(socc-cooc)-lfact(tocc-cooc)-lfact(all+cooc-tocc-socc)+\
             xlogx(socc)+xlogx(tocc)+xlogx(all-socc)+xlogx(all-tocc)\
             -2*xlogx(all)
#    if lval>=0:
#        print >>sys.stderr, "Got score less than 0:", lval,"===>(",cooc,socc,tocc,all,")"
             
    return -lval
 
def binomial_likelihood(cooc, socc, tocc, all):
    lval=lcomb(all, cooc)+cooc*log((float(socc)/all)*(float(tocc)/all))+(all-cooc)*log(1-(float(socc)/all)*(float(tocc)/all))
#    if lval>=0:
#        print >>sys.stderr, "Got score less than 0:", lval,"===>(",cooc,socc,tocc,all,")"
    
    return -lval

def poisson_likelihood(cooc, socc, tocc, all):
    lval=-(float(socc)/all)*tocc + cooc*log((float(socc)/all)*tocc)-lfact(cooc)
#    if lval>=0:
#        print >>sys.stderr, "Got score less than 0:", lval,"===>(",cooc,socc,tocc,all,")"
    return -lval
    
def poisson_stirling(cooc, socc, tocc, all):
    lval=cooc*(log((float(cooc)/tocc)*(float(all)/socc))-1)
#    if lval<=0:
#        print >>sys.stderr, "Got score less than 0:", lval,"===>(",cooc,socc,tocc,all,")"
    return lval
    
def hypergeometric_likelihood(cooc, socc, tocc, all):
    lval=lcomb(tocc, cooc)+lcomb(all-tocc, socc-cooc)-lcomb(all, socc)
#    if -lval<=0:
#        print >>sys.stderr, "Got score less than 0:", -lvalcat ,"===>(",cooc,socc,tocc,all,")"
    return -lval
    ####    
def z_score(cooc, socc, tocc, all):
    ##shift value added to get only positive scores
    score=(cooc-(float(socc)/all)*tocc)/sqrt((float(socc)/all)*tocc) #+(0.5*all-1.5+2./(all+1))/sqrt(all)
#    if score<=0:
#        print >>sys.stderr, "Got score less than 0:", score,"===>(",cooc,socc,tocc,all,")"
    return score
    
def t_score(cooc, socc, tocc, all):
    score=(cooc-(float(socc)/all)*tocc)/sqrt(cooc) #+(0.25*all-0.5+0.25/all)
#    if score<=0:
#        print >>sys.stderr, "Got score less than 0:", score,"===>(",cooc,socc,tocc,all,")"
    return score
     
def chi_square_ind(cooc, socc, tocc, all):
    val=( ( (float(all)/(socc))/tocc )*(cooc-(float(socc)/all)*tocc)**2) / (1-float(tocc)/all) / (1-float(socc)/all)
#    if lval>=0:
#        print >>sys.stderr, "Got score less than 0:", score,"===>(",cooc,socc,tocc,all,")"
    return val
       
def chi_square(cooc, socc, tocc, all):
#    score=(float(socc*tocc)*((float(all)/socc)*(float(cooc)/tocc)-1)**2)/((float(all)/socc-1)*(float(all)/tocc-1))
    e11=float(socc*tocc)/all
    e22=(1-float(socc)/all)*(1-float(tocc)/all)
    score=((cooc-e11)**2/(e11*e22))
#    if score<=0:
#        print >>sys.stderr, "Got score less than 0:", score,"===>(",cooc,socc,tocc,all,")"
    return score
    
def llr(cooc, socc, tocc, all):    
    socc, tocc, cooc=float(socc), float(tocc), float(cooc)
    val=cooc*(log(cooc)+log(all)-log(socc)-log(tocc))
    if tocc>cooc:
        val+=(tocc-cooc)*(log(all)+log(tocc-cooc)-log(tocc)-log(all-socc))
    if socc>cooc:
        val+=(socc-cooc)*(log(all)+log(socc-cooc)-log(socc)-log(all-tocc))
    f=all+cooc-socc-tocc
    if f>0:
        val+=f*(log(all)+log(f)-log(all-tocc)-log(all-socc))
    return 2*val
    lval=2*(
                
                xlogy(cooc, (cooc*all)/socc/tocc)
                +xlogy(tocc-cooc, (all/tocc)*(tocc-(cooc))/(all-(socc)))
                +xlogy(socc-cooc, (all/socc)*(socc-(cooc))/(all-(tocc)))
                +xlogy(all+cooc-socc-tocc, (all/(all-tocc))*((all-socc-tocc+cooc)/(all-socc)))
                )
#    return max(lval, 1e-50)
    return lval #, 1e-50)
    
    
#def llr(cooc, socc, tocc, all):    
#    socc, tocc, cooc=float(socc), float(tocc), float(cooc)
#    
#    lval=2*(
#                xlogy(cooc, (cooc*all)/socc/tocc)
#                +xlogy(tocc-cooc, (all/tocc)*(tocc-(cooc))/(all-(socc)))
#                +xlogy(socc-cooc, (all/socc)*(socc-(cooc))/(all-(tocc)))
#                +xlogy(all+cooc-socc-tocc, (all/(all-tocc))*((all-socc-tocc+cooc)/(all-socc)))
#                )
#    return max(lval, 1e-10)
#    lval=2*(xlogx(cooc)+xlogx(all)-2*(ALPHA)*xlogx(socc)-2*(1-ALPHA)*xlogx(tocc)+2*(ALPHA)*xlogx(socc-cooc)+2*(1-ALPHA)*xlogx(tocc-cooc)-2*(1-ALPHA)*xlogx(all-tocc)-2*(ALPHA)*xlogx(all-socc) +xlogx(all+cooc -2*(ALPHA)*socc-2*(1-ALPHA)*tocc))
#    return lval
    
def llr_dunning(cooc, socc, tocc, all):    
    f3=(socc-cooc)*log(float(socc-cooc)/(all-tocc)) if socc>cooc else 0
    f2=(tocc-cooc)*log(1-float(cooc)/tocc) if tocc>cooc else 0
    f4=(all+cooc-socc-tocc)*log(1-float(socc-cooc)/(all-tocc)) if all +cooc> tocc+socc else 0
    f1=(all-socc)*log(1-float(socc)/all) if all > socc else 0
    
    lval=socc*log(float(socc)/all) +f1 - cooc*log(float(cooc)/tocc)- f2-f3-f4
    return -2*lval
 
def mi(cooc, socc, tocc, all):    
    lval=log(1+ (float(all)/socc**(2*ALPHA))*(float(cooc)/tocc**(2*(1-ALPHA)) ) )
    
    return lval
 
def mi2(cooc, socc, tocc, all):    
    return log(1+ (float(all)/socc**(2*ALPHA))*(float(cooc)**2/tocc**(2*(1-ALPHA))))
 
def mi3(cooc, socc, tocc, all):    
    return log(1+ (float(all)/socc**(2*ALPHA))*(float(cooc)**3/tocc**(2*(1-ALPHA))))
       
def lmi(cooc, socc, tocc, all):    
    lval=cooc*log(1+(float(all)/socc**(2*ALPHA))*(float(cooc)/tocc**(2*(1-ALPHA))))
    return lval
    
def ami(cooc, socc, tocc, all):    
    lval=xlogx(cooc)+xlogx(all)-xlogx(socc)-xlogx(tocc)+xlogx(socc-cooc)+xlogx(tocc-cooc)-xlogx(all-tocc)-xlogx(all-socc)+xlogx(all+cooc-socc-tocc)
    if lval <ZERO:
#        print >>sys.stderr, "Got score very low:", lval, "==>", cooc, socc, tocc, all
        lval=ZERO
    return lval
    
def odds_ratio  (cooc, socc, tocc, all):    
    lval=log(1+((cooc+0.5)/(socc-cooc+0.5))*(all+cooc-socc-tocc+0.5)/(tocc-cooc+0.5) )
    
    return lval

def gmean(cooc, socc, tocc, all):    
    val=float(cooc)/((socc**ALPHA)*(tocc**(1-ALPHA)))
    return val

def dice(cooc, socc, tocc, all):    
    val=(cooc)/((ALPHA)*socc+(1-ALPHA)*tocc)
    return val
    
def jaccard(cooc, socc, tocc, all):    
    val=float(cooc)/(2*(ALPHA)*socc+2*(1-ALPHA)*tocc-cooc)
    return val

def amean(cooc, socc, tocc, all):    
    val=float(cooc)*(ALPHA/socc+(1-ALPHA)/tocc)
    return val
    
def rho(cooc, socc, tocc, all):
    val=2-(2-float(cooc)/socc)**ALPHA*(2-float(cooc)/tocc)**(1-ALPHA)
    return val
#     
#def rho2(cooc, socc, tocc, all):
#    val=cooc*(1-(1-float(cooc)/socc)**ALPHA*(1-float(cooc)/tocc)**(1-ALPHA))
#    return val
#    
def lnorm(cooc, socc, tocc, all):
    val=(ALPHA*socc+(1-ALPHA)*tocc)**(1./NORM) - (ALPHA*socc+(1-ALPHA)*tocc-cooc)**(1./NORM) 
    return val
  
def lnorm2(cooc, socc, tocc, all):
    val=1 - (1-cooc/(ALPHA*socc+(1-ALPHA)*tocc))**(1./NORM) 
    return val
    
def fisher(cooc, socc, tocc, all):    #fisher exact test
    val=hg.cdf(min(socc, tocc), all, tocc, socc)-hg.cdf(cooc-1, all, tocc, socc )
    return val 
   
def comb_dicts(*dicts):
    md={}
    for d in dicts:
        for k in set(md.keys()) & set(d.keys()):
            md[k]+=d.pop(k) #.get(k, 0)
        md.update(d) #(k,  d[k]) for k in d )
    return md      


def load_tdict(f):
    d={}
#    with codecs.open(dfname,encoding="utf8") as f:
    trans=int
    for line in f:
        
        try:
            stoken, ttoken, count =line.split() 
        except ValueError:
            print >>sys.stderr, "Bad entry:", line
            continue
#            print "Looking up: %s#%s" % (stoken,ttoken)
        k=WORD_SEP.join((stoken, ttoken))
        
        try:
            d[k]=trans(count)
        except ValueError:
            trans=float
            d[k]=float(count)
            
#            print "Now dict(%s)=%lf"% (key, dict[key])
            
    return d
 
def ispos(ng, s,t,gd,sd,td,gn):
    return (gd[ng] ) >sd[s]*(float(td[t])/gn)
    

less=lambda x, y: (x-y) < EPS
less_eq=lambda x, y: (x-y) <= EPS

gt=lambda x, y: (x-y) > EPS
gt_eq=lambda x, y: (x-y) >= EPS


def tune_xi_kn(count_counts,  N_XI_PARAMS):
    y=float(count_counts[0])/(count_counts[0]+2*count_counts[1])
    
    p=arange(1, N_XI_PARAMS+1, 1)
    return p -(p+1)*y*count_counts[1:N_XI_PARAMS+1]/count_counts[:N_XI_PARAMS]
    
def tune_xi(counts, count_counts,   xif, N_XI_PARAMS, indexes=None, W0=None, xi_arg="kn"):
    
    print >>sys.stderr, "Tuning XI function..."
    if xi_arg=="wb" and N_XI_PARAMS > 1:
        print >>sys.stderr, "############### WARNING ###############"
        print >>sys.stderr,  "Using large number of parameters with"
        print >>sys.stderr, "Witten-Bell discounting may lead to"
        print >>sys.stderr, "unexpected results !!!"
        print >>sys.stderr, "#######################################"
        
    def ll_wb_multi(p,  ddp, indexes, w0, mask, xi_args, xi_mask):
        if any(p<0) or any(p>1e100):
            return  -1e100        
        try:
#            pweights=weights[gt(datapoints, pivot)]
#            pdatapoints=datapoints[gt(datapoints, pivot)]
#            d_pdatapoints=pdatapoints-pivot
#            mask=gt(ddp, 0)
            locxi=xif(xi_args,  p) #*sum(disc_w*datapoints)/sum(wdp)
            
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
            locxi[mask]=xif(xi_args[mask],  p) #*sum(disc_w*datapoints)/sum(wdp)
            
            ll1=sum(data[mask]*log(locxi[mask]))
            
            ll0=w0[0]*log(sum((1-locxi[mask])*ddp[mask]))
            ll00=sum(w0[1:]*ma.log(add.reduceat((1-locxi)*ddp, indexes)))
            return (2*ll1+ll0+ll00) #+penal
        except FloatingPointError:
            return -1e100     
#    negll_kn_multi=lambda p, ddp, i, w0, m, a: -ll_kn_multi(p, ddp, i, w0, m, a)
    
    def ll(p,  ddp,  w0,  xi_args, i0):
        if any(p<0) or any(p>1e100):
            return  -1e100        
        try:
#            pweights=weights[gt(datapoints, pivot)]
#            pdatapoints=datapoints[gt(datapoints, pivot)]
#            d_pdatapoints=pdatapoints-pivot
#            mask=gt(ddp, 0)
            
            locxi=xif(xi_args,  p) #*sum(disc_w*datapoints)/sum(wdp)
            
            ll1=sum(counts[i0:]*count_counts[i0:]*log(locxi))
            
            ll0=w0*log(sum((1-locxi)*count_counts[i0:]*ddp))
            return (ll1+ll0) #+penal
        except FloatingPointError:
            return -1e100     
    negll=lambda p, ddp, w0,  a, i0=1: -ll(p, ddp, w0, a, i0)
    i0=1
    if W0 is None:
#        pivot=1. #datapoints[0]
    #    print >>sys.stderr, "First term:", wdp[datapoints<=pivot], "sum:", sum(wdp[datapoints<=pivot])
        
        
        print >>sys.stderr, "\nThe unseen data mass will be estimated through leaving-one-out"
    #    dp=data[:, 0]
    #    w0=sum(wdp[less_eq(datapoints, pivot)])
    #    w0=sum(clip(dp, None,  pivot))
    #    w0=sum(wdp[less_eq(datapoints, pivot)]) +pivot*sum(weights[gt(datapoints, pivot)])
#        pivot=amin(data) #1.
        print >>sys.stderr, "Leaving %d from each occurrence: total number of data points: %d" %(counts[0], sum(count_counts))
      
        ddp=(counts- counts[0])[1:].astype(float)
        
        w0=count_counts[0]
#        w0=pivot*len(data) #sum(ddp<=0)
        if indexes is not None:
            W0=empty(len(indexes)+1,dtype=float)
            W0[0]=w0
            W0[1:]=pivot*add.reduceat(ddp<=0,indexes)
            w0=W0
            
    else:
        
        print >>sys.stderr, "\nThe unseen data mass is fixed (w0=",W0,")"
        ddp=counts.astype(float)
        w0=W0
        i0=0
    
#    mask=ddp > 0 #gt(ddp, 0)

    curxi=ones(N_XI_PARAMS)    
    if  xi_arg.startswith( "wb"):
        print >>sys.stderr, "Witten-Bell discouting strategy: XI depends on the ratio of unique over the total number of data points"
        
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
        print >>sys.stderr, "Kneser-Ney/Good-Turing discouting strategy: XI depends on the counts of data points"
        
        if indexes is not None:
            xi_arg=zeros_like(ddp, dtype=float)
            xi_arg[mask]=ddp[mask]**-1
            
            negll=lambda p, ddp, i, w0, m, a: -ll_kn_multi(p, ddp, i, w0, m, a)
        else:
            xi_arg=ddp**-1
        
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
    
    print >>sys.stderr, "W0=",w0 #,"Entropy deleted:", w0*log(w0), "Entrop
    xip, negllk, _, _, _, _=fmin_powell(negll,curxi,args=( ddp,   w0,  xi_arg, i0), maxiter=1000,maxfun=1000, full_output=True, disp=0)
    
        
    
    print >>sys.stderr,"Best XI parameters=", xip, "Likelihood value=", -negllk #, "augmented entropy=", sum(weights*datapoints)*ent1(xip) #, "Sum LL+ENT=", sum(weights*datapoints)*ent(xip)+llk 
#    sys.exit(0)
    return xip
            

####################
PREC=1e-5
TAU=10
def estimate_unseen(uv, ff, method="efron-thisted", args=None):    
    print >>sys.stderr, "Estimating the number of unseen ngrams; using '%s' method"%(method)
#  
#    print >>sys.stderr, "The first 5 frequencies of frequencies:", ff[:5]
#    print >>sys.stderr, "Their cumulative sum:", cumsum(ff[:5])
#    return estimate_unseen_poisson_mix_bounded(uv, ff)
    
    n=sum(uv*ff)
    un=sum(ff)
    TAU=3
    if method=="efron-thisted":
        if args is None:
            args=TAU
        elif args<0:
            args=len(ff)
        coef=-hstack((ff[:args][::-1], [0]))
       
        p=polyval(coef, -1./n) #sum(ff[:args]/(-power(-n, arange(1, args+1))))
        
        if p<=0:
            raise TypeError("Estimated Efron-Thisted probability not appropriate: %g"%(p))
        estimated=un*p/(1-p)
        print >>sys.stderr, "Estimated unseen probability:", p    
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
    elif method=="nonparametric-lindley":        
        a=(ff[1]+ff[0]/3-ff[2])/(ff[2]+ff[1]/3)        
        estimated=2*(a+1)*ff[1]-(2-a)*ff[0]
    
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
        estimated=n*(un+ff[0]*( (ff[0] -n-un + sqrt(5*n**2 +2*n*(un-3*ff[0])+(un-ff[0])**2) )/(2*n) ))/(n-ff[0])-un
    
    elif method=="mix-exp":
        estimated=un/(average(uv, weights=ff)-1)
    elif method=="good-turing":
        estimated=(1./ff[0]-1./un)**-1
#    elif method=="mix-gamma-mle":
#        estimate_unseen_poisson_mix(uv, ff)

#    elif method=="mix-gamma-mom":
#    elif method=="mix-lindley-mle":
#    elif method=="mix-lindley-mom":
#    elif method=="mix-lomax-mle":
#    elif method=="mix-lomax-mom":
    else:
        raise UnknownEstimatorError("Unknown unseen estimator: '%s'."%(method))
        
    print >>sys.stderr, "Estimated unseen number of pairs:", estimated
    
    return estimated
    
####################
    
methods={
         "cooc":cooccurrence, 
         "multinomial":multinomial_likelihood, 
         "binomial":binomial_likelihood, 
         "poisson":poisson_likelihood, 
         "poisson-stirling":poisson_stirling, 
         "hypergeometric":hypergeometric_likelihood, 
         "z-score":z_score, 
         "t-score":t_score, 
         "chi-square-independence":chi_square_ind, 
         "chi-square":chi_square, 
         "log-likelihood-ratio":llr, 
         "log-likelihood-ratio-dunning":llr_dunning, 
         "mutual-information":mi, 
         "local-mi":lmi, 
         "average-mi":ami, 
         "mi2":mi2, 
         "mi3":mi3, 
         "odds-ratio":odds_ratio, 
         "geometric-mean":gmean, 
         "arithmetic-mean":amean,
         "correlation":rho, 
#         "correlation2":rho2, 
         "norm":lnorm, 
         "dice":dice, 
         "jaccard":jaccard, 
         "fisher":fisher #exact fisher test
         
         }
         
TWO_SIDED=set([
        "multinomial", 
         "binomial", 
         "poisson", 
         "poisson-stirling", 
         "hypergeometric", 
         "chi-square-independence", 
         "chi-square",
          "log-likelihood-ratio", 
         "log-likelihood-ratio-dunning" 
         
             ])
    
CORRECTION=set([ "z-score","t-score"]) #, "poisson-stirling", "odds-ratio"]) #, "mi2", "mi3"])
BIAS=1 #e-5 
ispos=lambda cooc, socc, tocc, nlinks: cooc*nlinks > socc*tocc

def correct_neg(raw_score):
    def score(*args):
        val=raw_score(*args)
        if val<0:
            return BIAS/(1-val)
        return val+BIAS
    return score
def correct_2_sided(raw_score):
    def score(*args):
        val=BIAS+max(0, raw_score(*args) )
        if not ispos(*args) :
            return BIAS/(val)
        return val
    return score
def correct_2_sided1(raw_score):
    def score(*args):
        val=raw_score(*args)
        if ispos(*args) :
            return val 
        return -val
        
    return score
HIGHEST_XI=.9999

LOWER_ORD_DIST="lower"
UNIGRAM_DIST="unigram"
COUNT_UNIGRAM_DIST="cunigram"
UNIFORM_DIST="uniform"
NO_SMOOTH="none"
EMPTY_WORD="YTPME"
def discount_and_sum(scores_dict, counts_v, sum_dict, smoothing_dict, xi, get_ctxt, out=None, smoothing=LOWER_ORD_DIST):
    if out is None:
        out=scores_dict
    get_xi=lambda i, v: counts_v[i]
    if counts_v is None:
        get_xi=lambda i, v:xi(v)
    norm_dict={}
    if smoothing==LOWER_ORD_DIST:
        uniq={}
    else:
        uniq=None
    for i, (ng, s) in enumerate(scores_dict.iteritems()):
        ctxt=get_ctxt(ng)
        norm_dict[ctxt]=norm_dict.get(ctxt, 0)+s
        s*=get_xi(i, s) #xi(counts_dict[ng])
        out[ng]=s #scores_dict[ng]=s
        sum_dict[ctxt]=sum_dict.get(ctxt, 0)+s
        if uniq is not None:#            except KeyError:
#                continue

            uniq[ctxt]=uniq.get(ctxt, 0)+1
    for ng in out: #scores_dict:
        ctxt=get_ctxt(ng)
        out[ng]/=norm_dict[ctxt] #scores_dict[ng]/=norm_dict[ctxt]
        
    for ctxt in sum_dict:
        sum_dict[ctxt]=1-sum_dict[ctxt]/norm_dict[ctxt]
        
    if smoothing==LOWER_ORD_DIST:
        smoothing_dict.update(uniq)
    elif smoothing==UNIGRAM_DIST:
        smoothing_dict.update(norm_dict)
#    elif smoothing==UNIFORM_DIST:
#        smoothing_dict=None

def dump_probs(outf, pr, sm, get_smooth, split_ng):
#    print >>sys.stderr, "SIze of sm:", len(sm)
    for ng, p in pr.iteritems():
        ctxt, pre=split_ng(ng)
        
#        print >>sys.stderr, ctxt, ng
#        print >>sys.stderr, ctxt, "===>", sm[ctxt] #,  get_smooth(pre, ctxt)
        print >>outf, WORD_SEP.join((pre, ctxt)), "%.10g"%(p+sm[ctxt]*get_smooth(pre, ctxt))
   
def dump_probs_coocs(outf, coocs, pr, sm, get_smooth, join_ng):
    for ctxt, ts in coocs.iteritems():
        for pre in ts:
            try:
                p=pr[ join_ng(ctxt, pre)]
            except KeyError:
                p=0
            try:
                mass=sm[ctxt]
            except KeyError:
                mass=1
            
            p+=mass*get_smooth(pre, ctxt)
#            except KeyError:
#                continue
#        print >>sys.stderr, ctxt, "===>", sm[ctxt]
            print >>outf, WORD_SEP.join((pre, ctxt)), "%.10g"%(p)
   
def dump_sum(outf, sm):
    for w, s in sm.iteritems():
        
        print >>outf, WORD_SEP.join((EMPTY_WORD, w)), "%.10g"%(s)
 
def dump_smoothing(outf, sm, get_smooth):
    for w in sm:
        
        print >>outf, WORD_SEP.join((w, EMPTY_WORD)), "%.10g"%(get_smooth(w))
 
def normalize_and_print(out,  probs_0, sums_0, smoothing_1, smoothing_dist, split_ng, tvocab=None):
    tvocab.update( smoothing_1.iterkeys())
    if smoothing_dist==LOWER_ORD_DIST or smoothing_dist==UNIGRAM_DIST:
#        print >>sys.stderr, "Sum smoothing=", __builtin__.sum(smoothing_1.itervalues())

        s_norm=float(len(tvocab)*DELTA+__builtin__.sum(smoothing_1.itervalues()))
        smoothing_p_0=lambda t, *args: (smoothing_1.get(t, 0.0)+DELTA)/s_norm
#            smoothing_p_1=lambda s, *args: smoothing_0[s]/s_norm
#        else:
#            smoothing_p_1=lambda s, *args: smoothing_0[s]/s_norm
#    elif smoothing_dist==UNIGRAM_DIST:
#
#        s_norm=float(sum(gcounts.itervalues()))
#        smoothing_p_0=lambda t, *args: smoothing_1[t]/s_norm
#            smoothing_p_1=lambda s, *args: smoothing_0[s]/s_norm
#        else:
            
        
    elif smoothing_dist==UNIFORM_DIST:

        s_norm_0=(len(tvocab))
#            s_norm_1=(len(smoothing_0))
        smoothing_p_0=lambda *args: 1./s_norm_0
#            smoothing_p_1=lambda *args: 1./s_norm_1
    elif smoothing_dist==NO_SMOOTH:
        smoothing_p_0=lambda *args: 0
#            smoothing_p_1=lambda *args: 0
    else:
        raise NameError("Unknown smoothing distribution: '%s'"%(ptions.smoothing_dist))
    
    
    dump_probs(out, probs_0, sums_0, smoothing_p_0, split_ng)
    dump_sum(out, sums_0)
    dump_smoothing(out, tvocab, smoothing_p_0)
    
DELTA=1
    
def normalize_and_print_cooc(out, scooc,  probs_0, sums_0, smoothing_1, smoothing_dist, join_ng):    
#################
    
    if smoothing_dist==LOWER_ORD_DIST or smoothing_dist==UNIGRAM_DIST:
        s_norm=dict((s, __builtin__.sum(smoothing_1.get(t, 0.0)+DELTA for t in scooct)) for s, scooct in scooc.iteritems() )
        smoothing_p_0=lambda t, s, *args: float(smoothing_1.get(t, 0.0)+DELTA)/s_norm[s]
    else:
        s_norm=dict((s, len(scooct)) for s, scooct in scooc.iteritems() )
        smoothing_p_0=lambda t, s, *args: 1./s_norm[s]
    
            
    
    dump_probs_coocs(out, scooc,probs_0, sums_0, smoothing_p_0, join_ng)


def compute_assoc(gcounts, method, split_ng, out=None, penalize=False): #, scale=1):
    if out is None:
        out=gcounts
    tocc={}
    socc={}
#        assocs={}
#        if gparams["xi_func"] == "xi_kn":
#            counts_copy=fromiter(gcounts.itervalues(),dtype=int)
#        else:

    for ng, cooc in gcounts.iteritems():
        s, t=split_ng(ng)
        tocc[t]=tocc.get(t, 0)+cooc
        socc[s]=socc.get(s, 0)+cooc
    
    nlinks=__builtin__.sum(gcounts.itervalues())
#    print >>sys.stderr, "Nb links:", nlinks
#    neg_s=0
#    n1=sum(1 for i in gcounts.itervalues() if i==1)
#    print >>sys.stderr, "Scale=", scale
#    print >>sys.stderr,"Ratio of 1's:", 1-float(n1)/len(gcounts)
#
#    scale=1
#    scale=exp(7.28*(1-float(n1)/len(gcounts))**0.64 -2.16)
#    scale=exp(3.57*(1-float(n1)/len(gcounts))**2.59 -0.73)
#    scale=exp(5.103*(1-float(n1)/nlinks)-1.391)
#    scale=exp(4.679*(1-float(n1)/nlinks)**4-0.236)
#    scale=exp(4.97*(1-float(n1)/nlinks)**5-0.226)
#    scale=4.3443937189473676*(1-float(n1)/nlinks)
#    scale=5.8445085585033452*log(2-float(n1)/nlinks)
#    scale=1.4239490910805133*exp(1-float(n1)/nlinks)
#    print >>sys.stderr, "Estimated Scale=",scale
    
    assoc=methods[method]
    if method in TWO_SIDED:
        assoc=correct_2_sided(methods[method])
#    elif method in CORRECTION:
#        assoc=correct_neg(methods[method])
#        minposassoc=BIG
    
    max_s=-inf #1e100
    for ng, cooc in gcounts.iteritems():
        s, t=split_ng(ng)
        if cooc<=0:
            a=0.
        else:
            a=assoc(cooc,socc[s],tocc[t],nlinks)
#        if ng=="developments about":
#            print >>sys.stderr, "'%s'==> Cooc= %d; Socc= %d; Tocc= %d; ALL= %d; Cooc*All= %d; Tocc*Socc= %d; Assoc= %g; Uncorrected assoc= %g"%(ng, cooc,socc[s],tocc[t], nlinks, cooc*nlinks,  socc[s]*tocc[t], a, methods[method](cooc,socc[s],tocc[t],nlinks))
        aa=abs(a)         
        if aa<1e-100:
            print >>sys.stderr,"Assoc('%s','%s')= %g, Cooc= %d, Socc= %d, tocc= %d, all= %d"%(s,t,a,cooc,socc[s],tocc[t],nlinks)
            a=ZERO if a>=0 else -ZERO
#        neg_s+=cooc > socc[s]*(float(tocc[t])/nlinks)
#            ispos=cooc > socc[s]*(float(tocc[t])/nlinks)
#            if not ispos and a:
#                if method not in TWO_SIDED: # and (separate or penalize_neg) :
#                    pass
#                    a=-1./a
                
#                else: #if  method in TWO_SIDED and (not separate  or penalize_neg):
#                    a=-a
#            elif a>0 and a<minposassoc:   
#                minposassoc=a
        if penalize:
            if a <0:
                a=1./(1-a)
            
            out[ng]=cooc*a/(1+a)    
        else:
            out[ng]=a #(a, cooc)
        if aa>max_s:
            max_s=aa
#    print >>sys.stderr, "Percentage of negative scores:", float(neg_s)/nlinks
#    smax, _=max(out.itervalues())
#    for ng, (a, cooc) in out.iteritems():
#        out[ng]=(cooc*scale*a/smax/(cooc+1)+1)*cooc
    
    return out, socc, tocc, max_s
    

def compute_assoc1(gcounts, method, split_ng, out=None): #, scale=1):
    if out is None:
        out=gcounts
    wocc={}
#    socc={}
#        assocs={}
#        if gparams["xi_func"] == "xi_kn":
#            counts_copy=fromiter(gcounts.itervalues(),dtype=int)
#        else:

    for ng, cooc in gcounts.iteritems():
        s, t=split_ng(ng)
        wocc[t]=wocc.get(t, 0)+cooc
        wocc[s]=wocc.get(s, 0)+cooc
    
    nlinks=__builtin__.sum(gcounts.itervalues())
#    print >>sys.stderr, "Nb links:", nlinks
#    neg_s=0
#    n1=sum(1 for i in gcounts.itervalues() if i==1)
#    print >>sys.stderr, "Scale=", scale
#    print >>sys.stderr,"Ratio of 1's:", 1-float(n1)/len(gcounts)
#
#    scale=1
#    scale=exp(7.28*(1-float(n1)/len(gcounts))**0.64 -2.16)
#    scale=exp(3.57*(1-float(n1)/len(gcounts))**2.59 -0.73)
#    scale=exp(5.103*(1-float(n1)/nlinks)-1.391)
#    scale=exp(4.679*(1-float(n1)/nlinks)**4-0.236)
#    scale=exp(4.97*(1-float(n1)/nlinks)**5-0.226)
#    scale=4.3443937189473676*(1-float(n1)/nlinks)
#    scale=5.8445085585033452*log(2-float(n1)/nlinks)
#    scale=1.4239490910805133*exp(1-float(n1)/nlinks)
#    print >>sys.stderr, "Estimated Scale=",scale
    
    assoc=methods[method]
    if method in TWO_SIDED:
        assoc=correct_2_sided(methods[method])
#    elif method in CORRECTION:
#        assoc=correct_neg(methods[method])
#        minposassoc=BIG
    
    max_s=-inf #1e100
    for ng, cooc in gcounts.iteritems():
        s, t=split_ng(ng)
        if gcounts[ng]<=0:
            a=0.
        else:
            a=assoc(cooc,wocc[s],wocc[t],nlinks)
        aa=abs(a)         
        if aa<1e-100:
            print >>sys.stderr,"Assoc('%s','%s')= %g, Cooc= %d, Socc= %d, tocc= %d, all= %d"%(s,t,a,cooc,wocc[s],wocc[t],nlinks)
            a=ZERO if a>=0 else -ZERO
#        neg_s+=cooc > socc[s]*(float(tocc[t])/nlinks)
#            ispos=cooc > socc[s]*(float(tocc[t])/nlinks)
#            if not ispos and a:
#                if method not in TWO_SIDED: # and (separate or penalize_neg) :
#                    pass
#                    a=-1./a
                
#                else: #if  method in TWO_SIDED and (not separate  or penalize_neg):
#                    a=-a
#            elif a>0 and a<minposassoc:   
#                minposassoc=a
        
        out[ng]=a #(a, cooc)
        if aa>max_s:
            max_s=aa
#    print >>sys.stderr, "Percentage of negative scores:", float(neg_s)/nlinks
#    smax, _=max(out.itervalues())
#    for ng, (a, cooc) in out.iteritems():
#        out[ng]=(cooc*scale*a/smax/(cooc+1)+1)*cooc
    
    return out, wocc, wocc, max_s

EPSILLON=1e-50
ZERO=1e-70
def penalize_dict(d, lowest=EPSILLON):
#    minall=min(min(d[s].values()) for s in d)
    for ng, s in d.iteritems():
        if s<=0:
#                print >>sys.stderr, "Before: %g"%(d[s][t])
            d[ng]=lowest/(1-s)
#                print >>sys.stderr, "After: %g"%(d[s][t])

if __name__=="__main__":
    description= "This script aggregates partial lexical counts and produces the lexical dictionaries. If smoothing ir requested, two more tables are output containing the smoothing probability for the corresponding word"
    usage = "Usage: %prog [options]"
    parser = OptionParser(usage)
    parser.set_description(description)
    
    
    parser.add_option("-o", "--output-prefix", action="store",type="string",
                          dest="prefix",
                          help="Prefix for the output name, two dictionaries will be saved with names PREFIX.{e2f | f2e}",
                          default="lex.0-0")
    parser.add_option("-m", "--method", action="store",type="string",
                          dest="method",
                          help="Pair scoring method. Can be one  of  '%s' or the index of the method name. Default 'cooc'"%(sorted(methods.keys())),
                          default="cooc")
    
    parser.add_option("-d", "--dump-scores", action="store_true",
                          dest="dump_scores",
                          help="Save the scores in a separate file",
                          default=False)
    parser.add_option("-r", "--output-raw", action="store_true",
                          dest="output_raw",
                          help="Outtput the raw scores instead of probabilities",
                          default=False)
    parser.add_option("-n", "--separate-negative", action="store_true",
                          dest="separate",
                          help="Output negeative and positive associations to different files. By default two-sided scores (e.g. chi-square) are converted to one-sided by multiplying negative association by -1. Then positive and negative are saved to the same output.",
                          default=False)
    parser.add_option("-k", "--keep-negative-scores", action="store_true",
                          dest="keepneg",
                          help="Keep the scores uncorrected",
                          default=False)
    parser.add_option("-p", "--as-penalizer", action="store_true",
                          dest="penalize",
                          help="The association score is used to penalize the count",
                          default=False)
                          
    parser.add_option("-t", "--use-tuple-keys", action="store_true",
                          dest="use_tuple",
                          help="Use the tuple pair of words as key, instead of their concatenation",
                          default=False)
                          
    parser.add_option("-x", "--xi-func", action="store",type="string",
                      dest="xi_func",
                      help="Type of the function to be used as discount function (xi_hyp: hyperbolic; xi_log: logarithmic; xi_exp: exponential)",
                      default="xi_kn")
    
    parser.add_option("-s", "--smoothing-distribution", action="store",type="string",
                      dest="smoothing_dist",
                      help="Smoothing distribution ('unigram', 'cunigram', 'lower' order unigram, 'uniform', 'none')",
                      default=NO_SMOOTH) #LOWER_ORD_DIST)
                      
    parser.add_option("-u", "--unseen-estimator", action="store",type="string",
                      dest="unseen_estimator",
                      help="Unseen estimator ('chao', 'chao-bunge',... ), by default Leave-One-Out is used to estimate the parameters",
                      default=None)
                      
#                      
    parser.add_option("-a", "--alpha",  action="store",type="float",
                      dest="alpha",
                      help="Constant to control the influence of the association method on the cooccurrence value",
                      default=1.)

  
    parser.add_option("-c", "--cooccurrence-file", action="store",type="string",
                      dest="cooc_file",
                      help="File containing pair of words cooccurring in parallel sentences. This is used to limit the smoothing distribution",
                      default=None)
                      
  
    parser.add_option("-S", "--svocab", action="store",type="string",
                      dest="svocab",
                      help="Source vocabulary. This is used to compute the smoothing distribution (to account for the unaligned words). The total voabulary would be the union of this file and the aligned words.",
                      default=None)
  
    parser.add_option("-T", "--tvocab", action="store",type="string",
                      dest="tvocab",
                      help="Target vocabulary. This is used to compute the smoothing distribution (to account for the unaligned words). The total voabulary would be the union of this file and the aligned words.",
                      default=None)
  
    (options, args) = parser.parse_args()                      
    
    options.globals=__builtin__.sum((g.split(",") for g in args), [])
    
    if not options.globals: # and options.sources and options.targets):
        parser.error("At least one count file should be provided")
    print >>sys.stderr,"Loading the dictionaries from: %s"%(", ".join("'%s'"%f for f in options.globals))
    
    
    if options.method not in methods.keys():
        print >>sys.stderr, "Unknown scoring method: '%s',, Falling-back to 'cooc'"%(options.method)
        options.method="cooc"
    if  options.method== "cooc" and options.smoothing_dist==COUNT_UNIGRAM_DIST:
        ptions.smoothing_dist=UNIGRAM_DIST
    print >>sys.stderr, "Scoring method: '%s'"%(options.method)
    print >>sys.stderr, "Separate positive and negative associations:", "yes" if  options.separate else "no"
    print >>sys.stderr, "Correcting negative scores:", "yes" if  not options.keepneg else "no"
    print >>sys.stderr, "Type of scores:", "raw scores" if  options.output_raw else "probabilities"
    
    try:
        raise
        gcounts=comb_dicts(*(cPickle.load(open_infile(f)) for f in options.globals))
    except :
#        gcounts=comb_dicts(*(load_tdict(codecs.open(f, "r", "utf8")) for f in options.globals))        
        gcounts=comb_dicts(*(load_tdict(open_infile(f)) for f in options.globals))
#    print >>sys.stderr, "10 first entries:", list(gcounts.iteritems())[:10]
    print >>sys.stderr,"Done loading"
    if options.smoothing_dist != NO_SMOOTH:
        gparams={}
        gparams["xi_func"]=options.xi_func
        
        print >>sys.stderr, "XI function form:", gparams["xi_func"] # xi_func
        
        SMALLEST_DISC=0.0001
        LARGEST_DISC=.999
    #    N_XI_PARAMS_hyp=1
        XI_hyp=lambda x,p: (1.)/(1+x)**p
        gparams["N_XI_PARAMS"]=1
        gparams["XI"]=lambda x,p: (1.)/(1+x)**p
        
         
        if gparams["xi_func"] =="xi_hyp1":
            gparams["N_XI_PARAMS"] =1
            gparams["XI"] =lambda x,p: 1./(1+p*x)
    #        N_XI_PARAMS=2
    #        XI=lambda x,p: (1+p[1]/(1+x)**p[0])/(1+p[1])

        elif gparams["xi_func"] =="xi_kn":
            gparams["N_XI_PARAMS"] =NBR_KN_CONSTS
            gparams["XI"] =lambda x,p: (x-p[clip(atleast_1d(x), None, len(p)).astype(int)-1])/x
            
        elif gparams["xi_func"] =="xi_hyp2":
            gparams["N_XI_PARAMS"] =3
            gparams["XI"] =lambda x,p: (1)/(1+p[0]*x**p[1]+p[2])  
            
    #        N_XI_PARAMS=2
    #        XI=lambda x,p: (1.)/(1+(p[0]*x)**p[1])  

        elif gparams["xi_func"] =="xi_hyp3":
            gparams["N_XI_PARAMS"] =3
            gparams["XI"] =lambda x,p: (1+(p[1])/(1+p[0]*x)**(p[2]))/(1+(p[1]))
            
    #        N_XI_PARAMS=3
    #        XI=lambda x,p: (1+(p[1])/(1+p[0]*x)**(p[2]))/(1+(p[1]))
         
        elif gparams["xi_func"] =="xi_hyp4":
            gparams["N_XI_PARAMS"] =2
            gparams["XI"] =lambda x,p: (1.)/(1+p[0]*x**p[1])
            
    #        N_XI_PARAMS=2
    #        XI=lambda x,p: (1.)/(1+p[0]*x**p[1])


        elif gparams["xi_func"] =="xi_hyp5":
            gparams["N_XI_PARAMS"] =2
            gparams["XI"] =lambda x,p: (1.)/(1+p[0]*x**p[1])
            

    #        N_XI_PARAMS=2
    #        XI=lambda x,p: (1.)/(1+p[0]*x**p[1])
    #        lambda x,p
        elif gparams["xi_func"] =="xi_hyp6":
            gparams["N_XI_PARAMS"] =3
            gparams["XI"] =lambda x,p: (1.)/(1+p[0]*x**p[2])**p[1]
            

    #        N_XI_PARAMS=2
    #        XI=lambda x,p: (1.)/(1+p[0]*x)**p[1]
    #        
            
        elif gparams["xi_func"] =="xi_hyp7":
            gparams["N_XI_PARAMS"] =2
            gparams["XI"] =lambda x,p: (1.)/(1+p[0]*x+p[1]*x**2)
            

    #        N_XI_PARAMS=1
    #        XI=lambda x,p: (1.)/(1+p*x)
             
        elif gparams["xi_func"] =="xi_hyp8":
            gparams["N_XI_PARAMS"] =2
            gparams["XI"] =lambda x,p:(1+ (p[0])/(1+p[1]*x))/(1+p[0])
            

             
        elif gparams["xi_func"] =="xi_hyp9":
            gparams["N_XI_PARAMS"] =2
            gparams["XI"] =lambda x,p:(1./(1+p[0]*x)**p[1])
            
    #        N_XI_PARAMS=2
    #        XI=lambda x,p:(1+ (p[0])/(1+p[1]*x))/(1+p[0])
             
        elif gparams["xi_func"] =="xi_hyp_log":
            gparams["N_XI_PARAMS"] =2
            gparams["XI"] =lambda x,p: (1.)/(1+p[0]*log10(x+1))**p[1]  
            
        elif gparams["xi_func"] =="xi_hyp_hyp_log":
            gparams["N_XI_PARAMS"] =3
            gparams["XI"] =lambda x,p: (1.)/(1+p[0]*log10(x+1)+p[1]*x)**p[2]   
            
        elif gparams["xi_func"] =="xi_hyp_hyp_log2":
            gparams["N_XI_PARAMS"] =3
            gparams["XI"] =lambda x,p: ((1.)/(1+p[0]*log10(x+1))/(1+p[1]*x))**p[2]          
            
        elif gparams["xi_func"] =="xi_hyp_hyp_log3":
            gparams["N_XI_PARAMS"] =2
            gparams["XI"] =lambda x,p: ((1.)/(1+p[0]*log10(x+1))/(1+p[0]*x))**p[1]   
            
            
        elif gparams["xi_func"] =="xi_hyp_hyp_log4":
            gparams["N_XI_PARAMS"] =4
            gparams["XI"] =lambda x,p: 1./(1+p[0]*log10(x+1))**p[2]/(1+p[1]*x)**p[3]   
            
        
        elif gparams["xi_func"] =="xi_hyp_hyp_log5":
            gparams["N_XI_PARAMS"] =2
            gparams["XI"] =lambda x,p: (1.)/(1+p[0]*log10(x+1)+p[1]*x)   

        elif gparams["xi_func"] =="xi_hyp_hyp_log6":
            gparams["N_XI_PARAMS"] =2
            gparams["XI"] =lambda x,p: (1.)/(1+p[0]*log10(x+1))**p[1]/(1+p[0]*x)**p[1]   
            
        elif gparams["xi_func"] =="xi_hyp_hyp_log7":
            gparams["N_XI_PARAMS"] =4
            gparams["XI"] =lambda x,p: (1.)/(1+p[0]*log10(x+1)+p[1]*x+p[3]*x**2)**p[2]  
            
        elif gparams["xi_func"] =="xi_hyp_hyp_log8":
            gparams["N_XI_PARAMS"] =4
            gparams["XI"] =lambda x,p: (1.)/(1+p[0]*log10(x+1)+p[1]*x+p[3]*log10(x+1)**2)**p[2]  
            
        elif gparams["xi_func"] =="xi_hyp_hyp_log9":
            gparams["N_XI_PARAMS"] =4
            gparams["XI"] =lambda x,p: (1.)/(1+p[0]*log10(x+1)**p[1]+p[2]*x**p[3])
            
        elif gparams["xi_func"] =="xi_hyp_hyp_log10":
            gparams["N_XI_PARAMS"] =3
            gparams["XI"] =lambda x,p: (1.)/(1+(p[0]*log10(x+1)+p[1]*x)**p[2])
            
        elif gparams["xi_func"] =="xi_hyp_hyp_log1":
            gparams["N_XI_PARAMS"] =4
            gparams["XI"] =lambda x,p: (1.)/(1+(p[0]*log10(x+1)+p[1]*x)**p[2])**p[3]
            

    #        N_XI_PARAMS=2
    #        XI=lambda x,p: (1.)/(1+p[0]*log10(x+1))**p[1]   
           
        elif gparams["xi_func"] =="xi_hyp_log1":
            gparams["N_XI_PARAMS"] =2
            gparams["XI"] =lambda x,p: 1./(1+p[0])+(p[0])/(1+p[0])/(1+p[1]*log10(x+1))
            

    #        N_XI_PARAMS=2
    #        XI=lambda x,p: 1./(1+p[0])+(p[0])/(1+p[0])/(1+p[1]*log10(x+1))

        elif gparams["xi_func"] =="xi_hyp_log2":
            gparams["N_XI_PARAMS"] =3
            gparams["XI"] =lambda x,p: (1.)/(1+p[0]*log10(p[2]*x+1))**p[1]   
            
    #        N_XI_PARAMS=3
    #        XI=lambda x,p: (1.)/(1+p[0]*log10(p[2]*x+1))**p[1]   
           

        elif gparams["xi_func"] =="xi_hyp_log3":
            gparams["N_XI_PARAMS"] =3
            gparams["XI"] =lambda x,p: (1.)/(1+p[0]*log10(p[2]+x+1))**p[1]   
            
        elif gparams["xi_func"] =="xi_hyp_log4":
            gparams["N_XI_PARAMS"] =3
            gparams["XI"] =lambda x,p:  ( (1.)/(1+p[0]*x)+p[2]/(1+p[1]*log10(1+x)))/(1+p[2])
        
        elif gparams["xi_func"] =="xi_hyp_log5":
            gparams["N_XI_PARAMS"] =1
            gparams["XI"] =lambda x,p: (1.)/(1+p*log10(x+1))  
                
            
    #        N_XI_PARAMS=3
    #        XI=lambda x,p: (1.)/(1+p[0]*log10(p[2]+x+1))**p[1]   
              
        elif gparams["xi_func"] =="xi_exp":
            gparams["N_XI_PARAMS"] =1
            gparams["XI"] =lambda x,p: (1.)/(1+p)**(x)
            

    #        N_XI_PARAMS=1
    #        XI=lambda x,p: (1.)/(1+p)**(x)
           
        elif gparams["xi_func"] =="xi_exp1":
            gparams["N_XI_PARAMS"] =2
            gparams["XI"] =lambda x,p:  1./(1+p[0])+(p[0])/(1+p[0])/(1+p[1])**(x)
            
    #
    #        N_XI_PARAMS=2
    #        XI=lambda x,p:  1./(1+p[0])+(p[0])/(1+p[0])/(1+p[1])**(x)
           
        elif gparams["xi_func"] =="xi_exp2":
            gparams["N_XI_PARAMS"] =2
            gparams["XI"] =lambda x,p: (1.)/(1+p[0]*x)**(p[1]*x)
            
        elif gparams["xi_func"] =="xi_exp3":
            gparams["N_XI_PARAMS"] =2
            gparams["XI"] =lambda x,p: (1.)/(1+p[0])**(p[1]*x)
            
            
        elif gparams["xi_func"] =="xi_exp4":
            gparams["N_XI_PARAMS"] =3
            gparams["XI"] =lambda x,p: (1.)/(1+p[0])**(p[1]*x+p[2])
            
        elif gparams["xi_func"] =="xi_exp5":
            gparams["N_XI_PARAMS"] =3
            gparams["XI"] =lambda x,p: (1.)/(1+p[0])**(p[1]*x**p[2])
            
        elif gparams["xi_func"] =="xi_exp6":
            gparams["N_XI_PARAMS"] =4
            gparams["XI"] =lambda x,p: (1.)/(1+p[0])**(p[1]*x**p[2]+p[3])
         
        elif gparams["xi_func"] =="xi_exp7":
            gparams["N_XI_PARAMS"] =2   
            gparams["XI"] =lambda x,p: 1./(1+p[0]*exp(p[1]*x))
            
        elif gparams["xi_func"] =="xi_exp8":
            gparams["N_XI_PARAMS"] =3   
            gparams["XI"] =lambda x,p: 1./(1+p[0]*exp(p[1]*x+p[2]))
            
    #        N_XI_PARAMS=2
    #        XI=lambda x,p: (1.)/(1+p[0]*x)**(p[1]*x)
          
        elif gparams["xi_func"] =="xi_exp_log":
            gparams["N_XI_PARAMS"] =2
            gparams["XI"] =lambda x,p: (1.)/(1+p[0])**(p[1]*log10(x+1))
            

    #        N_XI_PARAMS=2
    #        XI=lambda x,p: (1.)/(1+p[0])**(p[1]*log10(x+1))
         
        elif gparams["xi_func"] =="xi_exp_log1":
            gparams["N_XI_PARAMS"] =3
            gparams["XI"] =lambda x,p: (1.)/(1+p[0])**(p[1]*log10(x+1)**p[2])
            

    #        N_XI_PARAMS=3
    #        XI=lambda x,p: (1.)/(1+p[0])**(p[1]*log10(x+1)**p[2])
            
        elif gparams["xi_func"] =="xi_exp_log2":
            gparams["N_XI_PARAMS"] =2
            gparams["XI"] =lambda x,p: (1+x)**(-p[0])
            

    #        N_XI_PARAMS=2
    #        XI=lambda x,p: (1+x)**(-p[0])
                
        elif gparams["xi_func"] =="xi_exp_log3":
            gparams["N_XI_PARAMS"] =2
            gparams["XI"] =lambda x,p: (1+x)**(-p[0]*log(1+p[1]))
            

    #        N_XI_PARAMS=2
    #        XI=lambda x,p: (1+x)**(-p[0]*log(1+p[1]))
          
         
        elif gparams["xi_func"] =="xi_exp_log4":
            gparams["N_XI_PARAMS"] =3
            gparams["XI"] =lambda x,p: (1.)/(1+p[0])**(p[1]*log10(x+1)+p[2])
            
        elif gparams["xi_func"] =="xi_exp_log5":
            gparams["N_XI_PARAMS"] =1
            gparams["XI"] =lambda x,p: (1.)/(1+p)**(log10(x+1))
                                       
        elif gparams["xi_func"] =="xi_exp_log6":
            gparams["N_XI_PARAMS"] =3
            gparams["XI"] =lambda x,p: (1.)/(1+p[0])**(p[1]*log10(x+1)**p[2])
            
        elif gparams["xi_func"] =="xi_exp_log7":
            gparams["N_XI_PARAMS"] =4
            gparams["XI"] =lambda x,p: (1.)/(1+p[0])**(p[1]*log10(x+1)**p[2]+p[3])
            
            
    #
    #        N_XI_PARAMS=3
    #        XI=lambda x,p: (1.)/(1+p[0])**(p[1]*log10(x+1)+p[2])
                               

    #
    #        N_XI_PARAMS=1
    #        XI=lambda x,p: (1.)/(1+p)**(log10(x+1))
            
        elif gparams["xi_func"] =="xi_multilog":
            gparams["N_XI_PARAMS"] =2
            gparams["XI"] =lambda x,p:  1./(1+p[0])+(p[0])/(1+p[0])/(1+p[1]*(multilog(x, 2)))
            

    #        N_XI_PARAMS=2
    #        XI=lambda x,p:  1./(1+p[0])+(p[0])/(1+p[0])/(1+p[1]*(multilog(x, 2)))

         
        elif gparams["xi_func"] =="xi_log":
            gparams["N_XI_PARAMS"] =2
            gparams["XI"] =lambda x,p: (1.)/(1+p[0]*(log10(x+1)**p[1]))
            

    #        N_XI_PARAMS=2
    #        XI=lambda x,p: (1.)/(1+p[0]*(log10(x+1)**p[1]))

        elif gparams["xi_func"] =="xi_log1":
            gparams["N_XI_PARAMS"] =1
            gparams["XI"] =lambda x,p: (1.)/(1+p*(log10(x+1)))
            

    #        N_XI_PARAMS=1
    #        XI=lambda x,p: (1.)/(1+p*(log10(x+1)))

        elif gparams["xi_func"] =="xi_log6":
            gparams["N_XI_PARAMS"] =2
            gparams["XI"] =lambda x,p: (1+p[0])/(1+p[0]+p[1]*(log10(x+1)))
            

    #        N_XI_PARAMS=2
    #        XI=lambda x,p: (1+p[0])/(1+p[0]+p[1]*(log10(x+1)))

            
        elif gparams["xi_func"] =="xi_log2":
            gparams["N_XI_PARAMS"] =2
            gparams["XI"] =lambda x,p: (1.)/(1+p[0]*(log10(p[1]+x+1)))
            

    #        N_XI_PARAMS=2
    #        XI=lambda x,p: (1.)/(1+p[0]*(log10(p[1]+x+1)))
            
        elif gparams["xi_func"] =="xi_log3":
            gparams["N_XI_PARAMS"] =2
            gparams["XI"] =lambda x,p: (1.)/(1+p[0]*(log10(p[1]*x+1)))
          
            

    #        N_XI_PARAMS=2
    #        XI=lambda x,p: (1.)/(1+p[0]*(log10(p[1]*x+1)))
          
        elif gparams["xi_func"] =="xi_log4":
            gparams["N_XI_PARAMS"] =3
            gparams["XI"] =lambda x,p: (1.)/(1+p[2]+p[0]*(log10(p[1]*x+1)))
            

    #        N_XI_PARAMS=3
    #        XI=lambda x,p: (1.)/(1+p[2]+p[0]*(log10(p[1]*x+1)))
         
        elif gparams["xi_func"] =="xi_log5":
            gparams["N_XI_PARAMS"] =3
            gparams["XI"] =lambda x,p: (1.)/(1+p[1]+p[0]*(log10(x+1))) **p[2]
            

        elif gparams["xi_func"] =="xi_hyp_exp":
            gparams["N_XI_PARAMS"] =3
            gparams["XI"] =lambda x,p: ( (1.)/(1+p[0]*x)+p[2]/(1+p[1])**x)/(1+p[2])
            
    #        N_XI_PARAMS=3
    #        XI=lambda x,p: (1.)/(1+p[1]+p[0]*(log10(x+1))) **p[2]

        seterr(all="raise")
        
        data=fromiter(gcounts.itervalues(), dtype=int)

        counts, ind=unique(data, return_inverse=True)     
        counts_counts=bincount(ind)
    #    print >>sys.stderr, "First 10 counts:", data[:10]
        minval, maxval=amin(data), amax(data)
        dfuns={}
        
        unseen_mass=None
        if options.unseen_estimator is not None:
            try:
                unseen_mass=estimate_unseen(counts, counts_counts,  method=options.unseen_estimator)
                print >>sys.stderr,"Estimated unseen pair number:", unseen_mass
            except :
                print >>sys.stderr,"Unkown estimation method, falling back to leave-one-out"
    #                raise
                unseen_mass=None
                
        if gparams["xi_func"] != "xi_kn":
            dfuns["xi_p"]=tune_xi(counts, counts_counts,  gparams["XI"], gparams["N_XI_PARAMS"], None, W0=unseen_mass, xi_arg="kn") 
            print >>sys.stderr, "Best XI params:", dfuns["xi_p"]
            dfuns["xi"]=lambda x:   clip(gparams["XI"]((1.*x)**-1, dfuns["xi_p"]), 1-HIGHEST_XI, HIGHEST_XI)
        else:
            dfuns["xi_p"]=tune_xi_kn(counts_counts, gparams["N_XI_PARAMS"]) 
                
            print >>sys.stderr, "Best XI params:", dfuns["xi_p"]
            dfuns["xi"]=lambda x:   clip(gparams["XI"]((1.*x), dfuns["xi_p"]), 1-HIGHEST_XI, HIGHEST_XI)
            
        
    
    get_tgt=lambda x:x[1]    
    get_src=lambda x:x[0]    
    split_ng=lambda x: x
    if not options.use_tuple:
        split_ng=lambda x: x.split()
        get_tgt=lambda x:x.split(None, 1)[1]    
        get_src=lambda x:x.split(None, 1)[0]    
    counts_copy=None
    method=options.method
    if method!= "cooc":
        print >>sys.stderr,"\nComputing associations "
        print >>sys.stderr,  "========================"
        counts_copy=fromiter(gcounts.itervalues(),dtype=float)
        _, socc, tocc, smax=compute_assoc(gcounts, method, split_ng, penalize=options.penalize)
    
#        print >>sys.stderr,"Correcting negative scores. Minimum positive score:", minposassoc
            
#        penalize_dict(gcounts, PENALIZE*minposassoc) 
#        reliable=1-sum(counts_copy<1.5, dtype=float)/len(counts_copy)
#        scale=exp(2.75*(reliable)+0.4)
#        scale=1
#        print >>sys.stderr,"Reliability=",reliable,"Scale=", scale
        print >>sys.stderr,"Regularization param=", options.alpha, "Normalizing constant=", smax
#        smax=max(gcounts.itervalues())
#        for x, (ng, s) in enumerate(gcounts.iteritems()):
#            cooc=float(counts_copy[x])
#            gcounts[ng]=(exp((cooc/(options.alpha+cooc))*s/smax))*cooc
#            
#    else:
#        socc={}
#        tocc={}
#            
#        for ng, score in gcounts.iteritems():
#            s, t=split_ng(ng)
#            tocc[t]=tocc.get(t, 0)+score
#            socc[s]=socc.get(s, 0)+score
        
    
#    dfuns["func"]=lambda x: x*(1-dfuns["xi"](x))  if x>0 else 0
    if options.dump_scores:
        print >>sys.stderr, "Saving the scores..."
#        f=sum(counts_copy)/__builtin__.sum(gcounts.itervalues())
#        print >>sys.stderr, "Factor=", f
        assoc_out=file(options.prefix+".scores", "wb")
        join_k=lambda x: " ".join(x)
        if not options.use_tuple:
            join_k=lambda x: x
        for i, (x, y) in enumerate(gcounts.iteritems()):
#            print >>assoc_out, join_k(x), counts_copy[i], y #*f
            print >>assoc_out, join_k(x), y #*f
        assoc_out.close()
        exit(0)
    ###########################
    
    ###########################
    print >>sys.stderr,"\nComputing probabilities "
    print >>sys.stderr,  "========================"
    if options.smoothing_dist != NO_SMOOTH:
        print >>sys.stderr,"\n1. Attributes for source-target"
        if counts_copy is not None:
            counts_copy[:]=dfuns["xi"](counts_copy)
        sums_0={}
        if options.smoothing_dist == COUNT_UNIGRAM_DIST:
            smoothing_0=socc
        else:
            smoothing_0={}
        probs_0={} #gcounts.copy()
        discount_and_sum(gcounts, counts_copy, sums_0, smoothing_0, dfuns["xi"], get_src, out=probs_0, smoothing=options.smoothing_dist)
        
#        print >>sys.stderr, "FIRST 10 items of sums:", sums_0.items()[:10]
#        exit(0)
       
        print >>sys.stderr,"\n2. Attributes for target-source"
        
        sums_1={}
        
        if options.smoothing_dist == COUNT_UNIGRAM_DIST:
            smoothing_1=tocc
        else:
            smoothing_1={}
    #    probs_1=gcounts.copy()
        discount_and_sum(gcounts, counts_copy, sums_1, smoothing_1, dfuns["xi"], get_tgt, smoothing=options.smoothing_dist)
        
        if options.smoothing_dist == COUNT_UNIGRAM_DIST:
            options.smoothing_dist = UNIGRAM_DIST
    #    print >>sys.stderr, "First 10 probs=", list(probs_0.iteritems())[:10]
    #    print >>sys.stderr, "First 10 sums=", list(sums_0.iteritems())[:10]
    #    print >>sys.stderr, "First 10 smoothing=", list(smoothing_0.iteritems())[:10]
        join_ng=lambda x, y: (x, y)
        
        print >>sys.stderr, "Writing", options.prefix+".f2e"
        f2np=file(options.prefix+".f2e", "wb")
        
        if options.cooc_file is not None:
            print >>sys.stderr, "Loading the cooccurrence file '%s'..."%(options.cooc_file )
            scooc={}
            for line in open_infile(options.cooc_file):
                s, t=line.split()
                try:
                    scooc[s].add(t)
                except KeyError:
                    scooc[s]=set([t])
            
            normalize_and_print_cooc(f2np, scooc,  probs_0, sums_0, smoothing_1, options.smoothing_dist, join_ng)
        else:
            vocab=set() #None
            if options.tvocab is not None:
                print >>sys.stderr, "Loading the target vocabulary from '%s'..."%(options.tvocab)
                vocab=set(l.split(None, 1)[0] for l in open_infile(options.tvocab))
#                for l in open_infile(options.tvocab):
#                    vocab.update(l.split())
    #       options.tvocab=set(l.split() for l in open_infile(options.tvocab))
                print >>sys.stderr, "Target vocabulary contains %d words."%(len(vocab))
            normalize_and_print(f2np, probs_0, sums_0, smoothing_1, options.smoothing_dist, split_ng, tvocab=vocab)
            if options.tvocab is not None:
                del options.tvocab
      
        del probs_0
        del sums_0
        del smoothing_1
        
        split_ng=lambda x: x[-1::-1]
        join_ng=lambda x, y:(y, x)
        
        if not options.use_tuple:
            split_ng=lambda x: list(reversed(x.split()))
            get_tgt=lambda x:x.split(None, 1)[1]    
            get_src=lambda x:x.split(None, 1)[0]    
        print >>sys.stderr, "Writing", options.prefix+".e2f"
        n2fp=file(options.prefix+".e2f", "wb")
        
        if options.cooc_file is not None:
            tcooc={}
            while True:
                try:
                    s, ts=scooc.popitem()
                except KeyError:
                    break
                for t in ts:
                    try:
                        tcooc[t].add(s)
                    except KeyError:
                        tcooc[t]=set([s])
            normalize_and_print_cooc(n2fp, tcooc, gcounts, sums_1, smoothing_0, options.smoothing_dist, join_ng)
        else:
            vocab=None
            if options.svocab is not None:
                print >>sys.stderr, "Loading the source vocabulary from '%s'..."%(options.svocab)
#                vocab=set()
                vocab=set(l.split(None, 1)[0] for l in open_infile(options.svocab))
#                for l in open_infile(options.svocab):
#                    vocab.update(l.split())
    #            options.svocab=set(l.split() for l in open_infile(options.svocab))
                print >>sys.stderr, "Source vocabulary contains %d words."%(len(vocab))
                
            normalize_and_print(n2fp, gcounts, sums_1, smoothing_0, options.smoothing_dist, split_ng, tvocab=vocab)
    else:
        
        
        print >>sys.stderr, "Writing..."
        if not     options.separate or options.method=="cooc":
            socc={}
            tocc={}
                
            for ng, score in gcounts.iteritems():
                s, t=split_ng(ng)
                tocc[t]=tocc.get(t, 0)+score
                socc[s]=socc.get(s, 0)+score
                
        if options.separate:
            print >>sys.stderr, "The negative associations..."
            nlinks=__builtin__.sum(gcounts.itervalues())
            neg_gcounts, neg_tocc, neg_socc={}, {}, {}
            ngs=gcounts.keys()
            for ng in ngs:                
                s, t=split_ng(ng)
                score=gcounts[ng]
                if score<0 or (options.method=="cooc" and not ispos(score, socc[s], tocc[t], nlinks)):
                    neg_gcounts[ng]=gcounts.pop(ng)
                    neg_tocc[t]=neg_tocc.get(t, 0)+score
                    neg_socc[s]=neg_socc.get(s, 0)+score
                    
            
            socc, tocc={}, {}                
            for ng, score in gcounts.iteritems():
                s, t=split_ng(ng)
                tocc[t]=tocc.get(t, 0)+score
                socc[s]=socc.get(s, 0)+score
            
            f2np=file(options.prefix+".n.f2e", "wb")
            n2fp=file(options.prefix+".n.e2f", "wb")
            for ng, score in neg_gcounts.iteritems():
                s, t=split_ng(ng)
                print >>n2fp, s, t, float(score)/neg_tocc[t]
                print >>f2np, t, s, float(score)/neg_socc[s]
            f2np.close()
            n2fp.close()
            print >>sys.stderr, "The positive associations..."
#            del neg_gcounts
#            del neg_tocc
#            del neg_socc
        
#        else:
        f2np=file(options.prefix+".f2e", "wb")
        n2fp=file(options.prefix+".e2f", "wb")
        
        for ng, score in gcounts.iteritems():
            s, t=split_ng(ng)
            print >>n2fp, s, t, float(score)/tocc[t]
            print >>f2np, t, s, float(score)/socc[s]
            
        
    #        
#    split_ng=lambda x: x[-1::-1]
#    print >>sys.stderr, "Writing", options.prefix+".n2f"
#    n2fp=file(options.prefix+".n2f", "wb")
#    dump_probs(n2fp, probs_1, sums_1, smoothing_p_1, split_ng)
#    dump_sum(n2fp, sums_1)
#    dump_smoothing(n2fp, smoothing_0.iterkeys(), smoothing_p_1)
