/*
 * Copyright (c) 2015, <copyright holder> <email>
 * All rights reserved.
 * 
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright
 *     notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *     notice, this list of conditions and the following disclaimer in the
 *     documentation and/or other materials provided with the distribution.
 *     * Neither the name of the <organization> nor the
 *     names of its contributors may be used to endorse or promote products
 *     derived from this software without specific prior written permission.
 * 
 * THIS SOFTWARE IS PROVIDED BY <copyright holder> <email> ''AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL <copyright holder> <email> BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 * 
 */

#include <fstream>
#include <iostream>
#include <sstream>
#include <ctime>
#include <numeric>
#include <algorithm>
#include <vector>

#include <unordered_map>
#include <iterator>

#include <assert.h>
// #include <omp.h>

#include "anyoption.h"

#define BOS "<s>"
#define EOS "</s>"
#define DEFAULT_NGRAM 4
#define DEFAULT_LOWER_NGRAM 2
typedef unsigned long icount_t;
typedef double fcount_t;
typedef pair<double,icount_t>  wcount_t;
typedef vector<string> str_vec_t;
typedef unordered_map<string,wcount_t> word_count_t;

typedef unordered_map<unsigned short,word_count_t> ng_count_t;

inline void tokenize_sentence(const string & s, str_vec_t & v,bool add_boundaries=false)
{
    string tmp;
    istringstream sent_tok(s);
    v.clear();
    if(add_boundaries)
        v.push_back(BOS);
    while(sent_tok>>tmp){
        v.push_back(tmp);
    }
    if(add_boundaries)
        v.push_back(EOS);
}

inline void count_ngrams(const str_vec_t & v, ng_count_t & wc,unsigned lower_order,unsigned higher_order, double weight=1,string sep=" ")
{
    ostringstream out;
    unsigned order;
    for(unsigned i=0;i<v.size();++i){
        for(unsigned j=i;j<v.size() && j<i+higher_order;++j){
            out<<v[j];
            order=j-i+1;
            if(order>=lower_order){
                wc[order][out.str()].first+=weight;
                wc[order][out.str()].second+=1;
            }
            out<<sep;
        }
        out.str(string());
    }
}


// double get_real(AnyOption * opt,char short_opt='t',
//                         const char * long_opt="tolerance",double default_val=TOL)
// {
//     char * th_str=opt->getValue( short_opt );
//     if(th_str==NULL)
//         th_str=opt->getValue( long_opt ) ;
// 
//     if(th_str==NULL)
//         return default_val;
//     return strtod(th_str,NULL);
// 
// }

unsigned long get_size(AnyOption * opt,char short_opt,
                        const char * long_opt,unsigned long default_val)
{
    char * sizestr=opt->getValue( short_opt );
    if(sizestr==NULL)
        sizestr=opt->getValue( long_opt ) ;

    if(sizestr==NULL)
        return default_val;
    unsigned long size=strtoul(sizestr,NULL,10);

//     if(size==0)
//         return default_val;
    return size;
}


char * get_fname(AnyOption * opt,char short_opt='o',const char * long_opt="output")
{
    char * fname=opt->getValue( short_opt );
    if(fname==NULL)
        return opt->getValue( long_opt ) ;
    return fname;
}


char * get_update_formula(AnyOption * opt)
{
    char * fname=opt->getValue( 'u' );
    if(fname==NULL)
        return opt->getValue( "update-formula" ) ;
    return fname;
}

string get_str(AnyOption * opt,char short_opt,const char * long_opt,string default_str="")
{
    char * c_str=opt->getValue( short_opt );
    if(c_str==NULL)
        c_str=opt->getValue( long_opt );
    if(c_str)
        return c_str;
    return default_str;
        
}
int main(int argc, char** argv)
{
    

    AnyOption opt;


    string usage_str("Usage: ");
    usage_str.append(argv[0]).append(" [corpus-file] [options]");
    usage_str.append("\nPerform weighted ngram penalty counting");
    
    opt.addUsage( "" );
    opt.addUsage( usage_str.c_str());
    opt.addUsage( "" );
    opt.addUsage( "Options: " );
    opt.addUsage( " -h  --help\t\tPrints this help " );

    opt.addUsage( " -o  --output\t\tOutput file name prefix (default: stdout)" );

    opt.addUsage( " -n  --ngram-order\tHighest n-gram order (default: 4)" );
    
    opt.addUsage( " -N  --lower-ngram-order\tLowest n-gram order (default: 2)" );
    
    opt.addUsage( " -B  --no-sentence-boundaries\tDo not include the sentence boundary markers (default: false)" );
        
    opt.addUsage( " -w  --weights\tRead sentence weights from file (default: None, 1 each )" );
//     opt.addUsage( " -d  --distance-measure\t\t{cosine|euclidean|manhattan|chebyshev|canberra|braycurtis|jaccard|dice|chi2|lorentzian} the distance measure to be used. (default: cosine)" );

//     opt.addUsage( " -t  --tolerance\tThe distance tolerance. We stop when the difference between successive iterations' total distance is less than this value (default: 1e-4)" );
// 
//     opt.addUsage( " -i  --init-from-file\tRead initial vectors from file (default: None, random init)" );
//     
//     opt.addUsage( " -u  --update-formula\tThe conjugate gradient update formula to compute Beta {pr : Polak–Ribière; fr : Fletcher–Reeves; hs : Hestenes-Stiefel; dy : Dai–Yuan; ls : Liu-Storey} (default: pr)" );
//     
//     opt.addUsage( " -w  --weight-function\tSelect a weighting function form {hyp : hyperbolic; pow : power; exp : exponential; hyp_log : hyperbolic logarithm; pow_log: power logarithm; exp_log: exponential logarithm} (default: original GloVe weighting)" );
//     
//     opt.addUsage( " -p  --weight-function-params\tParameters to be passed to the weighting function. Should be estimated apriori. The program exits if a weighting function was selected and no params were specified (default: None)" );
//     
//     opt.addUsage( " -t  --max-cg-trials\tMaximum number of line search trials in CG iterations (default: 15)" );
// 
// 
//     opt.addUsage( " -l  --n-line-search-iter\tMaximum number of line search iterations for each batch (default: 1)" );
//     
//     opt.addUsage( " -v  --vector-size\tThe desired vector size (default: 50)" );
//     
//     opt.addUsage( " -x  --x-max\tThe value which has to be associated with x-max in the weight function (default: 10.0)" );
//     opt.addUsage( " -a  --alpha\tThe power of the weighting function (default: 0.75)" );
//     
// #ifdef _WITH_BIAS
//     opt.addUsage( " -f  --write-full-vectors\tInclude the bias in the vectors when outputing (default: false)" );
// #endif
//     
//     opt.addUsage( " -W  --no-weighting\tUse the ordinary least squares instead of weighted LS (default: false)" );
//     
// //     opt.addUsage( " -c  --n-clusters\tThe desired number of clusters (default: 10)" );
// 
//     opt.addUsage( " -r  --learning-rate\tEta constant, which will be multiplied by the adaptive SGD AdaGrad rate (default: 0.005 for AdaGrad, 1.0 for AdaDelta)" );
//     
//     opt.addUsage( " -e  --delta-eps\tEpsilon for Adadelta learning: the constant to initialize the grad^2 and update^2 matrices (default: 1e-6)" );
//     
//     opt.addUsage( " -d  --delta-decay\tDecay constant for Adadelta learning (default: 0.95)" );
//     
//     opt.addUsage( " -T  --save-as-text\tSave the vectors as text files rather as binary (default: false)" );
// 
//     opt.addUsage( " -s  --serial\tPerform a non-parallel GD (default: false)" );
//     
//     opt.addUsage( " -E  --do-not-exp-for-weights\tDo not eponentiate while computing the weight of a given observation (default: false)" );
//     
//     opt.addUsage( " -c  --use-cg\tUse the CGD instead of SGD (default: false)" );
//     opt.addUsage( " -b  --batched\tRun CGD in batches (default: false)" );
//     opt.addUsage( " -B  --batch-size\tThe batch size (default: 10% of the data)" );
//     opt.addUsage( " -D  --adadelta-update\tUse Adadelta update rule for the SGD (default: false)" );
//     
//     opt.addUsage( " -R  --random-seed\tProvide a random seeding, so that different runs will have the same initialization (default: None)" );
//     
//     
//     opt.addUsage( " -C  --no-gradient-clip\tPrevent gradient clipping, which is activated by default (default: false)" );
//     opt.addUsage( " -g  --gradient-clip-threshold\tA positive threshold, used if gradient clipping is not prevented. The gradient value is forced into [-threshold,+threshold] (default: 10)" );
    
    
    



    opt.addUsage( "" );

    opt.setFlag( "help", 'h' );
    opt.setOption(  "weights", 'w' );
    opt.setOption(  "output", 'o' );
    opt.setOption(  "ngram-order", 'n' );
    opt.setOption(  "lower-ngram-order", 'N' );
    
    opt.setFlag( "no-sentence-boundaries", 'B' );
// #ifdef _WITH_BIAS
//     opt.setFlag( "write-full-vectors", 'f' );
// #endif
//     opt.setFlag( "serial", 's' );
//     opt.setFlag( "batched", 'b' );
//     opt.setFlag( "use-cg", 'c' );
//     opt.setFlag( "do-not-exp-for-weights", 'E' );
//     opt.setFlag( "save-as-text", 'T' );
//     opt.setFlag( "adadelta-update", 'D' );
//     opt.setFlag( "no-gradient-clip", 'C' );
//     
// //     opt.setFlag( "l1-normalize", 'l' );
// //     opt.setFlag( "l2-normalize", 'L' );
// //     opt.setFlag( "standardize", 's' );
//     opt.setOption(  "weight-function-params", 'p' );
//     //     opt.setOption(  "n-clusters", 'c' );
//     opt.setOption(  "delta-eps", 'e' );
//     opt.setOption(  "delta-decay", 'd' );
//     opt.setOption(  "x-max", 'x' );
//     opt.setOption(  "alpha", 'a' );
//     opt.setOption(  "update-formula", 'u' );
//     
//     opt.setOption(  "vector-size", 'v' );
//     opt.setOption(  "init-from-file", 'i' );
//     
//     opt.setOption(  "max-cg-trials", 't' );
//     
//     opt.setOption(  "n-line-search-iter", 'l' );
//     
//     opt.setOption(  "learning-rate", 'r' );
//     opt.setOption(  "batch-size", 'B' );
//     
//     opt.setOption(  "random-seed", 'R' );
// //     opt.setOption(  "distance-measure", 'd' );
//     
//     opt.setOption(  "tolerance", 't' );
//     opt.setOption(  "gradient-clip-threshold", 'g' );


    opt.processCommandArgs( argc, argv );


    if( opt.getFlag( "help" ) || opt.getFlag( 'h' ) ){

        opt.printUsage();        
        exit(0);
    }
   
    unsigned order=get_size(&opt,'n',"ngram-order",DEFAULT_NGRAM);
    unsigned lorder=get_size(&opt,'N',"lower-ngram-order",DEFAULT_LOWER_NGRAM);
    cerr<<"Ngram order: "<<order<<endl;
    cerr<<"Lower ngram order: "<<lorder<<endl;
    bool no_sbounds= opt.getFlag( "no-sentence-boundaries" ) || opt.getFlag( 'B' ) ;
    if(no_sbounds){
        cerr<<"Ignoring sentence boundaries"<<endl;
    }
    
    string wname=get_str(&opt,'w',"weights","");
    ifstream wfile;
    if(wname != ""){
        cerr<<"Reading sentence weights from file '"<<wname<<"'"<<endl;
        wfile.open(wname);
    }
   
    ifstream infile;
    if(opt.getArgc()>0){
        infile.open(opt.getArgv(0));
        cin.rdbuf(infile.rdbuf());
        cerr<<"Reading text from file '"<<opt.getArgv(0)<<"'..."<<endl;
    }else{
        cerr<<"Reading text from stdin..."<<endl;
    }
    
    str_vec_t tokens;
    ng_count_t counts;
    string line;
    double lweight=1;
    while(getline(cin,line)){
        if(wfile.is_open())
            wfile>>lweight;
//         cerr<<"Cur weight= "<<lweight<<endl;
        tokenize_sentence(line,tokens, !no_sbounds);
        count_ngrams(tokens,counts,lorder,order,lweight);
    }
    
    
    
    for(auto & wcounts: counts){
    
        char * oname_str=get_fname(&opt);
        ofstream outfile;
        std::streambuf *coutbuf = std::cout.rdbuf();
        
        if(oname_str){
            string oname=string(oname_str)+"."+to_string(wcounts.first);
            outfile.open(oname);
            cout.rdbuf(outfile.rdbuf());
            cerr<<"Writing to file '"<<oname<<"'..."<<endl;
        }else{
            cerr<<"Writing to STDOUT..."<<endl;
                
        }
        
        
        for(auto & p: wcounts.second){
            cout<<p.first<<" " <<p.second.first/p.second.second<<endl;
        }
        
        
        if(oname_str){
            outfile.close();
        }
            
        
        cout.rdbuf(coutbuf);
    }
//     cerr<<"Done"<<endl;
    
    
    return 0;
}