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

#include <unordered_set>

#include <assert.h>

#include "anyoption.h"

#define BOS "<s>"
#define EOS "</s>"
#define UNK "<unk>"
#define DEFAULT_NGRAM 4
typedef unsigned long icount_t;
typedef double fcount_t;
typedef pair<double,icount_t>  wcount_t;
typedef vector<string> str_vec_t;
typedef unordered_map<string,icount_t> word_count_t;
typedef unordered_set<string> word_list_t;

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

inline void count_ngrams(const str_vec_t & v, word_count_t & wc,unsigned order, string sep=" ")
{
    ostringstream out;
    for(unsigned i=0;i<v.size();++i){
        for(unsigned j=i;j<v.size() && j<i+order;++j){
            out<<v[j];
            wc[out.str()]+=1;
            out<<sep;
        }
        out.str(string());
    }
}



inline void count_ngrams(const str_vec_t & v, word_count_t & wc,unsigned order, const word_list_t& vocab, bool unk_rep=false,string sep=" ")
{
    ostringstream out;
    for(unsigned i=0;i<v.size();++i){
        for(unsigned j=i;j<v.size() && j<i+order;++j){
             if(vocab.find(v[j]) == vocab.end()){
                 if(unk_rep){
                     out<<UNK;
                }else
                    break;
            }else
                out<<v[j];
            wc[out.str()]+=1;
            out<<sep;
        }
        out.str(string());
    }
}


unsigned long get_size(AnyOption * opt,char short_opt,
                        const char * long_opt,unsigned long default_val)
{
    char * sizestr=opt->getValue( short_opt );
    if(sizestr==NULL)
        sizestr=opt->getValue( long_opt ) ;

    if(sizestr==NULL)
        return default_val;
    unsigned long size=strtoul(sizestr,NULL,10);

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
    usage_str.append("\nPerform weighted ngram counting");
    
    opt.addUsage( "" );
    opt.addUsage( usage_str.c_str());
    opt.addUsage( "" );
    opt.addUsage( "Options: " );
    opt.addUsage( " -h  --help\t\tPrints this help " );

    opt.addUsage( " -o  --output\t\tOutput file name (default: stdout)" );

    opt.addUsage( " -v  --vocab\t\tVocabulary file name (default: none, all corpus words)" );
    opt.addUsage( " -u  --unk-mark\t\tReplace unknown words with <unk>, rather than ignoring them (default: false)" );

    opt.addUsage( " -n  --ngram-order\tHighest n-gram order (default: 4)" );
    
    opt.addUsage( " -B  --no-sentence-boundaries\tDo not include the sentence boundary markers (default: false)" );
        


    opt.addUsage( "" );

    opt.setFlag( "help", 'h' );

    opt.setOption(  "output", 'o' );
    opt.setOption(  "ngram-order", 'n' );
    
    opt.setOption(  "vocab", 'v' );
    
    opt.setFlag( "unk-mark", 'u' );
    opt.setFlag( "no-sentence-boundaries", 'B' );
 

    opt.processCommandArgs( argc, argv );


    if( opt.getFlag( "help" ) || opt.getFlag( 'h' ) ){

        opt.printUsage();        
        exit(0);
    }
   
    unsigned order=get_size(&opt,'n',"ngram-order",DEFAULT_NGRAM);
    cerr<<"Ngram order: "<<order<<endl;
    bool no_sbounds= opt.getFlag( "no-sentence-boundaries" ) || opt.getFlag( 'B' ) ;
    if(no_sbounds){
        cerr<<"Ignoring sentence boundaries"<<endl;
    }
    
   
    
    char * vocab_fname=get_fname(&opt,'v',"vocab");
    word_list_t voc;
    
    if(vocab_fname){
        ifstream vocab_f(vocab_fname);
        cerr<<"Vocabulary: "<<vocab_fname<<endl;
        copy(istream_iterator<string>(vocab_f),
         istream_iterator<string>(),
         inserter(voc, voc.end()));
        cerr<<"Vocabulary size: "<<voc.size()<<endl;
        voc.insert(BOS);
        voc.insert(EOS);
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
    word_count_t counts;
    string line;

    if(vocab_fname){
        
        while(getline(cin,line)){
            tokenize_sentence(line,tokens, !no_sbounds);
            count_ngrams(tokens,counts,order,voc,opt.getFlag( "unk-mark" ) || opt.getFlag( 'u' ));
            
        }
        for(auto & w: voc){
            counts[w]+=0;
        }
    
    }else{
        while(getline(cin,line)){
            tokenize_sentence(line,tokens, !no_sbounds);
            count_ngrams(tokens,counts,order);
            
        }
    }
    
    
    char * oname=get_fname(&opt);
    ofstream outfile;
    std::streambuf *coutbuf = std::cout.rdbuf();
    
    if(oname){
        outfile.open(oname);
        cout.rdbuf(outfile.rdbuf());
        cerr<<"Writing to file '"<<oname<<"'..."<<endl;
    }else{
        cerr<<"Writing to STDOUT..."<<endl;
            
    }
    
    
   for(auto & p: counts){
   	 cout<<p.first<<" " <<p.second<<endl;
   }
    
    
    if(oname){
        outfile.close();
    }
    
    cout.rdbuf(coutbuf);
    
    
    return 0;
}
