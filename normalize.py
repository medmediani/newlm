#!/usr/bin/env python

import sys
import nltk

import argparse
languages={
            "en":"english",
            "ar":"arabic"
          }


def print_err(*args, **kwargs):
    print(*args, **kwargs, file=sys.stderr)


if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Corpus tokenization and normalization",                                
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    parser.add_argument("-l","--language", type=str, default="en",
                        help="The language of the corpus")
    

    args = parser.parse_args()

    try:
        lang=languages[args.language.lower()]
    except KeyError:
        lang="english"
    print_err("Language of the corpus:",lang)
    for line in sys.stdin:
        line=" ".join(nltk.tokenize.word_tokenize(line, language=lang, preserve_line=True))
        print( line)
        
