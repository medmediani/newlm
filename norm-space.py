#!/usr/bin/env python

import sys

for line in sys.stdin:
    print(" ".join(line.strip().split()))
