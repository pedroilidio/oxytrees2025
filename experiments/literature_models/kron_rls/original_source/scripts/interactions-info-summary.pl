#!/usr/bin/perl
# Script that builds a summary table by repeatedly calling intersctions-info.pl

use strict;
use lib 'scripts';

#-------------------------------------------------------------------------------
# Simple hackish script
#-------------------------------------------------------------------------------

print "\\begin{tabular}{llll}\n";
print "\\hline\n";
print "Dataset & Top 20 & Top 50 & Top 80 \\\\\n";
print "\\hline\n";
for (@ARGV) {
	if (/[-_]([a-z]+?).txt/) {
		print "{\\$1}";
		print `perl scripts/interactions-info.pl --mode=latex-summary < $_`;
		print "\\\\\n";
	}
}
print "\\hline\n";
print "\\end{tabular}\n";
