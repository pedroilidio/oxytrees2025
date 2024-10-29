#!/usr/bin/perl
# Script that downloads names of drugs and proteins from the KEGG database
# Outputs a latex table

use strict;
use lib 'scripts';
use DrugTargetInfo;
use Getopt::Long;

#-------------------------------------------------------------------------------
# Command line arguments and flags
#-------------------------------------------------------------------------------

my $MODE;
my $MAX_RANK = 99999; # Only output the top N
my $ONLY_CONFIRMED = 0;
my $ONLY_SURPRISING = 0;
$|=1;

GetOptions(
	"max-rank=i"=>\$MAX_RANK,
	"mode=s"=>\$MODE,
	'only-confirmed!'=>\$ONLY_CONFIRMED,
	'only-surprising!'=>\$ONLY_SURPRISING,
);

#-------------------------------------------------------------------------------
# Step 1: parse the input
#-------------------------------------------------------------------------------

my @lines = <>;
my $input = join '',@lines;
$input =~ s/^ans\s*=\s*[{](.*)[}]\s*$/$1/s; # Remove outer braces
$input =~ s/\r//g;
#Items are separated by whitespace
my @items;
my $rank = 1;
foreach (split /\n\s*\n/,$input) {
	last if $rank > $MAX_RANK;
	next unless /\[1,(\d+)\].*?score\s*=\s*(\S+).*?target\s*=\s*(\S+).*?drug\s*=\s*(\S+).*?near_drug_neighbor_name\s*=\s*(\S+).*?near_drug_neighbor_k\s*=\s*(\S+).*?near_target_neighbor_name\s*=\s*(\S+).*?near_target_neighbor_k\s*=\s*(\S+)/s;
	push @items, {
		rank         => $rank++,
		score        => $2,
		targKegg     => $3,
		drugKegg     => $4,
		nearDrugKegg => $5,
		nearDrugDist => $6,
		nearTargKegg => $7,
		nearTargDist => $8
	};
}

# Format from Bleakly & Yamanishi 2009
foreach (split /\n/,$input) {
	last if $rank > $MAX_RANK;
	if (/^hsa:(\d+)\t(D\d+)\t([-0-9.]+)$/) {
		push @items, {
			rank         => $rank++,
			score        => $3,
			targKegg     => "hsa$1",
			drugKegg     => $2
		};
	}
}

#-------------------------------------------------------------------------------
# Step 2: lookup names, and confirm in chembl / drugbank
#-------------------------------------------------------------------------------

foreach my $item (@items) {
	$$item{drugName}         = drug_name($$item{drugKegg});
	$$item{drugDrugbank}     = drug_drugbank($$item{drugKegg});
	$$item{targName}         = targ_name($$item{targKegg});
	$$item{targUniprot}      = targ_uniprot($$item{targKegg});
	$$item{nearDrugName}     = drug_name($$item{nearDrugKegg});
	$$item{nearTargName}     = targ_name($$item{nearTargKegg});
	#$$item{nearDrugDrugbank} = drug_drugbank($$item{nearDrugKegg});
	#$$item{nearTargUniprot}  = targ_uniprot($$item{nearTargKegg});
	
	# Try to confirm this interaction in chembl and drugbank
	$$item{inChembl}   = confirm_drugname_uniprot_chembl($$item{drugName},$$item{targUniprot});
	$$item{inDrugbank} = confirm_drugbank_uniprot($$item{drugDrugbank},$$item{targUniprot});
	$$item{inKegg}     = confirm_drug_targ_kegg($$item{drugKegg},$$item{targKegg});
	$$item{inAny}      = ($$item{inChembl} || $$item{inDrugbank} || $$item{inKegg});
	
	$$item{isSurprising} = ($$item{nearDrugDist} < 0.5 && $$item{nearTargDist} < 0.25);
}

#-------------------------------------------------------------------------------
# Step 2b: filter
#-------------------------------------------------------------------------------

if ($ONLY_CONFIRMED) {
	print STDERR "Selecting confirmed items\n";
	@items = grep {$$_{inAny}} @items;
}
if ($ONLY_SURPRISING) {
	print STDERR "Selecting surprising items\n";
	@items = grep {$$_{isSurprising}} @items;
}
@items = grep {$$_{rank} <= $MAX_RANK} @items;

#-------------------------------------------------------------------------------
# Step 2c: summarize
#-------------------------------------------------------------------------------

sub summary {
	my $n = shift;
	my @items = @_;
	
	my $out = {n => 0, num => 0, numChembl => 0, numDrugbank => 0, numKegg => 0, numAny => 0};
	foreach (@items) {
		last if $_->{rank} > $n;
		$out->{num}++;
		$out->{numChembl}++ if $_->{inChembl};
		$out->{numDrugbank}++ if $_->{inDrugbank};
		$out->{numKegg}++ if $_->{inKegg};
		$out->{numAny}++ if $_->{inAny};
	}
	sub frac { $_[1] ? $_[0]/$_[1] : 0; }
	$out->{frac}         = frac($out->{num}, $n);
	$out->{fracChembl}   = frac($out->{numChembl}, $out->{num});
	$out->{fracDrugbank} = frac($out->{numDrugbank}, $out->{num});
	$out->{fracKegg}     = frac($out->{numKegg}, $out->{num});
	$out->{fracAny}      = frac($out->{numAny}, $out->{num});
	return $out;
}

#-------------------------------------------------------------------------------
# Step 3: print the table: Latex
#-------------------------------------------------------------------------------

if ($MODE eq 'latex' || $MODE eq 'latex-wide') {
	my $WIDE = $MODE eq 'latex-wide';
	my $INCLUDE_RANK  = 1;
	my $INCLUDE_NN    = 1;
	my $INCLUDE_SCORE = $WIDE;
	
	# Header
	print "\\begin{tabular}{ll";
	print "l" if $INCLUDE_RANK;
	print "l" if $INCLUDE_NN;
	print "l" if $INCLUDE_SCORE;
	print "}\n";
	print "Rank & " if $INCLUDE_RANK;
	print "Score & " if $INCLUDE_SCORE;
	print "Pair & Description";
	print " & NN" if $INCLUDE_NN;
	print " \\\\\n";
	
	# Body
	sub latex_wrap {
		# line wrapping
		$_ = shift;
		my $sep = shift;
		if (length($_) > ($WIDE ? 65 : 45)) {
			# break string in two
			if (/^(.{25}.*?)\s([a-zA-Z(].*)$/) {
				$_ = "$1$sep$2";
			}
		}
		return $_;
	}
	foreach my $item (@items) {
		my $bold = "";
		$bold .= '\\bfseries' if $$item{inAny};
		#$bold .= '\textbf'  if $$item{inChembl};
		#$bold .= '\itshape' if $$item{inDrugbank};
		my $targName = latex_wrap($$item{targName},
		      ($INCLUDE_NN?"&":"")."\\\\\n     ".
		      ($INCLUDE_RANK?"&":"").
		      ($INCLUDE_SCORE?"&":"")."&$bold ");
		printf("\\hline\n");
		printf("$bold %d", $$item{rank}) if $INCLUDE_RANK;
		printf(" & ", $$item{rank}) if $INCLUDE_RANK;
		printf("$bold %.3f & ",$$item{score}) if $INCLUDE_SCORE;
		printf("$bold %s & $bold %s", $$item{drugKegg}, $$item{drugName});
		printf(" & %.3f",$$item{nearDrugDist}) if $INCLUDE_NN;
		printf(" \\\\\n");
		if ($$item{inAny}) {
			my @in;
			push @in,"C" if $$item{inChembl};
			push @in,"D" if $$item{inDrugbank};
			push @in,"K" if $$item{inKegg};
			printf("$bold\\indb{%s}",join(',',@in));
		}
		printf("   & ") if $INCLUDE_RANK;
		printf("   & ") if $INCLUDE_SCORE;
		printf("$bold %s & $bold %s", $$item{targKegg}, $targName);
		printf(" & %.3f",$$item{nearTargDist}) if $INCLUDE_NN;
		printf(" \\\\\n");
	}
	
	# Footer
	print "\\hline\n";
	print "\\end{tabular}\n";

#-------------------------------------------------------------------------------
# Step 3: print summary table: Latex Summary
#-------------------------------------------------------------------------------

} elsif ($MODE eq 'latex-summary') {
	my @ns = (20,50,80);
	foreach my $n (@ns) {
		$_ = summary($n, @items);
		printf(" & %d (%.f\\%%)",$_->{numAny}, 100*$_->{fracAny});
	}

#-------------------------------------------------------------------------------
# Step 3: print the table: HTML
#-------------------------------------------------------------------------------

} elsif ($MODE eq 'html') {
	# Header
	print <<EOF;
<html>
  <head>
    <style>
      .confirmN { text-align: center; color: #ccc; }
      .confirmY { text-align: center; background: #6f6; color: #000; font-weight: bold; }
      .rowY { background: #cfc; }
      .differentY { background: #ddf; }
      .results, .results th, .results td { border-collapse: collapse; border: 1px solid #888; }
      .nr td { border: 1px solid #888; margin-left:.1em; }
      th { background: #8af; }
      tr img { display: none; position: absolute; border: 1px solid black; }
      tr:hover img { display: block; }
    </style>
  </head>
<body>
  <h2>Summary</h2>
EOF
	# Top N summary
	my @ns = (10,20,50,100,200,500,1000);
	print "<table class='nr'>";
	print "<tr><td>Top #<td># selected<td># in Chembl<td># in DrugBank<td># in Kegg<td># in any DB";
	foreach my $n (@ns) {
		last if $items[-1]->{rank} < scalar @items;
		my $_ = summary($n, @items);
		printf("<tr>");
		printf("<td>%d",$n);
		printf("<td>%d (%.f%%)",$_->{num},        100*$_->{frac});
		printf("<td>%d (%.f%%)",$_->{numChembl},  100*$_->{fracChembl});
		printf("<td>%d (%.f%%)",$_->{numDrugbank},100*$_->{fracDrugbank});
		printf("<td>%d (%.f%%)",$_->{numKegg},    100*$_->{fracKegg});
		printf("<td>%d (%.f%%)",$_->{numAny},     100*$_->{fracAny});
	}
	print "</table>";

	print <<EOF;
  <h2>Ranking of Drug&mdash;Target Pairs</h2>
  <table class="results">
   <thead><tr>
    <th>Rank
    <th title="Interaction Found in ChemblDB">CH
    <th title="Interaction Found in DrugBank">DB
    <th title="Interaction Found in KEGG">KG
    <th>Drug
    <th>Target
    <th colspan=2>Most Similar Drug
    <th colspan=2>Most Similar Target
   <tbody>
EOF
	# Body
	sub yn {
		return shift() ? "Y" : "N";
	}
	sub print_drug {
		my ($kegg,$name) = @_;
		print "<a href='http://www.genome.jp/dbget-bin/www_bget?$kegg'>$name</a><br><img src='http://www.genome.jp/Fig/drug/$kegg.gif'>";
	}
	sub print_targ {
		my ($kegg,$name) = @_;
		$kegg =~ s/hsa(\d+)/$1/s;
		print "<a href='http://www.genome.jp/dbget-bin/www_bget?hsa:$kegg'>$name</a>";
	}
	foreach my $item (@items) {
		# print table cell
		printf("<tr class='row%s'>",yn($$item{inAny}));
		printf("<td>%d",$$item{rank});
		printf("<td class='confirm%s'>%s",yn($$item{inChembl}),yn($$item{inChembl}));
		printf("<td class='confirm%s'>%s",yn($$item{inDrugbank}),yn($$item{inDrugbank}));
		printf("<td class='confirm%s'>%s",yn($$item{inKegg}),yn($$item{inKegg}));
		print "<td>";
		print_drug($$item{drugKegg},$$item{drugName});
		print "<td>";
		print_targ($$item{targKegg},$$item{targName});
		printf("<td class='different%s'>%s<td>",$$item{nearDrugDist}<0.5 ? "Y' title='Considered Different'" : "N", $$item{nearDrugDist});
		print_drug($$item{nearDrugKegg},$$item{nearDrugName});
		printf("<td class='different%s'>%s<td>",yn($$item{nearTargDist}<0.25),$$item{nearTargDist});
		print_targ($$item{nearTargKegg},$$item{nearTargName});
	}
	
	# Footer
	print '</table>';
	print '</body>';
	print '</html>';

#-------------------------------------------------------------------------------
# Step 3: print the table: Raw
#-------------------------------------------------------------------------------

} elsif ($MODE eq 'raw') {
	#printf("%4s  %-30s %-30s\n", "rank", "drug", "target");
	foreach my $item (@items) {
		#printf("%d\t%-10s %-30s\n", $$item{rank},$$item{drugKegg},$$item{drugName});
		#printf("\t%-10s %-30s\n",              $$item{targKegg},$$item{targName});
		#printf("---\n");
		#printf("%4d  %-30s %-30s\n", $$item{rank},$$item{drugName},$$item{targName});
		printf("%-30s %-30s\n", $$item{drugName},$$item{targName});
	}
	
} elsif ($MODE eq 'raw-tsv') {
	foreach my $item (@items) {
		printf("%s\t%s\t%s\t%s\n", $$item{drugKegg},$$item{drugName},$$item{targKegg},$$item{targName});
	}
	
} elsif ($MODE eq 'raw-nn') {
	foreach my $item (@items) {
		printf("%d\t%f\t%f\t%f\t%d\t%d\n",
			$$item{rank},$$item{score},$$item{nearDrugDist},$$item{nearTargDist},$$item{inChembl},$$item{inDrugbank});
	}
	
} elsif ($MODE ne 'none') {
	die "Unsupported output mode: $MODE\n";
}
