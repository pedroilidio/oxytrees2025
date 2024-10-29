
my $full = shift @ARGV;
my $withbest = 1;

# Read log of accuracy
my @entries;
$_ = join '',<>;
while (/[{](.*?)[}]/sg) {
	push @entries, $1;
}

my @funs;
push @funs, {name=>"fun = best_by09_both",                                        method=>'\bybest', kernel=>'chem/gen'};
push @funs, {name=>"fun = best_by09_auc",                                         method=>'\bybestauc', kernel=>'chem/gen'};
push @funs, {name=>"fun = best_by09_aupr",                                        method=>'\bybestaupr', kernel=>'chem/gen'};
push @funs, "cline";
push @funs, {name=>"fun = predict_with_graph_kernel(predict_rls, 1, 1)",          method=>'RLS-avg',  kernel=>'GIP'};
push @funs, {name=>"fun = predict_with_graph_kernel(predict_rls, 0, 0)",          method=>'RLS-avg',  kernel=>'chem/gen'};
push @funs, {name=>"fun = predict_with_graph_kernel(predict_rls, 0.5, 0.5)",      method=>'RLS-avg',  kernel=>'avg.'};
push @funs, "cline";
push @funs, {name=>"fun = predict_with_graph_kernel(predict_rls_kron, 1, 1)",     method=>'RLS-Kron', kernel=>'GIP'};
push @funs, {name=>"fun = predict_with_graph_kernel(predict_rls_kron, 0, 0)",     method=>'RLS-Kron', kernel=>'chem/gen'};
push @funs, {name=>"fun = predict_with_graph_kernel(predict_rls_kron, 0.5, 0.5)", method=>'RLS-Kron', kernel=>'avg.'};
push @funs, "cline" if $full;
push @funs, {name=>"fun = predict_with_correlation_kernel(predict_rls, 1, 1)",      method=>'RLS-avg', kernel=>'correlation'} if $full;
push @funs, {name=>"fun = predict_with_correlation_kernel(predict_rls_kron, 1, 1)", method=>'RLS-Kron', kernel=>'correlation'} if $full;
push @funs, "hline";

my @cols = $full
	? qw(auc aupr sensitivity_1_100 specificity_1_100 ppv_1_100)
	: qw(auc aupr);

sub number {
	if ($_[0] =~ /^\s*([0-9.]+)\s*$/) {
		return sprintf("%.1f",$1*100);
	} elsif ($_[0] =~ /^\s*([0-9.]+)\s*[(]([0-9.]+)[)]\s*$/) {
		$p1 = $1*100 >= 99.7 && $1*100 < 99.999 ? 2 : 1;
		$p2 = $2*100 <= 0.1  ? 2 : 1;
		return sprintf("\\meanstddev{%.${p1}f}{%.${p2}f}",$1*100,$2*100);
	} else {
		die "Not a number: $_[0]";
	}
}

# Output data
$llfull = $full ? 'llllllll' : 'lllll';
$cline  = $full ? '\\cline{2-8}' : '\\cline{2-5}';
print "\\begin{tabular}{$llfull}\n";
print "\\hline\n";
print "Dataset & Method & Kernel & AUC & AUPR";
print " & Sensitivity & Specificity & PPV" if $full;
print "\\aroundspace\\\\\n";
print "\\hline\n";
for $d (qw(e ic gpcr nr)) {
	my @entries_d = grep(/data = drugtarget-$d/, @entries);
	
	# find the best value for each column
	my %best;
	for $fun (@funs) {
		my @entries_match = grep(/\Q$fun->{name}\E/, @entries_d);
		next if $#entries_match < 0;
		$_ = $entries_match[0];
		foreach $col (@cols) {
			/$col = \s*([0-9.]+)/;
			$best{$col} = $1 if $best{$col} < $1;
		}
	}
	print $best;
	
	# format table rows
	my $out = "";
	my $num = 0;
	my $line_state = 'hline';
	funloop: for $fun (@funs) {
		if ($fun eq "hline") {
			$out =~ s/\\\\\s*$/\\belowspace \\\\\n\\hline\n/;
			$line_state = "hline"
		} elsif ($fun eq "cline") {
			next if $line_state ne "";
			$out =~ s/\\\\\s*$/\\sbelowspace \\\\\n$cline\n/;
			$line_state = "cline"
		} else {
			my @entries_match = grep(/\Q$fun->{name}\E/, @entries_d);
			next if $#entries_match < 0;
			$_ = $entries_match[0];
			# stuff
			$row  = "";
			$row .= " & " . $fun->{method};
			$row .= " & " . $fun->{kernel};
			foreach $col (@cols) {
				/$col = (\s*([0-9.]+)[^\n]*)/ or next funloop;
				$row .= " & " . number($1);
				$row .= "\\best" if $withbest && $2 >= $best{$col};
			}
			# table stuff
			$num++;
			$out .= $row;
			$out .= "\\abovespace"  if $line_state eq "hline";
			$out .= "\\sabovespace" if $line_state eq "cline";
			$out .= "\\\\\n";
			$line_state = "";
		}
	}
	
	print "\\multirow{$num}{*}{\\$d}\n";
	print $out;
}

print "\\end{tabular}\n";
