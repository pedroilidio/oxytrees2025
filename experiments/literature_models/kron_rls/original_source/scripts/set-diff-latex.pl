use strict;

sub lines {
	open my $fh, "<", $_[0];
	my @out = <$fh>;
	close $fh;
	return @out;
}

my @a = (lines($ARGV[0]));
my @b = (lines($ARGV[1]));

my %a = map { $_ => 1 } @a;
my %b = map { $_ => 1 } @b;


sub print_line {
	$_ = shift;
	/^(.+?)\t(.+?)\t(.+?)\t(.+?)$/;
	#print "$1 & $2 \\\\\n";
	print "$1 & $2 &\n";
	print "$3 & $4 \\\\\n";
	print "\\hline\n";
}
sub print_start {
	#print "\\subsection{$_[0]}\n";
	print "\\diffHeading{$_[0]}\n";
	print "\\begin{tabular}{lp{37mm}|lp{73mm}}\n";
	print "\\hline\n";
}
sub print_end {
	print "\\end{tabular}\\\\\n"
}

print_start("Interactions found by both methods");
for (@a) {
	next unless $b{$_};
	print_line($_);
}
print_end();
print_start("Interactions only found by our method");
for (@a) {
	next if $b{$_};
	print_line($_);
}
print_end();
print_start("Interactions only found by Bleakley \\& Yamanishi");
for (@b) {
	next if $a{$_};
	print_line($_);
}
print_end();
