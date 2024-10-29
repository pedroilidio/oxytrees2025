use strict;
use lib 'scripts';
use DrugTargetInfo;

$|=1;
foreach my $i (<>) {
	chomp $i;
	next if $i eq '';
	my $code = targ_uniprot($i);
	my @drugs = uniprot_interactions($code);
	#print "$i\t$code [",join(',',@drugs),"]\n";
	print "$i\t$code\n",join("\n",@drugs),"\n";
}
