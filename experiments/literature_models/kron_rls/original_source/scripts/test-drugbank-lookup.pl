use strict;
use DrugTargetInfo;

my $db = drug_drugbank("D00195");
print "$db: ";
my @drugs = drugbank_interactions($db);
print "$_ " for (@drugs);
