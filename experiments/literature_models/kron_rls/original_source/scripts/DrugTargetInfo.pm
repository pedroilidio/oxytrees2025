package DrugTargetInfo;

require Exporter;
@ISA = qw(Exporter);
@EXPORT = qw(cached drug_name targ_name targ_uniprot uniprot_interactions targ_interactions drug_drugbank drugbank_interactions targ_interactions_kegg confirm_drugname_uniprot_chembl confirm_drugbank_uniprot confirm_drug_targ_kegg); 

use strict;
use feature "state";
use DBI;
use LWP::Simple;
use English;
use File::Basename;
use Cwd 'abs_path';

#-------------------------------------------------------------------------------
# Utilities
#-------------------------------------------------------------------------------

our $cache_path = dirname(abs_path($PROGRAM_NAME)).'/../results/cache/';

sub cached {
	# Usage:
	# return cached(cache_name,cache_arg,sub{ return @stuff; });
	my ($name,$arg,$sub) = @_;
	$name =~ s/[^-_a-zA-Z0-9]//g;
	$arg  =~ s/[^-_a-zA-Z0-9]//g;
	$name = $cache_path . $name . '-' . $arg;
	mkdir $cache_path;
	# check if the cache exists
	my @out;
	if (-r $name) {
		open F, "<", $name;
		while (<F>) {
			chomp;
			s/[\n\r]//g;
			push @out, $_ if $_ ne '';
		}
		close F;
	} else {
		@out = $sub->($arg);
		open F, ">", $name;
		local $, = "\n";
		print F @out;
		close F;
	}
	return wantarray ? @out : $out[0];
}

sub no_cached {
	# Usage:
	# return cached(cache_name,cache_arg,sub{ return @stuff; });
	my ($name,$arg,$sub) = @_;
	my @out = $sub->($arg);
	return wantarray ? @out : $out[0];
}

# Get a database handle
sub dbh {
	state $dbh = DBI->connect("DBI:mysql:chembl_09","chembl","");
	return $dbh;
}

sub tablecell {
	my ($content,$query) = @_;
	$content =~ s|[<]/?div[^>]*>||g;
	$content =~ /<th.*?>$query.*?<\/th>.*?<td.*?>(.*?)<\/td>/s or return undef;
	return $1;
}

sub download_drug_image {
	my $id = $_[0];
	if (!-f "results/kegg-cache/$id.gif") {
		print STDERR "Downloading drug image $id\n";
		mkdir "results/kegg-cache";
		my $response = getstore("http://www.genome.jp/Fig/drug/$id.gif","results/kegg-cache/$id.gif");
		die "Failed to get drug image $id, response $response" unless $response==200;
	}
}

#-------------------------------------------------------------------------------
# Drug and target properties
#-------------------------------------------------------------------------------

# Find the name given a Kegg drug id
sub drug_name {
	my $id = shift;
	return '' unless $id;
	my $out = cached("drug_name",$id,sub{
		print STDERR "Getting druginfo $id\n";
		my $url = "http://www.genome.jp/dbget-bin/www_bget?$id";
		my $content = get($url);
		# Look for table cell
		my $names = tablecell($content,"Name") or die("Can't find drug Name");
		my @names = split /;?(<br>)?\n/,$names;
		foreach my $name (@names) {
			if ($name=~/\(.*?US/) {
				return $name;
			}
		}
		return $names[0];
	});
	# Do some cleanup
	$out =~ s/\s*[(][A-Z0-9,\/ ]+[)]\s*$//;
	return $out;
}

# Find the name given a Kegg target id
sub targ_name {
	my $id = shift;
	$id =~ s/hsa:?//;
	return '' unless $id;
	my $out = cached("targ_name","hsa:$id",sub{
		print STDERR "Looking up name of target hsa:$id\n";
		my $url = "http://www.genome.jp/dbget-bin/www_bget?hsa:$id";
		my $content = get($url);
		# Look for table cell
		my $gene = tablecell($content,"Gene name");
		my $defn = tablecell($content,"Definition"); # or die("Can't find Definition")
		$gene =~ s/,.*//s;
		$defn =~ s/[<]br>//s;
		if (defined($defn)) {
			return $gene . ': ' . $defn;
		} else {
			return '?';
		}
	});
	# Do some cleanup
	$out =~ s/<br>\s*//;
	chomp $out;
	$out =~ s/^(.*?:) ([^,]+), ([^,]+)-, receptor(?:, surface)?$/$1 $3 $2 receptor/;
	$out =~ s/^PTGIR:.*/PTGIR: prostaglandin I2 receptor (IP)/;
	return $out;
}

# Find the uniprot id for a Kegg target id
sub targ_uniprot {
	my $id = shift;
	$id =~ s/hsa:?//;
	return '' unless $id;
	return cached("targ_uniprot","hsa:$id",sub{
		print STDERR "Looking up uniprot for hsa:$id\n";
		my $url = "http://www.genome.jp/dbget-bin/www_bget?hsa:$id";
		my $content = get($url);
		if ($content =~ m@"http://www.uniprot.org/uniprot/([^"]+)@) {
			return $1;
		} else {
			return "";
		}
	});
}

# Find the names of all drugs that interact with a certain protein, referenced by uniprot name
sub uniprot_interactions_chembl {
#return();
	my $uniprot = shift;
	return cached("uniprot_interactions_chembl",$uniprot,sub{
		print STDERR "Looking up Chembl interactions for $uniprot\n";
		my $sth = dbh->prepare(qq{
			SELECT m.pref_name
			FROM target_dictionary AS t, assay2target AS a, activities AS ac, molecule_dictionary AS m
			WHERE a.tid=t.tid AND a.assay_id=ac.assay_id AND m.molregno=ac.molregno AND m.pref_name IS NOT NULL
			 AND t.protein_accession=?
			GROUP BY m.chembl_id
			ORDER BY m.chembl_id;
		});
		$sth->execute($uniprot);
		my @drugs;
		while (my @row = $sth->fetchrow_array) {
			push @drugs, $row[0];
		}
		return @drugs;
	});
}

# Find the names of interacting drugs, for a given kegg target id
sub targ_interactions_chembl {
	return uniprot_interactions_chembl(targ_uniprot($_[0]));
}

# Find the drugbank id for a given kegg drug id
sub drug_drugbank {
	my $id = shift;
	# hardcoded things
	return 'DB06777' if $id eq 'D00163';
	return cached("drug_drugbank",$id,sub{
		print STDERR "Getting drugbank id for $id\n";
		my $url = "http://www.genome.jp/dbget-bin/www_bget?$id";
		my $content = get($url);
		# Also download the image
		if ($content =~ m@"http://www.drugbank.ca/[^"]*?(DB[0-9]+)"@) {
			return $1;
		} else {
			print STDERR "No drugbank id found :(\n";
			return '';
		}
	});
}

# Find the uniprot ids of all interactions with a given drug
sub drugbank_interactions {
	my $dbid = shift;
	return () if $dbid eq '';
	return cached("drugbank_interactions",$dbid,sub{
		print STDERR "Looking up interactions in drugbank for $dbid\n";
		my $url = "http://www.drugbank.ca/drugs/$dbid";
		my $content = get($url);
		my @targs;
		while ($content =~ m|"http://www.uniprot.org/uniprot/([A-Z0-9]+)"|g) {
			push @targs, $1;
		}
		return @targs;
	});
}

sub targ_interactions_kegg {
	my $id = shift;
	$id =~ s/hsa:?//;
	return () unless $id;
	return cached("targ_interactions_kegg",$id,sub{
		print STDERR "Looking up interactions in KEGG for hsa:$id\n";
		my $url = "http://www.genome.jp/dbget-bin/www_bget?hsa:$id";
		my $content = get($url);
		$_ = tablecell($content,"Drug&nbsp;target");
		my @drugs;
		while (m@"\Q/dbget-bin/www_bget?dr:\E(D[0-9]+)"@g) {
			push @drugs, $1;
		}
		return @drugs;
	});
}


sub confirm_drugname_uniprot_chembl {
	my($drug_name,$targ_uniprot) = @_;
	for my $d (uniprot_interactions_chembl($targ_uniprot)) {
		return 1 if (lc($d) eq lc($drug_name));
	}
	return 0;
}

sub confirm_drugbank_uniprot {
	my($drug_drugbank,$targ_uniprot) = @_;
	for my $t (drugbank_interactions($drug_drugbank)) {
		return 1 if $t eq $targ_uniprot;
	}
	return 0;
}

sub confirm_drug_targ_kegg {
	my($drug,$targ) = @_;
	for my $d (targ_interactions_kegg($targ)) {
		return 1 if $d eq $drug;
	}
	return 0;
}
