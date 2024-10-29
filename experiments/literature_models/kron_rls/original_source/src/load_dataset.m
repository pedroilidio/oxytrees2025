
function data = load_dataset(name)
	
	if isequal(name,'disease')
		% Human disease network
		load ../dataset/disease/diseases.octave.txt
		data.name = 'disease';
		data.y = full(X);
		data.l1 = {};
		data.l2 = {};
		data.k1 = eye(size(data.y,1));
		data.k2 = eye(size(data.y,2));
		return;
	
	% Make additional datasets by resizing
	elseif isequal(name,'ic-40')
		% Only the first 40 drugs/targets of the ic dataset
		data = load_drugtarget_dataset('ic');
		data.name = [data.name '-30'];
		na = 1:40; nb = 1:40;
	elseif isequal(name,'ic-30-210')
		% ic dataset, but 30 targets and 210 drugs
		data = load_drugtarget_dataset('ic');
		data.name = [data.name '-30-210'];
		na = 1:30; nb = 1:210;
	elseif isequal(name,'ic-nnr')
		% ic dataset, but same size as nr
		data = load_drugtarget_dataset('ic');
		data.name = [data.name '-nnr'];
		na = 1:26; nb = 1:54;
	elseif isequal(name,'e-40')
		% Only the first 40 drugs/targets of the e dataset
		data = load_drugtarget_dataset('e');
		data.name = [data.name '-40'];
		na = 1:40; nb = 1:40;
	elseif isequal(name,'e-80')
		% Only the first 80 drugs/targets of the e dataset
		data = load_drugtarget_dataset('e');
		data.name = [data.name '-80'];
		na = 1:80; nb = 1:80;
	elseif isequal(name,'e-200')
		% Only the first 200 drugs/targets of the e dataset
		data = load_drugtarget_dataset('e');
		data.name = [data.name '-200'];
		na = 1:200; nb = 1:200;
	else
		[~,~,~,~,ma] = regexp(name,'(p|c|pc)-(e|ic|gpcr|nr)');
		if length(ma)>0
			data = load_drugtarget_pharmaco_dataset(ma{1}{2},ma{1}{1});
		else
			data = load_drugtarget_dataset(name);
		end
		return
	end
	
	% Resize
	data.y = data.y(na,nb);
	data.k1 = data.k1(na,na);
	data.k2 = data.k2(nb,nb);
	data.l1 = data.l1(na);
	data.l2 = data.l2(nb);
end

function data = load_drugtarget_dataset(name)
	% The name must be one of {'nr','gpcr','ic','e'}
	path = '../dataset/drugtarget/';
	
	[y,l1,l2] = load_drugtarget([path name '_admat_dgc.txt']);
	data.y = y;
	data.dim1 = 'proteins';
	data.dim2 = 'drugs';
	data.l1 = l1;
	data.l2 = l2;
	data.s1 = load_drugtarget([path name '_simmat_dg.txt']);
	data.s2 = load_drugtarget([path name '_simmat_dc.txt']);
	data.k1 = data.s1;
	data.k2 = data.s2;
	data.name = ['drugtarget-' name];
	% The kernels are not always symmetric!
	% make them symmetric now.
	data.k1 = (data.k1+data.k1')/2;
	data.k2 = (data.k2+data.k2')/2;
	% And sometimes the kernels are also singular or otherwise not positive definite
	% so add a very small constant
	e1 = max(0, -min(eig(data.k1)) + 1e-4);
	e2 = max(0, -min(eig(data.k2)) + 1e-4);
	data.k1 = data.k1 + e1*eye(length(data.k1));
	data.k2 = data.k2 + e2*eye(length(data.k2));
end

function data = load_drugtarget_pharmaco_dataset(name,type)
	% The name must be one of {'nr','gpcr','ic','e'}
	% type must be one of {'p','c','pc'}
	path = '../dataset/drugtarget-pharmaco/';
	
	[y,l1,l2] = load_drugtarget([path name '_Amat.txt']);
	data.y = y;
	data.dim1 = 'proteins';
	data.dim2 = 'drugs';
	data.l1 = l1;
	data.l2 = l2;
	data.s1 = load_drugtarget([path name '_Gmat.txt']);
	data.k1 = data.s1;
	data.sChem  = load_drugtarget([path name '_Cmat.txt']);
	data.sPharm = load_drugtarget([path name '_Pmat.txt']);
	if isequal(type,'c')
		data.k2 = data.sChem;
	elseif isequal(type,'p')
		data.k2 = data.sPharm;
	elseif isequal(type,'pc')
		data.k2 = (data.sChem + data.sPharm) / 2;
	else
		error('Unknown type: %s', type);
	end
	data.name = ['drugtarget-pharmaco-' type '-' name];
	% The kernels are not always symmetric!
	% make them symmetric now.
	data.k1 = (data.k1+data.k1')/2;
	data.k2 = (data.k2+data.k2')/2;
	% And sometimes the kernels are also singular or otherwise not positive definite
	% so add a very small constant
	e1 = max(0, -min(eig(data.k1)) + 1e-4);
	e2 = max(0, -min(eig(data.k2)) + 1e-4);
	data.k1 = data.k1 +e1*eye(length(data.k1));
	data.k2 = data.k2 + e2*eye(length(data.k2));
end

function [data,rows,cols]=load_drugtarget(filename)
	% Load a table in the format of the drug-target interaction data provided by Yamanishi et al.
	%
	% This is a numeric table with row and column headings
	
	f = fopen(filename,'rt');
	
	line = fgetl(f);
	cols = break_list(line,line==9); % 9 = "\t"
	cols(1) = []; % first column for row labels
	
	rows = cell(0);
	data = zeros(0);
	while ~feof(f)
		line = fgetl(f);
		if numel(line) <= 1, break; end
		pos = find(line==9,1);
		rows = cat(1,rows,{line(1:pos-1)});
		data = cat(1,data,sscanf(line(pos:end),'%f')');
	end
	
	fclose(f);
end

function out=break_list(data,seps)
	% Break a list into parts
	% separators are the points where seps==1
	sep_pos = [0 find(seps) numel(data)+1];
	nparts = numel(sep_pos)-1;
	out = cell(1,nparts);
	for i=1:nparts
		out{i} = data(sep_pos(i)+1:sep_pos(i+1)-1);
	end
end
