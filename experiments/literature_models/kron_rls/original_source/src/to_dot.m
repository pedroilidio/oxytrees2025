function to_dot(y)
	if 1
		[f,msg]=fopen("bloe.dot",'w');
		fprintf(f,"graph G{\n");
		fprintf(f,"edge[color=gray75];\n");
		for i=1:size(y,1)
			for j=1:size(y,2)
				if y(i,j) >= 2
					fprintf(f,"a%d -- b%d [color=darkgreen,width=3];\n",i,j);
				elseif y(i,j)
					fprintf(f,"a%d -- b%d;\n",i,j);
				end
			end
		end
		for i=1:size(y,1)
			fprintf(f,"a%d [shape=point,color=blue];\n",i);
		end
		for i=1:size(y,2)
			fprintf(f,"b%d [shape=point,color=red];\n",i);
		end
		fprintf(f,"}\n");
		fclose(f);
	else
		[f,msg]=fopen("bloe.txt",'w');
		fprintf(f,"%d\n",size(y,1)+size(y,2));
		fprintf(f,"2\n");
		for i=1:size(y,1)
			fprintf(f,"1 \n");
		end
		for j=1:size(y,2)
			fprintf(f,"2 \n");
		end
		fprintf(f,"%d\n",nnz(y));
		for i=1:size(y,1)
			for j=1:size(y,2)
				if (y(i,j)), fprintf(f,"%d %d\n",i,j); end
			end
		end
		fclose(f);
	end
end
