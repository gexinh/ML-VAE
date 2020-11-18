% project the 3D coor into 2D coor
% import channel file as 'calp'
% import cortex signals as 'cs'
chan=cs.GoodChannel;
l=length(calp.Channel);
xyz=zeros(l,3);
name=string([]);  %空字符串
 for i=1:l
xyz(i,:)=calp.Channel(i).Loc;
name=[name ;calp.Channel(i).Name];
 end
 [X,Y] = bst_project_2d(xyz(:,1), xyz(:,2), xyz(:,3),'2dcap');
cap=[X Y name];
cap=cap(chan,:);
[r,c]=size(cap);            % 得到矩阵的行数和列数
fid = fopen('65.txt','w');
for i=1:r
    for j=1:c
        if j <=2
            fprintf(fid,'%f\t',str2num(cap(i,j)));
        end
        if j==3
            fprintf(fid,'%s\t',cap(i,j));
        end
    end
    fprintf(fid,'\r\n');
 end
fclose(fid);