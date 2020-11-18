
function S=calculateS(k,t)

k=10;  %最近邻数
t=1;   %分母系数

load data
options = [];
% options.Metric = 'Cosine';
% options.WeightMode = 'Cosine';
options.Metric = 'Euclidean';
options.WeightMode = 'Binary';
options.WeightMode = 'HeatKernel';
options.NeighborMode = 'KNN';
options.k = k;  % nearest neighbor
options.t = t;
ntrn=size(Y_train,1);
ntest=size(Y_test,1);
for i=1:ntest
    fmri=[Y_train;Y_test(i,:)];  %计算测试集与训练集的近似程度
    temp = constructW(fmri,options);
    temp2(:,i)=temp(1:end-1,end);
end 
temp2=full(temp2);   %Convert sparse matrix to full matrix.

S=zeros(ntrn,ntest);  %
[dump, idx] = sort(-temp2,1); % sort each row   B = sort(A,DIM) also specifies a dimension DIM to sort along.
for i=1:ntest
selectidx=idx(2:k+1,i);
S(selectidx,i)=-dump(2:k+1,i);
end
end