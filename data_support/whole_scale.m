%尝试一下将所有的trial放缩成同一个（-1 ，-1）之间
%% 处理仿真数据
eeg={};
flag=1;
whole_set=[];
portion=8
for i=1:5
 %method 1
    eeg{i}=load(sprintf('eeg/o-run-0%d',i));
    [u j k]=size(eeg{i}.eeg_r); %chan time trial
    a=eeg{i}.eeg_r;
    whole_set=cat(3,whole_set,a);
%     eeg_s=reshape(a,[u*j,k]);
%     m{i}=max(eeg_s);  %max
%     mi{i}=min(eeg_s); %min
%     
end
b=reshape(whole_set,[u,j*k*5]);
c=reshape(whole_set,[u*j,k*5]);
max_t=max(b')';
min_t=min(b')';
if flag==1
max_t=repmat(max_t,[1,j]);
min_t=repmat(min_t,[1,j]);
max_t=reshape(max_t,[u*j,1]);
min_t=reshape(min_t,[u*j,1]);
end
d=2*(c-min_t)./(max_t-min_t)-1;
e=reshape(d,[u,j,k*5]);
eeg=permute(e,[3 1 2]);

%保存矩阵

 l=length(eeg);
 l1=portion/10*l;
 l2=l1+1;
eeg_train=eeg(1:l1,:,:);
eeg_test=eeg(l2:end,:,:);
X_ori=reshape(permute(eeg_test,[2 3 1]),[u,j*k]);
save X X_ori
save max_t max_t
save min_t min_t
save eeg_same_scale eeg_train eeg_test
save X_same_scale X_ori
%

% f=reshape(e,[u,j*k*5]);
% g=e(:,:,1);
% f(1,1)
% f(1,2461)
% f(1,4921)
% f(1,7381)
% f(1,9841)