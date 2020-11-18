%decrease the time dimension of fmri 
load fmri;
%每4秒一个trial，将数据切分成60*4*876
s=["train","test"];
tmc=4;
for i=1:length(s)
a=eval(['fmri_',char(s(i))]);    %记住
[t v]=size(a);

a=reshape(a,[tmc,t/tmc,v]);
a=permute(a,[1 3 2]);
% b=reshape(a(:,:,1),[t/trial,v]);
%先尝试一下每个使用均值化的方法，即每段trial做一次均值处理
m=mean(a);
m=reshape(m,[v,t/tmc])';     % m即是均值化后所得的fMRI 
eval([['fmri_',char(s(i))],'=','m',';']);
end
save fmri_averige fmri_test fmri_train