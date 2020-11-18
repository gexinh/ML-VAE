% 中心化和0-1放缩，这个文本已经没用了


portion=8;
%% 处理仿真数据
eeg={};
eeg_f=cell(1,1,5);
eeg_f2=cell(1,1,5);
eeg_tm=cell(1,1,5);
m={};
s={};
me={};
max_t={};
min_t={};
for i=1:5
 %method 1
    eeg{i}=load(sprintf('eeg/o-run-0%d',i));
    [u j k]=size(eeg{i}.eeg_r);
    a=eeg{i}.eeg_r;
    eeg_s=reshape(a,[u*j,k]);
    m{i}=max(eeg_s);  %max
    mi{i}=min(eeg_s); %min
    eeg_s2=reshape(a,[u*j,60]);
    me{i}=mean(eeg_s2')'; %mean
    s{i}=std(eeg_s2')';   %std
    b=(eeg_s2-me{i})./s{i};  %处理每个通道的均值和方差，作为中心标准化数据 x-u/sigma
    eeg_f2{1,1,i}=reshape(b,[u,j,k]);
    
    max_n=reshape(repmat(m{i},[u*j,1]),[u,j,k]);
    min_n=reshape(repmat(mi{i},[u*j,1]),[u,j,k]);
    c=repmat(me{i},[1,60]);
    u2=reshape(c,[u j k]);
%     d=2*(a-min_n)./(max_n-min_n)-1;  %（-1，1）缩放处理，2*[x-min/(max-min)]-1
    d=(a-min_n)./(max_n-min_n);    %(0,1)处理
    eeg_f{1,1,i}=d;
    %减去均值的话，会导致某段趋于0的trial被拉高，从而与真实信号失真
   %考虑一下所有trail同比例放缩
    eeg_t=reshape(a,[u,j*k]);   %time course
    mm{i}=max(eeg_t')';
    mii{i}=min(eeg_t')';
    max_t{i}=reshape(repmat(mm{i},[1,k*j]),[u,j,k]);
    min_t{i}=reshape(repmat(mii{i},[1,k*j]),[u,j,k]);
    e=2*(a-min_t{i})./(max_t{i}-min_t{i})-1;
    eeg_tm{1,1,i}=e;     
    

end

save max_t max_t
save min_t min_t
%  x=reshape(d,[u,j*k]);
%  plot(x(1,:));
% X_ori=eeg_f{1,1,5};
% X_ori=reshape(X_ori,[63,2460]);
% save X X_ori
%  eeg_c=cell2mat(eeg_f);
%  eeg_c=permute(eeg_c,[3 1 2]);
%  l=length(eeg_c);
%  l1=portion/10*l;
%  l2=l1+1;
% eeg_train=eeg_c(1:l1,:,:);
% eeg_test=eeg_c(l2:end,:,:);
% save eeg_(0-1) eeg_train eeg_test
% %所有trial统一缩放
%  eeg_c=cell2mat(eeg_tm);
%  eeg_c=permute(eeg_c,[3 1 2]);
%  l=length(eeg_c);
%  l1=portion/10*l;
%  l2=l1+1;
% eeg_train=eeg_c(1:l1,:,:);
% eeg_test=eeg_c(l2:end,:,:);
% save eeg_same_sacale eeg_train eeg_test



%处理中心化的
%  eeg_c=cell2mat(eeg_f2);
%  eeg_c=permute(eeg_c,[3 1 2]);
% eeg_train=eeg_c(1:l1,:,:);
% eeg_test=eeg_c(l2:end,:,:);
% save eeg_mean eeg_train eeg_test
% %存储下均值和方差，恢复信号时使用
% save simulat_mean me
% save simulat_std  s

% %% 处理真实数据
% load EEGdata_sub-15.mat
% x=eeg_data(1:63,1:860,:);
% a=permute(x,[1,3,2]);
% [c tm tr]=size(a);
% b=reshape(a,[c*tm,tr]);
% u=mean(b')';
% s=std(b')';
% d=(b-u)./s;
% e=reshape(a,[c*tm,tr]);
% m=max(e);
% eeg_tm=(a-repmat(reshape(u,[c,tm]),[1,1,tr]))./reshape(repmat(m,[c*tm,1]),[c,tm,tr]); %(x-u)/max 
% eeg_tu=reshape(d,[c,tm,tr]);         %z-score
% eeg_tm=permute(eeg_tm,[3 1 2]);
% eeg_tu=permute(eeg_tu,[3 1 2]);
% eeg_train=eeg_tm(1:800,:,:);
% eeg_test=eeg_tm(861:end,:,:);
% save eeg_real_max eeg_train eeg_test
% eeg_train=eeg_tu(1:800,:,:);
% eeg_test=eeg_tm(801:end,:,:);
% save eeg_real_mean eeg_train eeg_test
% save real_mean u
% save real_std s