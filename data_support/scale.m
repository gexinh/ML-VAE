%scale
% cell(1,1,5)
eeg={};
eeg_f=cell(1,1,5);
m={};
fmri=cell(1,5);
for i=1:5
 %method 1
    eeg{i}=load(sprintf('eeg/o-run-0%d',i));
    [u j k]=size(eeg{i}.eeg_r);
    a=eeg{i}.eeg_r;
    eeg_s=reshape(a,[1,u*j*k]);
    m{i}=max(eeg_s);
    eeg_f{1,1,i}=a/m{i};
 %method 2
    %eeg{i}=load(['eeg/o-run-0',num2str(i)]);
%     eeg{1,1,i}=eeg{1,1,i}.eeg_r;
%     fmri{1,i}=load(['./fun/run-0',num2str(i)]);
%     fmri{1,i}=fmri{1,i}.fmri_bold;
end
% 
% eeg_c=cell2mat(eeg_f);
% eeg_c=permute(eeg_c,[3 1 2]);
% fmri_c=permute(fmri_c,[2 1]);

% em=reshape(eeg,[1,860*63*70]);
% mean(em)
% var(em)
% m=max(em);
% eeg_c=eeg/m;

eeg_train=eeg_c(1:240,:,:);
eeg_test=eeg_c(241:end,:,:);
save eeg_s eeg_train eeg_test

% es.F=reshape(eeg_f{1,1,1},[63,2460]);