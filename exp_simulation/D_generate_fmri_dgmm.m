SNR=1;           %指定信噪比： 5 10 15
load fMRI
% import cortex_8195v as cort 
% % plan1  % % % f1-V1=v1l
% % % % % % % % % f2-V2=v1r
% % % % % % % % % f3-V3=v2l
% % % % % % % % % f4-V4=v2r
%atlas=cort.Atlas(8).Scouts;

% % plan2  % % %
% f1-S1: lateral occipital R
% f2-S2: rostralmiddlefrontal L
% f3-S3: rostralmiddlefrontal R
% f4-S4: supramarginal L
atlas=cort.Atlas(9).Scouts;
%% get the ROI vertices
V=cell(1,4);
SNR_check=cell(1,4);
for i=1:4
    V{i}=atlas(i).Vertices;
end
% % 建立空表：
v=size(cort.Vertices,1);
t=size(fmri_b{1},2);
v_t=zeros(v,t);
for j=1:5
for i=1:4
    bold=repmat(fmri_b{i},size(V{i}'));
    [bold_n noise]=noisegen(bold,SNR);  
    SNR_check{i}=snr(bold_n(1,:),noise(1,:)) ;
%     v_t(V{i},:)=repmat(bold_n,size(V{i}')); %这样做会让相同区域的体素信号一致
   
%考虑每个体素分别加上噪音：
     v_t(V{i},:)=bold_n;
end

ind=find(v_t(:,1)~=0);
fmri_bold=v_t(ind,:);
% save(['fmri_vertice',num2str(v) ,'.mat'],'fmri_bold');
save(['run-0',num2str(j) ,'.mat'],'fmri_bold');
end