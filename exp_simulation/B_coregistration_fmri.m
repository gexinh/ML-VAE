%% generate fMRI BOLD signals in scout areas
% resample volume of T1 to dowmsample the fMRI voxels
% delete the formmer interpolation and caculate the new one:[MRI registration]
% import the cortex named as 'cort'
% import T2 Mask as 'M'
% import T2-mean as 'T2'
% import T2-mean-copy as T2C
SNR=10;           %指定信噪比： 5 10 15
load fMRI

% % plan1  % % % f1-V1=v1l
% % % % % % % % % f2-V2=v1r
% % % % % % % % % f3-V3=v2l
% % % % % % % % % f4-V4=v2r
% atlas=cort.Atlas(8).Scouts;
% % plan2  % % %
% f1-S1: lateral occipital R
% f2-S2: rostralmiddlefrontal L
% f3-S3: rostralmiddlefrontal R
% f4-S4: supramarginal L
atlas=cort.Atlas(9).Scouts;
%% get the ROI vertices
V=cell(1,4);
for i=1:4;
    V{i}=atlas(i).Vertices;
end
% wait a second, to coregistrate the cortex and voxel first
%% cortex(eeg) and voxels(fmri) coregistration
% 载入，并生成索引
mask=cort.tess2mri_interp;    
[likelihood ind]=max(mask);   %ind: index of cortex
if isempty(find(likelihood<=0.7))
    sprintf('fully project');
end
%% 初始化数据
%T2_t:T2 time course
%T2_r:T2 reshape
%T2_non0: T2 space which is non-zero;
%fv: index of unique ROI areas in fMRI 
%fc: fmri core area intersected with T2 area.
%% index parameters
coor=cell(1,4);                   %不同源增广后的索引
fv=cell(1,4);                     %fmri unique source
fc=cell(1,4);                     %fmri core area intersected with T2 area. 
fcore_bold=cell(1,4);             %fmri core bold signals
SNR_check=cell(1,4);              %check SNR of different ROI
%% area parameters:
% plan 1
mask_t=find(M.Cube~=0);           %import Mask.hdr as 'M'
mask_o=find(M.Cube==0);           %out of mask area
% plan 2
% T2_non0=find(T2.Cube~=0);         %T2 space : non-zero space which T2 belongs to
% T2_thr=find(T2.Cube>76);          %T2 outline
% time length
t=size(fmri_b{1},2);              %time course of fMRI
[r1 r2 r3]=size(T2.Cube);
%% 生成fMRI空间
% 方案一：mask内的为噪声和fMRI，mask外的全为0
T2_r=double(reshape(T2.Cube,[r1*r2*r3,1])); 
% 区域初始化
T2_t=zeros(size(T2_r,1),t);
T2_t(mask_t,:)=T2_r(mask_t)*zeros(1,t);    %区域内为0
T2_t(mask_o,:)=T2_r(mask_o)*zeros(1,t);    %区域外为0，其实可以乘以eyes矩阵，即区域外保持原状
% 区域内加噪声
T2_t(mask_t,:)=T2_t(mask_t,:)+randn(size(T2_t(mask_t,:))); %加噪声会为0
T2_t(mask_t,:)=T2_t(mask_t,:).*0.01 ;         %缩放到0.1量级，验证可得信噪比在8.7dB左右

% 方案二：保留脑源内一部分图像，只对边缘进行匹配






%% 数据增广
for k=1:4
[x y z]=ind2sub([91 109 91],ind(1,V{k}));
coor{k}=[x;y;z]';
aug=[];
%augmentation voxels space：以每个坐标点为中心增广6个点
     for i=1:size(V{k},2)         
        for j=1:3        
           for u=coor{k}(i,j)-2:coor{k}(i,j)+2
               temp=coor{k}(i,:);
               temp(j)=u;
               aug=[aug;temp];
           end
         end
     end
aug=aug(find((0<=aug(:,1)<=r1)&(0<=aug(:,2)<=r2)&(0<=aug(:,3)<=r3)),:);
coor{k}=[coor{k};aug];
%% 生成了4个源的增广坐标coor
ind_b=sub2ind([91 109 91],coor{k}(:,1),coor{k}(:,2),coor{k}(:,3));     %返回索引值                   %返回索引位置
fv{k}=unique(ind_b);                                                 %筛选出不重复的 
%找出共同部分：intersect
[core ia ib]=intersect(fv{k},mask_t);      % 筛选出位于功能区区域内的体素
%[core1 ia ib]=intersect(fv{1},T2_thr);    % plan 2
fc{k}=core;

%% 对视觉区域分别加上指定信噪比的噪声与fMRI信号
bold=repmat(fmri_b{k},size(fc{k}));
[bold_n noise]=noisegen(bold,SNR);         %生成指定信噪比的噪声
SNR_check{k}=snr(bold(1,:),noise(1,:)) ;    %check SNR  
fcore_bold{k}=bold_n;                      %record signal;
T2_t(fc{k},:)=fcore_bold{k};               %bold with noise =bold +noise
end
clear ia ib x y z noise                    %无用信息 可以清除掉解放空间
clear i j k
%% 生成总体之后再考虑归一化 平均化 和 增幅得到4D数据
%% T2的平均影像
%第一步进行归一化处理：映射到区间[a,b]内：mapminmax（x,a,b）
%分三类，0的不映射，噪声映射到80以下，激活区域映射到0-500内
T2_tmap=zeros(size(T2_r,1),t);
sum_v=size([fc{1};fc{2};fc{3};fc{4}],1);
core=unique([fc{1};fc{2};fc{3};fc{4}]);
%T2区域内放缩：
T2_tmap(mask_t,:)=mapminmax(T2_t(mask_t,:),0,100);
%源区域内放缩：
T2_tmap(core,:)=mapminmax(T2_t(core,:),0,800);             %% 缩放比例为1：5 
%T2_tmap=diyscale(T2_t,mask_t,0,250);
%第二步进行均值化处理：
T2_mean=mean(T2_tmap,2);                          
T2C.Cube=uint8(reshape(T2_mean,[r1 r2 r3])); %保存到T2C，在副本上查看效果
% a=T2_t(mask_t,:);
% b=T2_t(core,:);
% c=T2_tmap(mask_t,:);
% d=T2_tmap(core,:);
%% T2的时间影像：T2 timecourse
T2_f=T2_tmap(:,2:t);
T2_f=reshape(T2_f,[r1,r2,r3,t-1]);
T2_ff=single(T2_f);
img=single(reshape(T2_mean,[r1 r2 r3]));

%% 最后一步：将T2C载回brainstorm，T2C import 回 mean copy
%   之后用brainstorm生成.nii文件：t2.nii
nii=load_nii('t2.nii');
nii.img=T2_ff;
nii.hdr.dime.dim=[4    r1   r2    r3   t-1     1     1     1];
nii.original.hdr.dime.dim=[4    r1   r2    r3   t-1     1     1     1];
save_nii(nii,'run-01.nii');

%% 杂乱的检查时写的程序
% c=b(T2_non0);
% mean(c);
% h=histogram(c,500);

% T2_thr=find(b>60&b<160);
% b(T2_thr)=0;
% T2C.Cube=b;
% [core ia ib]=intersect(fv{1},T2_thr); 

% b(mask_t)=0;
% T2C.Cube=b;
%  [core ia ib]=intersect(fv{1},mask_t);

% 最初的
%              new_coor=coor{k}(i,:); 
%              if ((j==1||j==3)&&coor{k}(i,j)~=91)||(coor{k}(i,j)~=109&&j==2) 
%                new_coor(j)=coor{k}(i,j)+1;
%              end
% %              if ((j==1||j==3)&&coor{k}(i,j)~=90)||(coor{k}(i,j)~=108&&j==2) 
% %                new_coor(j)=coor{k}(i,j)+2;
% %              end
%              if coor{k}(i,j)~=0
%                new_coor(2,:)=new_coor(1,:);
%                new_coor(2,j)=coor{k}(i,j)-1;
%              end
%              coor{k}=[coor{k}; new_coor];
%          end
