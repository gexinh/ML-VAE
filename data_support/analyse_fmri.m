%分析原始的fMRI数据的特点
load fmri_averige.mat
c=fmri_test;
d=fmri_train;
a=std(fmri_test); %查看不同体素之间的方差
figure(1)
hold on
plot(a);%发现放缩之后不同体素之间的方差基本收敛在一定范围呢
title('the std of different voxels of testing set with orignal ')
%% 放缩testing set
%放缩到一个级别后再查看标准差
max_t=max(c);
min_t=min(c);
s=(c-min_t)./(max_t-min_t);  %scale to (0-1)
% figure(1)
% hold on
% plot(s);
s_s=std(s);
figure(2)
hold on
plot(s_s);%发现放缩之后不同体素之间的方差基本收敛在一定范围呢
title('the std of different voxels of testing set with (0-1) scale')
%% 放缩training set
max_t=max(d);
min_t=min(d);
s=(d-min_t)./(max_t-min_t);  %scale to (0-1)
% figure(1)
% hold on
% plot(s);
s_s=std(s);
figure(3)
hold on
plot(s_s);%训练集的方差也同样收敛在一定范围内
title('the std of different voxels of training set with (0-1) scale')
%% 使用极大似然得到测试集体素标准差的分布

N=length(s_s);
u=sum(s_s)/N;
sigma=sum((s_s-u).^2)./(N-1);
x = -1:0.1:1;
y = gaussmf(x,[u sigma]);
figure(4)
hold on
plot(x,y)
xlabel(['mean=',num2str(u),'var=',num2str(sigma)])
title('the distribution of std of training set voxels')
%可以看出每个体素的标准差十分的接近。


