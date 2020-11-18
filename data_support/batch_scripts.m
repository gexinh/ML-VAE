load max_t 
load min_t
% 前四个拼接成train_set ，最后一个拼接成test_set
max_train_set=[];
min_train_set=[];
for i=1:5
%     train_set=max_t{i};
    if i~=5
    max_train_set=cat(3,max_train_set,max_t{i});
    min_train_set=cat(3,min_train_set,min_t{i});
    else
    max_test_set=max_t{i};
    min_test_set=min_t{i};
    end
end
s1=["min","max"];
s2=["train","test"];
for i=1:2
    for j=1:2
    eval([char(s1(i)),'_',char(s2(j)),'_set=permute(',char(s1(i)),'_',char(s2(j)),'_set,[3 1 2])']);
    end
end
save scale_train max_train_set min_train_set
save scale_test max_test_set min_train_set
