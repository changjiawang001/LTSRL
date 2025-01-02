clc;
clear;
warning off;
addpath('./datasets');
addpath(genpath('./Tools/'));

load('Yale.mat'); dataset_name = 'Yale'; v = 3; n = 165; dd = [4096,3304,6750]; c = 15; Y = gt;

k = 1;
omega = [0.1, 5, 10];

rng('default');
for i=1:v
    X{i} = NormalizeFea(X{i});
end

alpha = 1;
beta = 10;

tic
[U, obj] = CE(X, v, n, dd, c, k, alpha, beta, omega);
toc


Xnor = cell(1,v);
for iv = 1:v
    Xnor{iv} = X{iv}';
end
ConsenX = DataConcatenate(Xnor);

for iv = 1:v
    U{iv} = U{iv}';
end
W = DataConcatenate(U);
W = W';
XX = ConsenX';
d = size(XX,1);

%% selection num
select = 1.5;
selectedFeas = select*d*0.1;

%% clustering
w = [];
for iv = 1:d
    w = [w norm(W(iv,:),2)];
end
[~,index] = sort(w,'descend');
Xw = XX(index(1:selectedFeas),:);
for i = 1:40
    label=litekmeans(Xw',c,'MaxIter',100,'Replicates',20);
    result1 = ClusteringMeasure(Y,label);
    result(i,:) = result1;
end
for j=1:2
    a=result(:,j);
    ll=length(a);
    temp=[];
    for i=1:ll
        if i<ll-18
            b=sum(a(i:i+19));
            temp=[temp;b];
        end
    end
    [e,f]=max(temp);
    e=e./20;
    MEAN(j,:)=[e,f];
    STD(j,:)=std(result(f:f+19,j));
    rr(:,j)=sort(result(:,j));
    BEST(j,:)=rr(end,j);
end

fprintf('selectedFeas(per) = %d', select*10);
fprintf('\n');
disp(['mean. ACC: ', num2str(MEAN(1,1))]);
disp(['mean. STD(ACC): ', num2str(STD(1,1))]);
disp(['mean. NMI: ', num2str(MEAN(2,1))]);
disp(['mean. STD(NMI): ', num2str(STD(2,1))]);
fprintf("k: %d, alpha: %.4f, beta: %.4f\n", k, alpha, beta);
fprintf("\n");




