%%
close all
clear

%%
rng(2019)

%%
orig_dataset = [repmat([2, 2], 200, 1) + randn(200, 2) * chol([1 .5; .5 2]);
           repmat([-2, 1], 200, 1) + randn(200, 2) * chol([1 -.8; -.8 1.6]);
           repmat([2, -2], 200, 1) + randn(200, 2) * chol([2 .4; .4 1]);
           repmat([-2, -2], 200, 1) + randn(200, 2) * chol([1 .5; .5 1]);
          ];

c = 2;
n = size(orig_dataset, 1);
dataset = orig_dataset;

no = [zeros(201, 1) + 1; zeros(199, 1) + 1; zeros(200, 1) + 2; zeros(200, 1) + 2];
yes = [zeros(200, 1) + 1; zeros(200, 1) + 2; zeros(200, 1) + 2; zeros(200, 1) + 1];

dataset = dataset - mean(dataset, 1);
dataset = dataset ./ std(dataset, 1);

figure
sgtitle('Preview')
subplot(1, 2, 1)
scatter(dataset(yes == 1, 1), dataset(yes == 1, 2), 2)
hold on
scatter(dataset(yes == 2, 1), dataset(yes == 2, 2), 2)
pbaspect([1 1 1])
title('colored by yes - want better separation')

subplot(1, 2, 2)
scatter(dataset(no == 1, 1), dataset(no == 1, 2), 2)
hold on
scatter(dataset(no == 2, 1), dataset(no == 2, 2), 2)
pbaspect([1 1 1])
title('colored by no - want less separation')



%%
%symk = @(x) exp(-mpdist(x) .^ 2 / (2 * 3 ^ 2)); % RBF kernel
symk = @(x) (x * x' + 2) .^ 3;   % polynomial kernel

%%
K = symk(dataset);

%% AAT

term2 = zeros(c);
for a = 1 : c
    for b = 1:(a - 1)
        term2(a, b) = mean(mean(K(no == a, no == b)));
        term2(b, a) = term2(a, b);
    end
    term2(a, a) = mean(mean(K(no == a, no == a)));
end

top = zeros(c, 1);
bot = zeros(c, 1);
num = zeros(c, 1);
for a = 1 : c
    num(a) = sum(no == a);
    if a == 1
        top(a) = 1;
    else
        top(a) = top(a - 1) + n - num(a - 1);
    end
    bot(a) = top(a) + n - num(a) - 1;
end

AAT = zeros((c - 1) * n);
for a = 1 : c
    for b = 1 : (a - 1)
        term4 = mean(K(no ~= a, no == b), 2);
        term3 = mean(K(no == a, no ~= b), 1);
        AAT(top(a) : bot(a), top(b) : bot(b)) = K(no ~= a, no ~= b) + term2(a, b) - term3 - term4;
        AAT(top(b) : bot(b), top(a) : bot(a)) = AAT(top(a) : bot(a), top(b) : bot(b))';
    end
    b = a;
    term4 = mean(K(no ~= a, no == b), 2);
    term3 = mean(K(no == a, no ~= b), 1);
    AAT(top(a) : bot(a), top(b) : bot(b)) = K(no ~= a, no ~= b) + term2(a, b) - term3 - term4;
end

AAT = AAT / eigs(AAT, 1);

th = 1e-12;
r = size(AAT, 1);

[u, s] = eigs(AAT, r);
s = diag(s);
mask = s / max(s) > th;
s = s(mask);
u = u(:, mask);

%% distances
term11 = zeros(n, c);
for a = 1 : c
    term11(:, a) = mean(K(:, no == a), 2);
end

ind1 = [];
ind2 = [];
for a = 1 : c
    ind1 = [ind1 find(no ~= a)'];
    ind2 = [ind2 (zeros(1, n - num(a)) + a)];
end

D = zeros(n);

%for ii = 1 : n
%    for jj = 1 : n
%        vec = K(ii, ind1) - K(jj, ind1) + term11(jj, ind2) - term11(ii, ind2);
%        D(ii, jj) = sqrt(sum((vec * u ./ s') .^ 2, 2));
%    end
%end

for jj = 1 : n
    vec = K(jj : n, ind1) - K(jj, ind1) + term11(jj, ind2) - term11(jj : n, ind2);
    D(jj : n, jj) = sqrt(sum((vec * u ./ s') .^ 2, 2));
end

D = D + (D' .* (1 - eye(n)));

%%

%u = umap();
%u.metric = 'precomputed';
%R = u.fit(D);
%R = tsne(D, [], 2);
R = mdscale(D, 2);

%%

figure
sgtitle('Corrected')
subplot(1, 2, 1)
scatter(R(yes == 1, 1), R(yes == 1, 2), 2)
hold on
scatter(R(yes == 2, 1), R(yes == 2, 2), 2)
pbaspect([1 1 1])
title('colored by yes - want better separation')

subplot(1, 2, 2)
scatter(R(no == 1, 1), R(no == 1, 2), 2)
hold on
scatter(R(no == 2, 1), R(no == 2, 2), 2)
pbaspect([1 1 1])
title('colored by no - want less separation')
