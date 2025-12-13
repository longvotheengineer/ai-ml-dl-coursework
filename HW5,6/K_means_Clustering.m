%Clustering With K Means
X = xlsread('income.csv');   
figure;
plot(X(:,1), X(:,2), '.');
title('Data');
xlabel('Age');
ylabel('Income($)');

opts = statset('Display','final');
[idx,C] = kmeans(X,3,'Distance','sqeuclidean',...
    'Replicates',5,'Options',opts);
figure;
plot(X(idx==1,1), X(idx==1,2), 'r.', 'MarkerSize', 12)
hold on
plot(X(idx==2,1), X(idx==2,2), 'b.', 'MarkerSize', 12)
plot(X(idx==3,1), X(idx==3,2), 'g.', 'MarkerSize', 12)

plot(C(:,1), C(:,2), 'kx', 'MarkerSize', 15, 'LineWidth', 3)

legend('Cluster 1','Cluster 2','Cluster 3','Centroids', ...
       'Location','NW')
title('Cluster Assignments and Centroids')
xlabel('Age')
ylabel('Income($)')
hold off

%Preprocessing using min max scaler
Xn = normalize(X, 'range');
Age_norm = normalize(X(:,1), 'range');
Income_norm = normalize(X(:,2), 'range');
Xn = [Age_norm, Income_norm];
figure;
plot(Xn(:,1), Xn(:,2), '.');
title('Data');
xlabel('Age');
ylabel('Income($)');
opts = statset('Display','final');
[idx,C] = kmeans(Xn,3,'Distance','sqeuclidean',...
    'Replicates',5,'Options',opts);
figure;
plot(Xn(idx==1,1), Xn(idx==1,2), 'r.', 'MarkerSize', 12)
hold on
plot(Xn(idx==2,1), Xn(idx==2,2), 'b.', 'MarkerSize', 12)
plot(Xn(idx==3,1), Xn(idx==3,2), 'g.', 'MarkerSize', 12)

plot(C(:,1), C(:,2), 'kx', 'MarkerSize', 15, 'LineWidth', 3)

legend('Cluster 1','Cluster 2','Cluster 3','Centroids', ...
       'Location','NW')
title('Cluster Assignments and Centroids')
xlabel('Age')
ylabel('Income($)')
hold off

%Elbow Plot
k_rng = 1:10;
sse = zeros(size(k_rng));    
for i = 1:length(k_rng)
    k = k_rng(i);
    [idx, C, sumd] = kmeans(Xn, k, 'Distance', 'sqeuclidean', ...
                            'Replicates', 5, 'Display', 'off');
    sse(i) = sum(sumd);
end

figure;
plot(k_rng, sse, 'b-o', 'LineWidth', 1.5, 'MarkerSize', 8);
xlabel('K');
ylabel('Sum of Squared Error (SSE)');
title('Elbow Method for Optimal K');
grid on;