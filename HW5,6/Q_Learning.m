function q = ReinforcementLearning
clc;
format short
format compact

%----------------------------------------------
n = input('Enter number of states (size of R): ');
disp('Enter Reward matrix R (use -inf for no connection):');
R = zeros(n);

for i = 1:n
    for j = 1:n
        prompt = sprintf('R(%d,%d) = ', i, j);
        R(i,j) = input(prompt);
    end
end
%----------------------------------------------
% R = [ -inf, -inf, -inf, -inf,  0,   -inf;
%       -inf, -inf, -inf,  0,   -inf,  100;
%       -inf, -inf, -inf,  0,   -inf,  -inf;      Test 
%       0,     0,   -inf,  -inf,  0,   -inf;
%       0,    -inf,  0,    -inf,  0,    100;
%       0,    -inf, -inf,  -inf,  0,    -inf ];

%----------------------------------------------
% learning parameters
%----------------------------------------------
gamma = input('gamma: ');
alpha = input('alpha: ');
goalState = size(R, 1);

% initialize Q as zero
q  = zeros(size(R));

% initialize previous Q as big number
q1 = ones(size(R)) * inf;

% counter
count = 0;

%----------------------------------------------
% Q-learning main loop
%----------------------------------------------
for episode = 0:50000
    % random initial state
    y = randperm(size(R,1));
    state = y(1);

    % find possible actions from this state
    x = find(R(state,:) >= 0);
    if size(x,2) > 0
        x1 = x(randperm(length(x),1)); % pick random valid action
    else
        continue;
    end

    % get max of all actions from the next state
    qMax = max(q, [], 2);
    q(state, x1) = (1 - alpha) * q(state, x1) + ...
                   alpha * (R(state, x1) + gamma * qMax(x1));

    % move to next state
    state = x1;

    %----------------------------------------------
    % break if convergence: small deviation on q 
    % for 1000 consecutive times
    %----------------------------------------------
    if sum(sum(abs(q1 - q))) < 0.0001 && sum(sum(q > 0))
        if count > 1000
            fprintf('Converged at episode %d\n', episode);
            break
        else
            count = count + 1; % deviation small
        end
    else
        q1 = q;
        count = 0; % reset counter when deviation is large
    end
end

%----------------------------------------------
% normalize q
%----------------------------------------------
g = max(max(q));
if g > 0
    q = 100 * q / g;
end

end
