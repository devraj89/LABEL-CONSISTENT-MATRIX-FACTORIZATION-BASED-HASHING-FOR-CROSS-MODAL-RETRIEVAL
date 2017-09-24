function [U, U3, V, V3, W, P1, P2] = solveUCMFH_devraj7_proj(X1, X2, X3, lambdas, gamma, alpha, bits, option)

% Use common dictionary and common hash codes
% All U's and V's are same

%% Using commong latent factors
%% and commong dictionaries also
% let us see how it performs 

%% random initialization
X1 = X1.'; X2 = X2.'; X3 = X3.';
[dim1, nsam] = size(X1);
[dim2, ~] = size(X2);
V = rand(bits, nsam);
V3 = rand(bits, nsam);
U = rand(dim1, bits);
U3 = rand(dim2, bits);
W = rand(bits,bits);

niter = 100;

%% initialization of the transformation matrix P
% This can be randomly initialized but ofcourse we can use someother
% advanced techniques like PCA, CCA, MCCA, CCCA, GMA, etc

% dimension of common domain c
c = 10;
P1 = rand(c,dim1);
P2 = rand(c,dim2);

% option 1 : pca separately in both the dimensions (fast version)
% option 2 : cca use the svd version
% some other option you can use the random initializations too


if option==1
    % use pca separately in both the dimensions
%     [coeff1, ~, latent1, ~, ~, ~] = pca_modified(X1.');
    [coeff1, ~, latent1, ~, ~, ~] = pca(X1.');
    t = latent1./sum(latent1); t = cumsum(t); idx1 = find(t>0.99); idx1 = idx1(1);    
%     [coeff2, ~, latent2, ~, ~, ~] = pca_modified(X2.');
    [coeff2, ~, latent2, ~, ~, ~] = pca(X2.');
    t = latent2./sum(latent2); t = cumsum(t); idx2 = find(t>0.99); idx2 = idx2(1);    
    c = min(idx1,idx2);
    P1 = coeff1(:,1:c); P1 = P1.';
    P2 = coeff2(:,1:c); P2 = P2.';    
elseif option==2
    % use the svd version of the cca code 
    % now this should ideally make the results much better - for cca is a
    % subspace learning algorithm
    [P1,P2,~,~,~] = cca_by_svd(X1.',X2.');    
    P1 = P1.'; P2 = P2.';
else
    % continue with the random initialization
    % actually since the data is not single label I cannot use the cluster
    % cca, gma codes, etc.
end

%% compute iteratively
tolerance = 1e-6;
for t=1:niter    
    if t==1
        f_prev = 1e100;
    else
        f_prev = f(t-1);
    end
    
	% update U and U3    
    A = (lambdas(1)*P1*X1*(V.') + lambdas(2)*P2*X2*(V.'));
    B = ((lambdas(1)+lambdas(2))*V*(V.') + gamma*eye(bits));
    U = A / B;
    U3 = X3*(V3.') / (V3 * V3.' + gamma\lambdas(3)*eye(bits));
    
	% update V and V3    
    A = (lambdas(1) + lambdas(2))*(U.')* U ...
        + gamma*eye(bits) + alpha*(W.')*W;
    B = lambdas(1)*(U.')*P1*X1 + lambdas(2)*(U.')*P2*X2 + alpha*(W.')*V3;
    V = A \ B;    
    
    A = lambdas(3)*(U3.')* U3 + (gamma + alpha)*eye(bits);
    B = lambdas(3)*(U3.')*X3 + alpha*W*V;
    V3 = A \ B;
    
    %update W1 and W2
    W = V3 * V.' / (V * V.' + gamma\alpha * eye(bits));
    
    % compute objective function
    norm1 = norm(P1 * X1 - U * V, 'fro') ^ 2;
    norm2 = norm(P2 * X2 - U * V, 'fro') ^ 2;
    norm3 = norm(X3 - U3 * V3, 'fro') ^ 2;    
    norm4 = norm(V3 - W * V, 'fro') ^ 2;    
    norm6 = norm(U, 'fro')^2 + norm(U3 , 'fro')^2 ...
        + norm(V, 'fro')^2 + norm(V3, 'fro')^2 + norm(W, 'fro')^2;
    f(t)= lambdas(1) * norm1 + lambdas(2) * norm2 + lambdas(3) * norm3 ...
        + alpha * norm4 + gamma * norm6;    
    if (f_prev-f(t))/f_prev <= tolerance
        break;
    end    
%     if mod(t,10)==0
%         fprintf('The iteration is : %i and f val is : %f \n', t, f(t));
%     end
end
end