function [U, U3, V, V3, W, P1, P2] = solveUCMFH_devraj7_proj_propagate(X1, X2, X3, lambdas, gamma, alpha, bits, U,U3,W,P1,P2, rho)
%% Using commong latent factors
%% and commong dictionaries also
% let us see how it performs 

%% random initialization
X1 = X1.'; X2 = X2.'; X3 = X3.';
[~, nsam] = size(X1);
V = rand(bits, nsam);
V3 = rand(bits, nsam);

niter = 100;

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
    U = (1-rho)*U + rho* A / B;
    U3 = (1-rho)*U3 + rho* X3*(V3.') / (V3 * V3.' + gamma\lambdas(3)*eye(bits));
    
	% update V and V3    
    A = (lambdas(1) + lambdas(2))*(U.')* U ...
        + gamma*eye(bits) + alpha*(W.')*W;
    B = lambdas(1)*(U.')*P1*X1 + lambdas(2)*(U.')*P2*X2 + alpha*(W.')*V3;
    V = A \ B;    
    
    A = lambdas(3)*(U3.')* U3 + (gamma + alpha)*eye(bits);
    B = lambdas(3)*(U3.')*X3 + alpha*W*V;
    V3 = A \ B;
    
    %update W1 and W2
    W = (1-rho)*W + rho* V3 * V.' / (V * V.' + gamma\alpha * eye(bits));
    
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