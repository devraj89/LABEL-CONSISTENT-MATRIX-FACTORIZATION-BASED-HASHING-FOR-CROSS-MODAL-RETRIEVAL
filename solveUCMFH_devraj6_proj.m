function [U1, U2, U3, V, V3, W] = solveUCMFH_devraj6_proj(X1, X2, X3, lambdas, gamma, alpha, bits)

%% Using commong latent factors
% All V's are same

%% random initialization
X1 = X1.'; X2 = X2.'; X3 = X3.';
[dim1, nsam] = size(X1);
[dim2, ~] = size(X2);
V = rand(bits, nsam);
V3 = rand(bits, nsam);
U1 = rand(dim1, bits);
U2 = rand(dim2, bits);
U3 = rand(dim2, bits);
W = rand(bits,bits);

niter = 100;

%% compute iteratively
tolerance = 1e-6;
for t=1:niter    
    if t==1
        f_prev = 1e100;
    else
        f_prev = f(t-1);
    end
    
	% update U1 and U2 and U3    
    U1 = X1*(V.') / (V * V.' + gamma\lambdas(1)*eye(bits));
    U2 = X2*(V.') / (V * V.' + gamma\lambdas(2)*eye(bits));
    U3 = X3*(V3.') / (V3 * V3.' + gamma\lambdas(3)*eye(bits));
    
	% update V and V3    
    A = lambdas(1)*(U1.')* U1 + lambdas(2)*(U2.')* U2 ...
        + gamma*eye(bits) + alpha*(W.')*W;
    B = lambdas(1)*(U1.')*X1 + lambdas(2)*(U2.')*X2 + alpha*(W.')*V3;
    V = A \ B;    
    
    A = lambdas(3)*(U3.')* U3 + (gamma + alpha)*eye(bits);
    B = lambdas(3)*(U3.')*X3 + alpha*W*V;
    V3 = A \ B;
    
    %update W1 and W2
    W = V3 * V.' / (V * V.' + gamma\alpha * eye(bits));
    
    % compute objective function
    norm1 = norm(X1 - U1 * V, 'fro') ^ 2;
    norm2 = norm(X2 - U2 * V, 'fro') ^ 2;
    norm3 = norm(X3 - U3 * V3, 'fro') ^ 2;    
    norm4 = norm(V3 - W * V, 'fro') ^ 2;    
    norm6 = norm(U1, 'fro')^2 + norm(U2, 'fro')^2 + norm(U3 , 'fro')^2 ...
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