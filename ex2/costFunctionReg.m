function [J, grad] = costFunctionReg(initial_theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(initial_theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

%cost = power(1+exp(-X*initial_theta),-1);
%J1 = -(y'*log(cost)+(ones(1,m)-y')*log(1-cost))/m;
%J2 = lambda*(initial_theta'*initial_theta-(initial_theta(1,1))^2)/(2*m);
%J = J1 + J2;
%grad3 = X'*(cost-y)/m+lambda*initial_theta/m;
%grad2 = ones(1,m)*(cost-y)/m;
%grad1 = grad2(1,1);
%grad3(1,1) = grad1;
%grad = grad3;


h = sigmoid(X * initial_theta);
cost = y' * log(h) + (ones(1,m)-y') * log(1-h);

J1 = - cost / m;
J2 = lambda * sum(initial_theta' * initial_theta-initial_theta(1,1)^2)/(2*m);
J = J1 + J2;

initial_grad = X' * (h - y) / m;
grad = X' * (h - y) / m + lambda * initial_theta / m;
grad(1) = initial_grad(1);








% =============================================================

end
