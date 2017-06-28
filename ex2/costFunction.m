function [J, grad] = costFunction(initial_theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

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
%
% Note: grad should have the same dimensions as theta
%
%cost = power(1+exp(-X*initial_theta),-1);
%J = -(y'*log(cost)+(ones(1,m)-y')*log(1-cost))/m;
%grad = X'*(cost-y)/m;


h = sigmoid(X * initial_theta);
cost = y' * log(h) + (ones(1,m)-y') * log(1-h);
J = - cost / m;
grad = X' * (h - y)/m;









% =============================================================

end
