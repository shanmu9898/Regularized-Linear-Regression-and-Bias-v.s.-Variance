function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%

m = length(y);
k = 1/(2*m);
theasquare = theta .^2;
J = sum(k* (((theta' * X')' - y) .^2)) + (lambda/(2 * m) * sum(theasquare(2:end,:)));


kk = 1/m;

h = X * theta;
p = kk * (X' * (h - y));

thetasum = (lambda/m) * theta;
thetasum(1) = 0;

grad = p + thetasum;	


% =========================================================================

grad = grad(:);

end
