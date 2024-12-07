function [J, grad] = computeCost(theta, X, y)
  %COMPUTECOST Compute cost and gradient for logistic regression
  %   J = COMPUTECOST(theta, X, y) computes the cost of using theta as the
  %   parameter for logistic regression and the gradient of the cost
  %   w.r.t. to the parameters.

  % Initialize some useful values
  m = length(y); % number of training examples

  % You need to return the following variables correctly
  J = 0;
  grad = zeros(size(theta));
  h = sigmoid(X * theta);
  cost_1st_term = -y .* log(h);
  cost_2nd_term = -(1-y).* log(1-h);
  cost = cost_1st_term + cost_2nd_term;
  J = (1 / m) * sum(cost);
  for j = 1:length(theta)
    feature_vec_j = X(:, j); % a m x 1 vector
    theta_j = (1 / m) * sum((h-y).* feature_vec_j);
    grad(j) = theta_j;
 endfor
end
