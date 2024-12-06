function [J, grad] = costFunction(theta, X, y)
  m = length(y); % Number of examples
  h = sigmoid(X * theta); % Predicted probabilities
  J = -(1/m) * (y' * log(h) + (1 - y)' * log(1 - h)); % Log-loss function
  grad = (1/m) * (X' * (h - y)); % Gradient of the cost function
end

