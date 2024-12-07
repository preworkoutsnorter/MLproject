 %PREDICT Predict whether the label is 0 or 1 using learned logistic
  %regression parameters theta
  %   p = PREDICT(theta, X) computes the predictions for X using a
  %   threshold at 0.5 (i.e., if sigmoid(theta'*x) >= 0.5, predict 1)
function p = predict(theta, X)
  m = size(X, 1); % Number of training examples

  % You need to return the following variables correctly
  p = zeros(m, 1);
  prob = sigmoid(X * theta);
  p = round(prob);
 end
