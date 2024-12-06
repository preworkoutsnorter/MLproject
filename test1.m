% Logistic Regression for Image Classification (Clean vs Dirty Water)
pkg load image; % Ensure the image package is loaded

% === Step 1: Load and Preprocess Images ===
function X = preprocess_images(image_files, img_size)
  % image_files: Cell array of image file paths
  % img_size: [height, width] for resizing images
  num_images = length(image_files);
  X = zeros(num_images, prod(img_size)); % Initialize feature matrix

  for i = 1:num_images
    img = imread(image_files{i});
    if size(img, 3) == 3
      img = rgb2gray(img); % Convert to grayscale
    end
    img_resized = imresize(img, img_size); % Resize image
    X(i, :) = img_resized(:)' / 255.0; % Flatten and normalize to [0, 1]
  end
end

% === Step 2: Sigmoid Function ===
function g = sigmoid(z)
  g = 1 ./ (1 + exp(-z));
end

% === Step 3: Define Cost Function ===
function [J, grad] = costFunction(theta, X, y)
  m = length(y); % Number of training examples
  h = sigmoid(X * theta); % Predicted probabilities
  J = -(1/m) * sum(y .* log(h) + (1 - y) .* log(1 - h)); % Log-loss function
  grad = (1/m) * (X' * (h - y)); % Gradient of the cost function
end

% === Step 4: Train Logistic Regression Model ===
function theta = trainModel(X, y)
  initial_theta = zeros(size(X, 2), 1); % Initialize theta
  options = optimset('GradObj', 'on', 'MaxIter', 400);
  [theta, ~] = fminunc(@(t)(costFunction(t, X, y)), initial_theta, options);
end

% === Step 5: Predict Function ===
function p = predict(theta, X)
  p = sigmoid(X * theta) >= 0.5; % Returns 1 if >= 0.5, else 0
end

% === Step 6: Main Script ===
% File paths for clean and dirty water images
image_files = {
    'C:\Users\verma\OneDrive\Desktop\Downsample\D6.png',
    'C:\Users\verma\OneDrive\Desktop\Downsample\C6.png'
};
labels = [1; 0]; % 1 for clean, 0 for dirty
img_size = [64, 64]; % Resize images to 64x64

% Preprocess images
X = preprocess_images(image_files, img_size);
X = [ones(size(X, 1), 1) X]; % Add intercept term

% Train the logistic regression model
theta = trainModel(X, labels);

% Test on new images
test_images = {
    'C:\Users\verma\OneDrive\Desktop\Downsample\D6.png',
    'C:\Users\verma\OneDrive\Desktop\Downsample\C6.png'
};
X_test = preprocess_images(test_images, img_size);
X_test = [ones(size(X_test, 1), 1) X_test]; % Add intercept term
predictions = predict(theta, X_test);
fprintf('Train Accuracy: %f\n', mean(double(p == y)) * 100);
% Display results
disp('Predictions for test images (1=Clean, 0=Dirty):');
disp(predictions);

