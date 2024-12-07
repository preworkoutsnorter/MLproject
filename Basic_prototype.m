pkg load image; % Ensure the image package is loaded

% === Step 1: Load Images Dynamically from Folder ===
function [X, y] = load_images_from_folder(folder_path, img_size, label_file)
  % folder_path: Path to the folder containing images
  % img_size: [height, width] for resizing images
  % label_file: Path to a file containing labels for the images

  % Read all image files from the folder
  image_files = dir(fullfile(folder_path, '*.png')); % Adjust '*.png' if needed
  num_images = length(image_files);

  % Initialize feature matrix and labels
  X = zeros(num_images, prod(img_size)); % Each image will be flattened into a row vector
  y = zeros(num_images, 1); % Initialize labels vector

  % Load labels from file
  labels = load(label_file); % Assumes labels are stored in the same order as image files

  if length(labels) ~= num_images
    error('Number of labels does not match the number of images.');
  end

  % Process each image
  for i = 1:num_images
    img_path = fullfile(folder_path, image_files(i).name); % Get the full path of the image
    img = imread(img_path); % Read the image

    if size(img, 3) == 3
      img = rgb2gray(img); % Convert to grayscale if it is a color image
    end
    img_resized = imresize(img, img_size); % Resize image to the desired size
    X(i, :) = double(img_resized(:)) / 255.0; % Flatten image, convert to double and normalize to [0, 1]

    y(i) = labels(i); % Assign the label
  end
end



% === Step 4: Train Logistic Regression Model ===
function theta = trainModel(initial_theta,X, y)
  options = optimset('GradObj', 'on', 'MaxIter', 400);
  [theta, ~] = fminunc(@(t)(computeCost(t, X, y)), initial_theta, options); % Optimize theta
end


% === Step 6: Main Script ===
folder_path = 'C:\Users\vizal\Documents\GitHub\MLproject\New_Downsample'; % Path to your folder with images
label_file = 'C:\Users\vizal\Documents\GitHub\MLproject\New_Downsample\label.txt'; % Path to the label file
img_size = [64, 64]; % Resize images to 64x64

% Load images and labels
[X, y] = load_images_from_folder(folder_path, img_size, label_file);
X = [ones(size(X, 1), 1) X]; % Add intercept term (column of ones)
initial_theta = zeros(size(X, 2), 1); % Initialize theta with zeros

% Compute and display initial cost and gradient
[cost, grad] = computeCost(initial_theta, X, y);

fprintf('Cost at initial theta (zeros): %f\n', cost);
fprintf('Gradient at initial theta (zeros): \n');
fprintf(' %f \n', grad);

fprintf('\nProgram paused. Press enter to continue.\n');
pause;

% Train the logistic regression model
theta = trainModel(initial_theta,X, y);


% Print theta to screen
fprintf('Cost at theta found by fminunc: %f\n', cost);
fprintf('theta: \n');
fprintf(' %f \n', theta);

% Plot Boundary
%plotDecisionBoundary(theta, X, y);

% Put some labels
%hold on;
% Labels and Legend
%xlabel('Test 1 (Theory) Score')
%ylabel('Test 2 (Coding) Score')

% Specified in plot order
%legend('Selected for Interview', 'Not Selected for Interview')
%hold off;



fprintf('\nProgram paused. Press enter to continue.\n');
pause;

%Testing

% Test on a new image
img_test_path = 'C:\Users\vizal\Downloads\TEST\TEST\sdfbsfsf.jpg';
img_test = imread(img_test_path); % Read the test image
%[X_test,y_test] = [X, y] = load_images_from_folder(folder_path, img_size, label_file);
if size(img_test, 3) == 3
    img_test = rgb2gray(img_test);
    imshow(img_test); % Convert to grayscale if it is a color image
end
img_test_resized = imresize(img_test, img_size); % Resize the test image
X_test = double(img_test_resized(:))' / 255.0; % Flatten the test image, convert to double and normalize

% Add intercept term (column of ones) to test data
X_test = [1, X_test];
y=0;
% Make a prediction
prediction = predict(theta, X_test);

% Display the result
if prediction == 1
    fprintf('The image is classified as Clean Water.\n');
else
    fprintf('The image is classified as Dirty Water.\n');
end

% Calculate training accuracy
predictions_train = predict(theta, X); % Predictions for training set
accuracy = mean(double(predictions_train == y)) * 100;
fprintf('Train Accuracy: %.2f%%\n', accuracy);

% Cross-validation predictions
%cv_predictions = sigmoid(X_cv * theta) >= 0.5;

% Cross-validation error
%cv_error = mean(double(cv_predictions ~= y_cv)) * 100;
%fprintf('Cross-Validation Error: %.2f%%\n', cv_error);

% Test predictions
test_predictions = sigmoid(X_test * theta) ;

% Test error
test_error = mean(double(test_predictions ~= y_test)) * 100;
fprintf('Test Error: %.2f%%\n', test_error);

