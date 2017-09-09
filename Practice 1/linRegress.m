% Machine Learning Linear Regression

%% Load Training Set

x = load('ex2x.dat');
y = load('ex2y.dat');

%% Plot Training Set

figure
plot(x, y, 'o')
xlabel('Age (years)')
ylabel('Height (meters)')

%% x0 = 1 Intercept

m = length(y); % store the number of training examples
x = [ones(m, 1), x]; % Adds the x0 Intercept to the Training Set

%% Linear Regression (Part 1: Implement Gradient Descent => 1 Iteration)

alpha = 0.07;
theta = zeros(size(x(1,:)))';

grad = (1/m).*x'*((x*theta)-y); % Calculate Gradient
theta = theta-alpha.*grad; % Calculate the new Theta

theta

%% Linear Regression (Part 2: Imprement Gradient Descent => Multiple Iteration until reaches Local Minima

