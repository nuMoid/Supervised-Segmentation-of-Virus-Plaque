% Supervised Segmentation of Virus Plaque Using Logistic Regression with
% Regularization.

%% Initialization

clear ; close all; clc

%% Load Data of Training Set
%  The first tree columns contains the RGB values and the third column
%  contains the label.

t = cputime;
data = load('trainingset.txt');
X = data(:, 1:3); y = data(:, 4);

%% ============ Part 1: Compute Cost and Gradient ============
%  Setup the data matrix appropriately, and add ones for the intercept term
[m, n] = size(X);

% Add intercept term to x and X_test
X = [ones(m, 1) X];

% Initialize fitting parameters
initial_theta = zeros(n + 1, 1);
lambda = 0.1;
% Compute and display initial cost and gradient
[cost, grad] = cost_reg(initial_theta, X, y,lambda);

fprintf('Cost at initial theta (zeros): %f\n', cost);
fprintf('Gradient at initial theta (zeros): \n');
fprintf(' %f \n', grad);

%% ============= Part 2: Optimizing using fminunc  =============

%  optimal parameters theta.
%  Set options for fminunc
options = optimset('GradObj', 'on', 'MaxIter', 400);

%  Run fminunc to obtain the optimal theta
%  This function will return theta and the cost 
[theta, cost] = ...
	fminunc(@(t)(cost_reg(t, X, y,lambda)), initial_theta, options);

% Print theta to screen
fprintf('Cost at theta found by fminunc: %f\n', cost);
fprintf('theta: \n');
fprintf(' %f \n', theta);

%% ============== Part 3: Predict ============================
%  Set up the test set

RGB = imread('Sample.tif');
RGB0 = RGB;
[a,b,c] = size(RGB);
RGB_val = zeros(a*b,3);
k = 1;
for i = 1:a
    for j = 1:b
        RGB_val(k,:) = RGB(i,j,:);
        k = k+1;
    end
end

% Add intercept term to x and X_test
RGB_val = [ones(a*b, 1) ,RGB_val];

p = predict(theta, RGB_val);


%% ============== Part 4: Process Image  ====================
%  Transform positive pixels into [255,255,255]
%  Transform negative pixels into [0,0,0]

%  Change RGB value 
%idx_pos = find(p == 1);
%idx_neg = find(p == 0);

%I_test(idx_pos,:) = [255 255 255];
%I_test(idx_neg,:) = [0 0 0];

k=1;
for i = 1:a
    for j = 1:b
        if p(k) == 1;
            RGB(i,j,:) = [255 255 255];
        else
            RGB(i,j,:) = [0 0 0];
        end
        k = k+1;
    end
end

subplot(1,3,1);imshow(RGB);title('Prediction Using Logistic Regression')

%Process background
BW = im2bw(RGB);
for i = 1:a
    for j = 1:b
        if ((i-a/2)^2+(j-b/2)^2)>=((a-50)/2)^2;
            BW(i,j) = 0;
        end
    end
end

%% ============== Part 5: Reject Outlier ====================
BW2 = imfill(BW,'holes');
%Obtain region properties
L = bwlabel(BW2);
STAT = regionprops(BW2, 'area','PixelIdxList');
A = [STAT.Area];
%Eliminate noise
idx_noise = find(A<50);

for i = idx_noise(1:end)
    BW2(STAT(i).PixelIdxList) = 0;
end

STAT = regionprops(BW2,'MajorAxisLength','MinorAxisLength','PixelIdxList','Area');

%reject the blocks whose major axis divided by minor axis is greater  
%than  1.5

MaL = [STAT.MajorAxisLength];MiL = [STAT.MinorAxisLength];
idx_imcomp = find(MaL./MiL>1.5);

for i = idx_imcomp(1:end)
    BW2(STAT(i).PixelIdxList) = 0;
end

%remove the combined multiple plaques whose area is 2.5 times more than 
%the mean area calculated from the left blocks

A = [STAT.Area];A_mean = sum(A)/length(A);
idx_comb = find(A>2.5*A_mean);

for i = idx_comb(1:end)
    BW2(STAT(i).PixelIdxList) = 0;
end

%Wipe out the blocks which are out of the culture dish

STAT = regionprops(BW2,'Centroid','PixelIdxList','Area');
A2 = [STAT.Area];
C2 = zeros(length(A2),1);
k = 1;
for i = 1:length(A2)
    C2(i) = (STAT(i).Centroid(1)-b/2)^2+(STAT(i).Centroid(2)-a/2)^2;
    if C2(i)>((a-100)/2)^2
        idx_out(k) = i;
        k=k+1;
    end
end
for i = idx_out(1:end)
    BW2(STAT(i).PixelIdxList) = 0;
end

subplot(1,3,2);imshow(BW2);title('Plaque Segmentation');
subplot(1,3,3);imshow(RGB0);title('Centroids of Plaques')
hold on;
%% ============== Part 6: Ouput Results ====================

STAT = regionprops(BW2,'Centroid','Area','Perimeter','MajorAxisLength','MinorAxisLength');
MaL = [STAT.MajorAxisLength];MiL = [STAT.MinorAxisLength];
diameter = (MaL+MiL)/2;
area = [STAT.Area];
number = length(area);
centroid = zeros(number,2);

%Plot centroids
for i = 1:number
    centroid(i,:) = STAT(i).Centroid;
    plot(centroid(i,1),centroid(i,2),'r+');
    hold on;
end

fprintf('\nPlaque number: %d\n',number);
fprintf('\nPlaque diameters: %f\n',diameter);
fprintf('\nPlaque area: %d\n',area);
fprintf('\nProgram executed in %f seconds\n', cputime-t);