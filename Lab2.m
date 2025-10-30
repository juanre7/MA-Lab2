%% Lab 2: 2D Convolution & FIR Filters (Blur, Sharpen, Edges)
% Course: Mathematical Algorithms (DSP) - Image Processing Labs
%
% -------------------------------------------------------------------------
% Notes:
% - Kernel = impulse response h[m,n].
% - Prefer Gaussian over large box for smoother frequency response.
% - Discuss separability (two 1D passes) and boundary handling.
%
% HOW TO SUBMIT:
% Include screenshots and short explanations for each section in your GitHub repo.
% Submit only the GitHub URL.
% -------------------------------------------------------------------------
close all; clear; clc;

%% Load image (no toolboxes required)
if exist('peppers.png','file')
    I0 = imread('peppers.png');
elseif exist('cameraman.tif','file')
    I0 = imread('cameraman.tif');
else
    % Fallback synthetic image
    P = peaks(256);
    P = (P - min(P(:))) / max(eps, (max(P(:)) - min(P(:))));
    I0 = uint8(255*P);
end

% Convert to grayscale double in [0,1] without rgb2gray/im2double
if ndims(I0) == 3
    I = rgb2gray_local(I0);   % custom, at end of file
else
    I = to_double01(I0);      % custom, at end of file
end

%% 1) Delta image & impulse response
delta = zeros(101,101);
delta(51,51) = 1;
h_avg = ones(3,3)/9;
H_vis = conv2(delta, h_avg, 'same');
figure;
imagesc(H_vis); axis image off; colormap gray; colorbar;
title('Impulse response of 3x3 average');

%% 2) Low-pass: box vs Gaussian, separability
h_box3 = ones(3,3)/9;
h_box7 = ones(7,7)/49;

sigma = 1.2;
g1d = gaussian1d(7, sigma);   % 1D Gaussian row vector, normalized
h_gauss = g1d' * g1d;         % separable 2D kernel

I_box3 = imfilter_basic(I, h_box3, 'replicate', 'corr');
I_box7 = imfilter_basic(I, h_box7, 'replicate', 'corr');
I_gauss = imfilter_basic(I, h_gauss, 'replicate', 'corr');

figure;
subplot(1,4,1); imagesc(I);      axis image off; colormap gray; title('Original');
subplot(1,4,2); imagesc(I_box3); axis image off; colormap gray; title('Box 3x3');
subplot(1,4,3); imagesc(I_box7); axis image off; colormap gray; title('Box 7x7');
subplot(1,4,4); imagesc(I_gauss);axis image off; colormap gray; title('Gaussian (separable)');

%% 3) Unsharp masking (sharpen)
I_blur  = imfilter_basic(I, h_gauss, 'replicate', 'corr');
mask    = I - I_blur;            % high-frequency mask
gain    = 1.0;
I_sharp = min(max(I + gain*mask, 0), 1);

figure;
subplot(1,4,1); imagesc(I);       axis image off; colormap gray; title('Original');
subplot(1,4,2); imagesc(I_blur);  axis image off; colormap gray; title('Blur');
subplot(1,4,3); imagesc(normalize01(mask)); axis image off; colormap gray; title('High-freq mask');
subplot(1,4,4); imagesc(I_sharp); axis image off; colormap gray; title('Sharpened');

%% 4) Edges: Sobel & Laplacian (manual kernels)
h_sobel_x = [-1 0 1; -2 0 2; -1 0 1];
h_sobel_y = h_sobel_x';

Gx   = imfilter_basic(I, h_sobel_x, 'replicate', 'corr');
Gy   = imfilter_basic(I, h_sobel_y, 'replicate', 'corr');
Gmag = hypot(Gx, Gy);

% Laplacian kernel (4-neighbor)
h_lap = [0 -1 0; -1 4 -1; 0 -1 0];
I_lap = imfilter_basic(I, h_lap, 'replicate', 'corr');

figure;
subplot(1,4,1); imagesc(normalize01(Gx));   axis image off; colormap gray; title('Sobel Gx');
subplot(1,4,2); imagesc(normalize01(Gy));   axis image off; colormap gray; title('Sobel Gy');
subplot(1,4,3); imagesc(normalize01(Gmag)); axis image off; colormap gray; title('Gradient magnitude');
subplot(1,4,4); imagesc(normalize01(I_lap));axis image off; colormap gray; title('Laplacian');

%% 5) Correlation vs convolution (kernel flip)
% conv2(...,'same') uses zero padding. Match that here with 'zero' boundary.
C1 = conv2(I, h_box3, 'same');                         % convolution (flips kernel)
C2 = imfilter_basic(I, h_box3, 'zero', 'conv');        % our imfilter with 'conv' mode
diff_val = max(abs(C1(:) - C2(:)));
fprintf('Max difference (conv2 vs custom imfilter with conv): %g\n', diff_val);

%% 6) Boundary handling
I_rep = imfilter_basic(I, h_box7, 'replicate', 'corr');
I_sym = imfilter_basic(I, h_box7, 'symmetric', 'corr');
I_cir = imfilter_basic(I, h_box7, 'circular', 'corr');

figure;
subplot(1,3,1); imagesc(I_rep); axis image off; colormap gray; title('Boundary: replicate');
subplot(1,3,2); imagesc(I_sym); axis image off; colormap gray; title('Boundary: symmetric');
subplot(1,3,3); imagesc(I_cir); axis image off; colormap gray; title('Boundary: circular');

%% 7) Reflections
% 1) Why is Gaussian preferred over large box LP?
%    Gaussian has a smoother, monotonic frequency response with fewer ripples,
%    which reduces ringing and preserves structures more gracefully for the
%    same blur strength.
%
% 2) What does separability do for computational cost?
%    A full k-by-k 2D convolution costs O(k^2) per pixel. A separable kernel
%    can be applied as two 1D passes (kx1 then 1xk) at O(2k) per pixel, which
%    is a large speedup for larger k.
%
% 3) How do boundary modes change corners/edges?
%    - replicate extends the nearest edge pixel outward so edges look less
%      attenuated but can create plateaus.
%    - symmetric mirrors content at the boundary so gradients are preserved
%      more naturally.
%    - circular wraps around the image which can introduce visible seams
%      unless the image is naturally periodic.

%% -------------------------- Helper functions ---------------------------

function g = gaussian1d(len, sigma)
%GAUSSIAN1D Create a 1D Gaussian kernel of given odd length and sigma.
% Returns a row vector normalized to sum to 1.
    if mod(len,2) == 0
        error('gaussian1d: length must be odd');
    end
    half = floor(len/2);
    x = -half:half;
    g = exp(-(x.^2)/(2*sigma^2));
    g = g / sum(g);
end

function J = imfilter_basic(I, h, boundary, mode)
%IMFILTER_BASIC Minimal imfilter replacement with boundary control.
%   J = imfilter_basic(I, h, boundary, mode)
%   boundary: 'replicate' | 'symmetric' | 'circular' | 'zero'
%   mode: 'corr' (no flip) | 'conv' (flip kernel)
    if nargin < 3 || isempty(boundary), boundary = 'zero'; end
    if nargin < 4 || isempty(mode), mode = 'corr'; end
    [kr, kc] = size(h);
    pr = floor(kr/2);
    pc = floor(kc/2);
    if strcmpi(mode, 'conv')
        h_use = rot90(h,2);
    else
        h_use = h;
    end
    Ip = pad_image(I, pr, pc, boundary);
    J = conv2(Ip, h_use, 'valid');  % exact same size as I
end

function Ip = pad_image(I, pr, pc, boundary)
%PAD_IMAGE 2D padding without padarray.
    [r, c] = size(I);
    switch lower(boundary)
        case 'replicate'
            top    = repmat(I(1,:),  pr, 1);
            bottom = repmat(I(end,:), pr, 1);
            I_row  = [top; I; bottom];
            left   = repmat(I_row(:,1), 1, pc);
            right  = repmat(I_row(:,end), 1, pc);
            Ip = [left, I_row, right];
        case 'symmetric'
            top    = flipud(I(1:pr,:));
            bottom = flipud(I(end-pr+1:end,:));
            I_row  = [top; I; bottom];
            left   = fliplr(I_row(:,1:pc));
            right  = fliplr(I_row(:,end-pc+1:end));
            Ip = [left, I_row, right];
        case 'circular'
            top    = I(end-pr+1:end,:);
            bottom = I(1:pr,:);
            I_row  = [top; I; bottom];
            left   = I_row(:, end-pc+1:end);
            right  = I_row(:, 1:pc);
            Ip = [left, I_row, right];
        otherwise % 'zero'
            Ip = zeros(r+2*pr, c+2*pc);
            Ip(1+pr:pr+r, 1+pc:pc+c) = I;
    end
end

function D = to_double01(A)
%TO_DOUBLE01 Convert numeric image to double in [0,1] without im2double.
    if isa(A,'double')
        D = A;
    elseif isa(A,'uint8')
        D = double(A) / 255;
    elseif isa(A,'uint16')
        D = double(A) / 65535;
    else
        D = double(A);
        m = max(D(:));
        if m > 1, D = D / m; end
    end
end

function G = rgb2gray_local(RGB)
%RGB2GRAY_LOCAL Convert RGB to grayscale double in [0,1] using ITU-R BT.601.
    RGB = to_double01(RGB);
    if size(RGB,3) ~= 3
        G = RGB;
    else
        w = reshape([0.298936, 0.587043, 0.114021], 1,1,3);
        G = sum(RGB .* w, 3);
    end
end

function J = normalize01(A)
%NORMALIZE01 Linearly map A to [0,1] for display.
    A = double(A);
    mn = min(A(:));
    mx = max(A(:));
    if mx > mn
        J = (A - mn) / (mx - mn);
    else
        J = zeros(size(A));
    end
end
