function img = Bin_1_f(muimg, sigimg, numimgs)
% Samples a background noise with a 1/f noise characteristic.

if nargin < 3
    numimgs = 1;  
end

dim = size(muimg);

% Generate frequency grid
[x, y] = meshgrid(-(dim(2)/2):(dim(2)/2-1), -(dim(1)/2):(dim(1)/2-1));
radius = sqrt(x.^2 + y.^2) + 1;  
scaling_factor = 2.5;  
power_spectrum = 1 ./ (radius.^scaling_factor); 

if numimgs == 1
    n = randn(dim) + 1i * randn(dim);  
    img_freq = fftshift(fft2(n)) .* power_spectrum;
    img = ifft2(ifftshift(img_freq));
    img = real(img);  
    img = img .* sigimg + muimg;
else
    img = zeros(prod(dim), numimgs);
    for i = 1:numimgs
        n = randn(dim) + 1i * randn(dim);
        img_freq = fftshift(fft2(n)) .* power_spectrum;
        im = ifft2(ifftshift(img_freq));
        im = real(im);
        im = im .* sigimg + muimg;
        img(:, i) = im(:);
    end
end

img = double(img);  
img = (img - min(img(:))) / (max(img(:)) - min(img(:)));  % Normalize to 0 to 1


t = mean(img(:)); 
sigma_t = std(img(:))*0.1; 
img = 1 ./ (1 + exp(-(img - t) / sigma_t));  % Sigmoid function

delta_mu = 0.35; %
mu_a = 0.3;
img = img * delta_mu + mu_a;

% % Add Gaussian noise
% noise_mean = 0;  
% noise_variance = 0.002;  
% gaussian_noise = noise_mean + sqrt(noise_variance) * randn(size(img));
% img = img + gaussian_noise;  % Add the Gaussian noise to the image
end

