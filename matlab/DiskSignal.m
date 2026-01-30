function img = DiskSignal(dim, alpha, radius, pos)
%
%  -Generate an image of dimensions dim with a white circle.
%  -The circle has a specified radius.
%  -If pos is not specified, then the circle is centered in the image.
%   pos is specified by (x,y) not (i,j).
%
if (length(dim) == 1)
  dim = [dim dim];
end

if (nargin < 4)
  % the flip is because we use (x,y) coordinates
  pos = fliplr((dim - [1 1]) / 2);
end

x = 0:(dim(2)-1);
y = 0:(dim(1)-1);
[X,Y] = meshgrid(x,y);

% Initialize the image
img = zeros(dim);

% Create a circular mask with value 1 inside the radius
% radius = round(radius); % Ensure the radius is an integer
circle = sqrt((X-pos(1)).^2 + (Y-pos(2)).^2) <= radius;
img(circle) = alpha;

end



