clc;
rng(0);

% number = 20;
number = 200000;
% number = 10000;

% train data
for i=1:number    
    i;

    % signal absent
    n = Bin_1_f(zeros(64,64),ones(64,64),1);

    lumpy = n;
    filename1=strcat('/home/wentaochen/Iterative_TV/data_binmvn4/lumpy/', num2str(i));
    str1=[filename1,'.mat'];
    save(str1,'lumpy');

    % temporary local saving
    % filename1=strcat('./lumpy',num2str(i));
    % str1=[filename1,'.png'];
    % imwrite(lumpy, str1);

    lumpy_noise = n + 0.1 * randn(64,64);
    filename2=strcat('/home/wentaochen/Iterative_TV/data_binmvn4/lumpy-noise/', num2str(i));
    str2=[filename2,'.mat'];
    save(str2,'lumpy_noise');


    % signal present
    sig = DiskSignal(64,0.07,2);
    sys = n + sig;

    lumpy = sys;
    filename1=strcat('/home/wentaochen/Iterative_TV/data_binmvn4/lumpy/',num2str(i+number));
    % str1=[filename1,'.mat'];
    % save(str1,'lumpy');

    lumpy_noise = sys + 0.1 * randn(64,64);
    filename2=strcat('/home/wentaochen/Iterative_TV/data_binmvn4/lumpy-noise/',num2str(i+number));
    % str2=[filename2,'.mat'];
    % save(str2,'lumpy_noise');

    
    % temporary plotting code for visualization
    % figure;
    % imagesc(lumpy_noise);         % or imshow(img, []); if you have Image Processing Toolbox
    % axis image;           % same scaling on x and y
    % colormap(gray);       % grayscale colormap
    % colorbar;             % optional
    % 
    % break;

end
