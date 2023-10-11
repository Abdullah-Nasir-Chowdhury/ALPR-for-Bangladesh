clear 
close all
clc

s = 'C:\Users\HP\Desktop\Thesis_Folder\Test_FolderNPDet\Test Images\car14.jpg';
im1=imread(s);
figure
imshow(im1)
title('loaded')


% Resize here if necessary:
% im = imresize(im,[x y]);
im = im1;
imgray = rgb2gray(im);
figure
imshow(imgray)
title('gray')

imbin = imbinarize(imgray);
figure
imshow(imbin)
title('imbinarize')

%noise removal: Average and Median filters
im_avgfilter = filter2(fspecial('average',3),imgray);
im_medfilt = medfilt2(imgray);
figure
imshowpair(im_avgfilter,im_medfilt,'montage');
title('Median filter better than Average filter')

% Change Edge Detection here:
im = edge(imbin,'canny');
figure
imshow(im)
title('binedge')
im = imbinarize(im_medfilt);
figure, imshow(im), title('medfilt binarized')
% im = imdilate(im,strel('disk',4));
% figure, imshow(im), title('dilated')
im = bwareaopen(im,500);
figure, imshow(im), title('opened')
im = ~im;
figure, imshow(im), title('inverted')
im = bwareaopen(im,1000);
figure, imshow(im), title('pixels removed')

%steps below determine the location of the number plate
Iprops=regionprops(im,'BoundingBox','Area', 'Image');
area = Iprops.Area;
count = numel(Iprops);
maxa= area;
boundingBox = Iprops.BoundingBox;
for i=1:count
   if maxa<Iprops(i).Area
       Image = imcrop(imbin,Iprops(i).BoundingBox);
       C=bwconncomp(Image);
       if C.NumObjects>5
       maxa=Iprops(i).Area;
       boundingBox=Iprops(i).BoundingBox;
       end
   end
end 
hold on
for n=1:size(Iprops) %I deleted a 1 here because I didn't know what it was for, it was: for n = 1 : size(propied,1)
  rectangle('Position',Iprops(n).BoundingBox,'EdgeColor','g','LineWidth',2)
%We are adding colorful rectangle boxes to the letters detected in the top half
end
hold off

% Find the number plate region:
%  RGB NumberPlate
NumberPlate = imcrop(im1,boundingBox);
figure
imshow(NumberPlate);
impixelinfo

% Gray_Scale NumberPlate
NumberPlate_gray = rgb2gray(NumberPlate);
figure
imshow(NumberPlate_gray)

% Binary NumberPlate
im = imcrop(imbin, boundingBox);
figure
imshow(im)
title('cropped')
impixelinfo

im = imopen(im, strel('rectangle', [4 4]));
im = bwareaopen(~im, 500);
Image_props = regionprops(im,'all');

figure
imshow(im);
title('inverted and filtered')
hold on
for n=1:size(Image_props)
  rectangle('Position',Image_props(n).BoundingBox,'EdgeColor','g','LineWidth',2)
%We are adding colorful rectangle boxes to the letters detected in the top half
end
hold off

