function result = test_processbincanny()
close all;
clear;
clc

% Specify the folder where the files live.
myFolder = 'C:\Users\HP\Desktop\Thesis_Folder\DecLetters\Detected Image Storage\DetectedBottomImages';
% Check to make sure that folder actually exists.  Warn user if it doesn't.
if ~isdir(myFolder)
  errorMessage = sprintf('Error: The following folder does not exist:\n%s', myFolder);
  uiwait(warndlg(errorMessage));         
  return;
end
% Get a list of all files in the folder with the desired file name pattern.
filePattern = fullfile(myFolder, '*.png'); % Change to whatever pattern you need.
theFiles = dir(filePattern);
for k = 1 : length(theFiles)
  baseFileName = theFiles(k).name;
  fullFileName = fullfile(myFolder, baseFileName);
  fprintf(1, 'Now deleting %s\n', fullFileName);
  delete(fullFileName);
end


% Specify the folder where the files live.
myFolder = 'C:\Users\HP\Desktop\Thesis_Folder\DecLetters\Detected Image Storage\DetectedTopImages';
% Check to make sure that folder actually exists.  Warn user if it doesn't.
if ~isdir(myFolder)
  errorMessage = sprintf('Error: The following folder does not exist:\n%s', myFolder);
  uiwait(warndlg(errorMessage));
  return;
end
% Get a list of all files in the folder with the desired file name pattern.
filePattern = fullfile(myFolder, '*.png'); % Change to whatever pattern you need.
theFiles = dir(filePattern);
for k = 1 : length(theFiles)
  baseFileName = theFiles(k).name;
  fullFileName = fullfile(myFolder, baseFileName);
  fprintf(1, 'Now deleting %s\n', fullFileName);
  delete(fullFileName);
end


s =  'C:\Users\HP\Desktop\Thesis_Folder\SelectedImage\SelectedImage.jpg';
im1=imread(s);
figure
imshow(im1)
title('loaded')
NET.addAssembly('System.Speech');
mySpeaker = System.Speech.Synthesis.SpeechSynthesizer;
mySpeaker.Rate = 1;
Speak(mySpeaker,'Commencing Processing.');


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

%steps below determine the location of the number plate
Iprops=regionprops(im,'BoundingBox','Area', 'Image');
area = Iprops.Area;
count = numel(Iprops);
maxa= area;
boundingBox = Iprops.BoundingBox;
for i=1:count
   if maxa<Iprops(i).Area
       Image = bwareaopen(imopen(~imcrop(imbin,Iprops(i).BoundingBox),strel('rectangle',[4 4])),500);
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


[h, w] = size(im);
[x, y] = size(NumberPlate); %y axis size is 3 times of the x axis size
[a, b] = size(NumberPlate_gray); % here (a,b) = (h,w); so we prepare our images in grayscale


% top part
im1 = im(1:(h/2), :);
figure
imshow(im1);

im1_props = regionprops(im1, 'all');

hold on
for count = 1 : size(im1_props)
    rectangle('Position',im1_props(count).BoundingBox,'EdgeColor','g','LineWidth',2)
end
hold off

% bottom part
im2 = im((h/2)+1:h, :);
im2_props = regionprops(im2, 'all');
figure
imshow(im2)
hold on
for count = 1 : size(im2_props)
    rectangle('Position',im2_props(count).BoundingBox,'EdgeColor','g','LineWidth',2)
end
hold off
%  For the rgb part:
NumberPlate_Divide=fix(size(NumberPlate,1)/2);
NumberPlate_Top=NumberPlate(1:NumberPlate_Divide,:,:);
NumberPlate_Bottom=NumberPlate(NumberPlate_Divide+1:end,:,:);
figure
imshow(NumberPlate_Top)
figure
imshow(NumberPlate_Bottom)



% For the gray part:
NumberPlateTop_gray =NumberPlate_gray(1:a/2,:); 
figure
imshow(NumberPlateTop_gray)
NumberPlateBottom_gray = NumberPlate_gray((a/2)+1:a,:);
figure
imshow(NumberPlateBottom_gray)



% find the connected components for top plate
[L_top,Ne_top] = bwlabel(im1);
TopCharacters_BW = [];
TopCharacters_Gray = [];
DetectedImageStorageFolder = 'C:\Users\HP\Desktop\Thesis_Folder\DecLetters\Detected Image Storage';
DetectedTopImageStorageFolder = 'C:\Users\HP\Desktop\Thesis_Folder\DecLetters\Detected Image Storage\DetectedTopImages';
DetectedBottomImageStorageFolder = 'C:\Users\HP\Desktop\Thesis_Folder\DecLetters\Detected Image Storage\DetectedBottomImages';

i = 0;

for m = 1 : Ne_top
   i = i + 1;
   [r_top,c_top] = find(L_top == m);
   
   TopCharacters_BW = im1(min(r_top):max(r_top),min(c_top):max(c_top));
   figure
   imshow(TopCharacters_BW)
   
%   To avoid complications images are not taken in RGB format here.

   
   TopCharacters_Gray = NumberPlateTop_gray(min(r_top):max(r_top),min(c_top):max(c_top));
   figure
   imshow(TopCharacters_Gray)
   
   
%   This is the section where the images are saved within the folder
    file_name = sprintf('TopCharacters');
    file_number = num2str(i);
    file_type = '.png';
    FileName = strcat(file_name,file_number,file_type);
    fullFileName = fullfile(DetectedTopImageStorageFolder, FileName);
    pause(1)
    figure
    imshow(TopCharacters_Gray);
    title(FileName);  
    imwrite(TopCharacters_Gray,file_name,'png') %save the image as a Portable Graphics Format file(png)into the MatLab
    pause(1); % pause for one second
    imshow(TopCharacters_Gray) % display the image for every second
    imgName = [DetectedTopImageStorageFolder,'\Image_',num2str(i),'.png'] ;
    imwrite(TopCharacters_Gray,imgName) ;
    
    
end
% find cc for bottom part
[L_bottom,Ne_bottom] = bwlabel(im2);
BottomCharacters_BW = [];
BottomCharacters_Gray = [];
i = 0;
for n = 1 : Ne_bottom  
    i = i + 1;
    [r_bottom,c_bottom] = find(L_bottom == n);
    BottomCharacters_BW = im2(min(r_bottom):max(r_bottom),min(c_bottom):max(c_bottom));
    figure
    imshow(BottomCharacters_BW)
    
    BottomCharacters_Gray = NumberPlateBottom_gray(min(r_bottom):max(r_bottom),min(c_bottom):max(c_bottom));
    figure
    imshow(BottomCharacters_Gray)
    
    
%     This is the section where the images are saved within the folder
    file_name = sprintf('BottomCharacters');
    file_number = num2str(i);
    file_type = '.png';
    FileName = strcat(file_name,file_number,file_type);
    fullFileName = fullfile(DetectedBottomImageStorageFolder, FileName);
    pause(1)
    figure
    imshow(BottomCharacters_Gray);
    title(FileName);  
    imwrite(BottomCharacters_Gray,file_name,'png') %save the image as a Portable Graphics Format file(png)into the MatLab
    pause(1); % pause for one second
    imshow(BottomCharacters_Gray) % display the image for every second
    imgName = [DetectedBottomImageStorageFolder,'\Image_',num2str(i),'.png'] ;
    imwrite(BottomCharacters_Gray,imgName) ;
    
    
end
Speak(mySpeaker,'Plate Detection Complete. Commencing Character Recognition');

result = classifyimage();
end