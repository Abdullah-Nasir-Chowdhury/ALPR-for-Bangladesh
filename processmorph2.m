function result = processmorph2()
close all;
clear;
clc
%//// Delete the contents of all the folders in which the images will be saved
% Specify the folder where the files live
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
% Specify the folder where the files live.
myFolder = 'C:\Users\HP\Desktop\Thesis_Folder\DecLetters\Detected Image Storage\DetectedTotalImages';
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


% TextToSpeech
NET.addAssembly('System.Speech');
mySpeaker = System.Speech.Synthesis.SpeechSynthesizer;
mySpeaker.Rate = 1;

s = 'C:\Users\HP\Desktop\Thesis_Folder\SelectedImage\SelectedImage.jpg';
im1=imread(s);
figure
imshow(im1)
title('loaded')

Speak(mySpeaker,'Commencing Processing.');

i1 = im1;
im1 = histeq(im1);
figure
imshow(im1)

imgray = rgb2gray(im1);
figure
imshow(imgray)
title('grayscale')


imbin = imbinarize(imgray);


% Make sure they are connected
skeletonImage = bwmorph(imbin, 'skel');
figure
imshow(skeletonImage)
title('connected?')
i3 = skeletonImage;

figure
imeroded = imerode(i3, strel('diamond',10));
imshow(imeroded)

% Image is now reduced to only a few cc

Iprops=regionprops(i3,'BoundingBox','Area', 'Image');
area = Iprops.Area;
count = numel(Iprops);
maxa= area;
boundingBox = Iprops.BoundingBox;
for i=1:count
   if maxa<Iprops(i).Area       
%        CC=bwconncomp(Image);
       maxa = Iprops(i).Area;
       boundingBox = Iprops(i).BoundingBox;
   end
end 
hold on
for n=1:size(Iprops) 
    rectangle('Position',Iprops(n).BoundingBox,'EdgeColor','g','LineWidth',2)
end
hold off

% Find the number plate region:
%  RGB NumberPlate
NumberPlate = imcrop(i1,boundingBox);
figure
imshow(NumberPlate);
impixelinfo


NP = NumberPlate;


npgray = rgb2gray(NP);
figure
imshow(npgray)
title('gray')

npbw = imbinarize(npgray);
figure
imshow(npbw)
title('bw')

inpbw = ~npbw;
figure
imshow(inpbw)
title('inverted') 

inpbw = bwareaopen(inpbw,100);
figure
imshow(inpbw)
title('areaopened')

SelectedImageFolder = 'C:\Users\HP\Desktop\Thesis_Folder\DecLetters\Detected Image Storage\DetectedTotalImages';
Iprops = regionprops(inpbw, 'all');
count = numel(Iprops);

for i = 1 : count
   cropd = imcrop(NP,Iprops(i).BoundingBox);
   figure
   imshow(cropd)
    file_name = sprintf('Image_');
    file_number = num2str(i);
    file_type = '.png';
    FileName = strcat(file_name,file_number,file_type);
    fullFileName = fullfile(SelectedImageFolder, FileName);
    imwrite(cropd,fullFileName);
end

[h, w] = size(inpbw);
[a, b] = size(npgray);
inpbwtop = inpbw(1:(h/2), :);
figure
imshow(inpbwtop);
title('inpbw_Top')
inpbwtop_props = regionprops(inpbwtop, 'all');
hold on
for count = 1 : size(inpbwtop_props)
    rectangle('Position',inpbwtop_props(count).BoundingBox,'EdgeColor','g','LineWidth',2)
end
hold off
npgraytop = npgray(1:(h/2),:);
figure
imshow(npgraytop);
title('npgray_Top')

SelectedImageFolder = 'C:\Users\HP\Desktop\Thesis_Folder\DecLetters\Detected Image Storage\DetectedTopImages';
Iprops = regionprops(inpbwtop, 'all');
bbox = Iprops.BoundingBox;
count = numel(Iprops);
for i = 1 : count
   cropd = imcrop(npgraytop,Iprops(i).BoundingBox);
   figure
   imshow(cropd)
    file_name = sprintf('Image_');
    file_number = num2str(i);
    file_type = '.png';
    FileName = strcat(file_name,file_number,file_type);
    fullFileName = fullfile(SelectedImageFolder, FileName);
    imwrite(cropd,fullFileName);
end
% bottom part
inpbwbot = inpbw((h/2)+1:h, :);
inpbwbot_props = regionprops(inpbwbot, 'all');
figure
imshow(inpbwbot)
hold on
for count = 1 : size(inpbwbot_props)
    rectangle('Position',inpbwbot_props(count).BoundingBox,'EdgeColor','g','LineWidth',2)
end
hold off
npgraybot = npgray((h/2)+1:h,:);
figure
imshow(npgraybot)
title('npgray_Bot')


SelectedImageFolder = 'C:\Users\HP\Desktop\Thesis_Folder\DecLetters\Detected Image Storage\DetectedTotalImages';
Iprops = regionprops(inpbw, 'all');
count = numel(Iprops);

for i = 1 : count
   cropd = imcrop(NP,Iprops(i).BoundingBox);
   figure
   imshow(cropd)
    file_name = sprintf('Image_');
    file_number = num2str(i);
    file_type = '.png';
    FileName = strcat(file_name,file_number,file_type);
    fullFileName = fullfile(SelectedImageFolder, FileName);
    imwrite(cropd,fullFileName);
end

result = classifyimage();
end