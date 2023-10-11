function result = processcontrast4()
clear
close all
clc
% Specify the folder where the files live.
myFolder = 'C:\Users\HP\Desktop\Thesis_Folder\DecLetters\Detected Image Storage\DetectedTotalImages';
% Check to make sure that folder actually exists.  Warn user if it doesn't.
if ~isfolder(myFolder)
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
ConstantImage = imread(s);
SelectedImageFolder ='C:\Users\HP\Desktop\Thesis_Folder\SelectedImage';
%    Save the image in a specific folder:
    file_name = sprintf('SelectedImage');
    file_type = '.jpg';
    FileName = strcat(file_name,file_type);
    fullFileName = fullfile(SelectedImageFolder, FileName);
    imwrite(ConstantImage,fullFileName);
im1=imread(s);
figure
imshow(im1)
title('loaded')
NET.addAssembly('System.Speech');
mySpeaker = System.Speech.Synthesis.SpeechSynthesizer;
mySpeaker.Rate = 2;
Speak(mySpeaker,'Commencing Processing.');

i1 = im1;
% im1 = histeq(im1);
figure
imshow(i1)

imgray = rgb2gray(i1);
figure
imshow(imgray)
title('grayscale')


% imgray2 = adapthisteq(imgray);
imbin = imbinarize(imgray);
figure, imshow(imbin)

edges = edge(imgray,'sobel');
figure, imshow(edges)

dilated = imdilate(edges,strel('diamond',2));
figure, imshow(dilated)

% fill = imfill(dilated,'holes');
% figure, imshow(fill)

% 1: first remove 500 white cc to check:
remove_extra_cc = bwareaopen(dilated, 500);
figure, imshow(remove_extra_cc)
% 2: remove 500 more:
remove_extra_cc = bwareaopen(dilated, 1000);
figure, imshow(remove_extra_cc)
% % 3: remove 500 more:
% remove_extra_cc = bwareaopen(fill, 1500);
% figure, imshow(remove_extra_cc)
% % 4: remove 500 more:
% remove_extra_cc = bwareaopen(fill, 2000);
% figure, imshow(remove_extra_cc)
% % 5: remove 500 more:
% remove_extra_cc = bwareaopen(fill, 2500);
% figure, imshow(remove_extra_cc)
% % % 6: remove 500 more:
% % remove_extra_cc = bwareaopen(fill, 3000);
% % figure, imshow(remove_extra_cc)

img = remove_extra_cc;

% img = dilated;

Iprops=regionprops(img,'BoundingBox','Area', 'Image');
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


SelectedImageFolder = 'C:\Users\HP\Desktop\Thesis_Folder\DecLetters\Detected Image Storage\DetectedTotalImages';

% NP = NumberPlate;
for i = 1 : count
   cropd = imcrop(imgray,Iprops(i).BoundingBox);
   figure
   imshow(cropd)
    file_name = sprintf('Image_');
    file_number = num2str(i);
    file_type = '.png';
    FileName = strcat(file_name,file_number,file_type);
    fullFileName = fullfile(SelectedImageFolder, FileName);
    imwrite(cropd,fullFileName);
end
result = classifyImageMP();
end
