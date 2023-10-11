function result = test_processerosiondilation()   
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


i1 = im1;
im1 = histeq(im1);
figure
imshow(im1)

imgray = rgb2gray(im1);
figure
imshow(imgray)
title('grayscale')
imopen = bwareaopen(imgray,1500);
figure, imshow(imopen), title('opened')
invopen = ~imopen;
figure
imshow(invopen), title('inverted')

se = strel('disk',4);
imdil = imdilate(invopen,se);
figure, imshow(imdil), title('dilated')

iprops = regionprops(imdil,'all');
count = numel(iprops);
bbox = iprops.BoundingBox;

hold on
for i = 1 : count
   rectangle('Position',iprops(i).BoundingBox,'EdgeColor','g','LineWidth',2);
end
hold off

for i = 1 : count
   image = imcrop(imdil,iprops(i).BoundingBox);
   cc = bwconncomp(image);
   if cc.NumObjects > 5 
      bbox = iprops(i).BoundingBox;
   end
end

graynew = imcrop(imgray,bbox);
figure, imshow(graynew),title('cropped')
rgb = imcrop(im1,bbox);
figure, imshow(rgb),title('rgbplate')
inva = ~graynew;
figure, imshow(~graynew), title('inverted a')
adil = imdilate(inva,strel('square',14));
figure, imshow(adil), title('dilated a')
% imadapthist(),imtophat(),imbothat() are great for shadow removal
se = strel('disk',10);
botfilt = imbothat(graynew,se);
figure, imshow(botfilt), title('botfiltered')
b = adapthisteq(botfilt);
figure, imshow(b)
c = imbinarize(b);
figure, imshow(c)
d = bwareaopen(c,250);
figure, imshow(d)
derode = imerode(d,strel('square',2));
figure, imshow(derode), title('eroded')
ddil = imdilate(derode,strel('square',3));
figure, imshow(ddil), title('dilated')
iprops = regionprops(ddil, 'all');
count = numel(iprops);
area = iprops.Area;
bbox = iprops.BoundingBox;
fprintf('Number of connected components: %d\n',count)
for i = 1 : count
   rectangle('Position',iprops(i).BoundingBox,'EdgeColor','g','LineWidth',2) 
end
for i = 1 : count
   fprintf('crop number: %d...',i)
   image = imcrop(rgb,iprops(i).BoundingBox);
   figure
   imshow(image)
   image = imcrop(graynew, iprops(i).BoundingBox);
   figure
   imshow(image)
end
invfinal = ddil;
[h, w] = size(invfinal);
% split plate:
invfinaltop = invfinal(1:(h/2)+5, :);
figure, imshow(invfinaltop), title('top part inverted')
[hg,wg] = size(graynew);
grayfinaltop = graynew(1:(h/2)+5, :);
figure, imshow(grayfinaltop), title('gray part top')
invfinalbot = invfinal((h/2):h, :);
figure, imshow(invfinalbot), title('bot part inverted')
grayfinalbot = graynew((h/2):h, :);
figure, imshow(grayfinalbot), title('gray part inverted')

ipropstop = regionprops(invfinaltop,'all');
count = numel(ipropstop);
SelectedImageFolder = 'C:\Users\HP\Desktop\Thesis_Folder\DecLetters\Detected Image Storage\DetectedTopImages';
for i = 1 : count
   image = imcrop(invfinaltop,ipropstop(i).BoundingBox);
   figure, imshow(image)
   ttle = num2str(i);
   ttle = strcat('topimage',ttle);
   title(ttle)
   grayimage = imcrop(grayfinaltop, ipropstop(i).BoundingBox);
   figure, imshow(grayimage)
   ttle = strcat('topgrayimage',num2str(i));
   title(ttle)
    
    file_name = sprintf('Image_');
    file_number = num2str(i);
    file_type = '.png';
    FileName = strcat(file_name,file_number,file_type);
    fullFileName = fullfile(SelectedImageFolder, FileName);
    imwrite(grayimage,fullFileName);
end
ipropsbot = regionprops(invfinalbot,'all');
count = numel(ipropsbot);
maxa = ipropsbot.Area;
bbox = ipropsbot.BoundingBox;
SelectedImageFolder = 'C:\Users\HP\Desktop\Thesis_Folder\DecLetters\Detected Image Storage\DetectedBottomImages';
for i = 1 : count
   image = imcrop(invfinalbot, ipropsbot(i).BoundingBox);
   figure, imshow(image)
   ttle = strcat('botimage',num2str(i));
   title(ttle)
   grayimage = imcrop(grayfinalbot, ipropsbot(i).BoundingBox);
   figure, imshow(grayimage)
   ttle = strcat('botgrayimage',num2str(i));
   title(ttle)
   
    file_name = sprintf('Image_');
    file_number = num2str(i);
    file_type = '.png';
    FileName = strcat(file_name,file_number,file_type);
    fullFileName = fullfile(SelectedImageFolder, FileName);
    imwrite(grayimage,fullFileName);
end
result = classifyimage();
end    