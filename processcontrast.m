
function result = processcontrast()
myFolder = 'C:\Users\HP\Desktop\Thesis_Folder\DecLetters\Detected Image Storage\DetectedBottomImages';
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


% Specify the folder where the files live.
myFolder = 'C:\Users\HP\Desktop\Thesis_Folder\DecLetters\Detected Image Storage\DetectedTopImages';
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


% TextToSpeech
NET.addAssembly('System.Speech');
mySpeaker = System.Speech.Synthesis.SpeechSynthesizer;
mySpeaker.Rate = 1;
Speak(mySpeaker,'select a license plate image');

[file,path]=uigetfile({'*.jpeg;*.jpg;*.bmp;*.png;*.tif'},'Choose an image');
s=[path,file];

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

Speak(mySpeaker,'Commencing Processing.');

i1 = im1;
im1 = histeq(im1);
figure
imshow(im1)

imgray = rgb2gray(im1);
figure
imshow(imgray)
title('grayscale')


% imgray2 = adapthisteq(imgray);
imbin = imbinarize(imgray);
i = bwareaopen(imgray,3000);
figure
imshow(i)
title('area opening')

figure
imshow(~i)
title('inverted')

% Do dilation to connect broken connections
se = strel('disk', 10);
i2 = imdilate(~i,se);
figure
imshow(i2)
title('dilated')

% Make sure they are connected
skeletonImage = bwmorph(i2, 'skel');
figure
imshow(skeletonImage)
title('connected?')
i3 = skeletonImage;
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

inpbw = bwareaopen(inpbw,1500);
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

result = classifyimage();
end
