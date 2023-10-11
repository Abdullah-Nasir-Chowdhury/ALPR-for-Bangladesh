function result = processbysegmentation()

%//// Delete the contents of all the folders in which the images will be saved
% Specify the folder where the files live.
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

SelectedImageFolder = 'C:\Users\HP\Desktop\Thesis_Folder\DecLetters\Detected Image Storage\DetectedBottomImages';
Iprops = regionprops(inpbwbot, 'all');
bbox = Iprops.BoundingBox;
count = numel(Iprops);
for i = 1 : count
   cropd = imcrop(npgraybot,Iprops(i).BoundingBox);
   figure
   imshow(cropd)
    file_name = sprintf('Image_');
    file_number = num2str(i);
    file_type = '.png';
    FileName = strcat(file_name,file_number,file_type);
    fullFileName = fullfile(SelectedImageFolder, FileName);
    imwrite(cropd,fullFileName);
end

clear 
close all
clc
NET.addAssembly('System.Speech');
mySpeaker = System.Speech.Synthesis.SpeechSynthesizer;
mySpeaker.Rate = 1;
Speak(mySpeaker,'Running Plate Text Classification');
cmd = sprintf('Running Plate Text Classification');
disp(cmd)
outputFolder = fullfile('DecLetters');
rootFolder = fullfile(outputFolder,'DecCharacters');


BottomImageDetectionFolder = 'C:\Users\HP\Desktop\Thesis_Folder\DecLetters\Detected Image Storage\DetectedBottomImages';
TopImageDetectionFolder = 'C:\Users\HP\Desktop\Thesis_Folder\DecLetters\Detected Image Storage\DetectedTopImages';


DirectoryforBottomImages = dir([BottomImageDetectionFolder,'/*.png']);
NumberofBottomImages = size(DirectoryforBottomImages,1);
sprintf('Detected characters for bottom part of the plate:')
disp(NumberofBottomImages)

DirectoryforTopImages = dir([TopImageDetectionFolder,'/*.png']);
NumberofTopImages = size(DirectoryforTopImages,1);
sprintf('Detected characters for top part of the plate:')
disp(NumberofTopImages)

categories = {'0','1','2','3','4','5','6','7','8','9','Dhaka','Chotto','Sylhet','Go','Mo','No','Bho','Cho','Metro','-',...
    'Noise','Noise1','Noise2','Noise3','Noise4','Noise5', 'Noise6', 'Noise7','Noise8', 'Noise9',...
    'Noise10', 'Noise11', 'Noise12', 'Noise13', 'Noise14','Noise15', 'Noise16', 'Noise17', 'Noise18',...
    'Noise19', 'Noise20', 'Noise21','Noise22','Noise23','Noise24','Noise25','Noise26','Noise27',...
    'Noise28','Noise29','Noise30','Noise31','Noise32','Noise33','Noise34','Noise35','Noise36',...
    'Noise37','Noise38','Noise39','Noise40','Noise41','Noise42','Noise43','Noise44','Noise45',...
    'Noise46','Noise47','Noise48','Noise49','Noise50','Noise51','Noise52','Noise53','Noise54',...
    'Noise55','Noise56','Noise57','Noise58','Noise59','Noise60','Noise61','Noise62','Noise63',...
    'Noise64','Noise65','Noise66','Noise67','Noise68','Noise69','Noise70','Noise71','Noise72',...
    'Noise73','Noise74','Noise75','Noise76','Noise77','Noise78','Noise79','Noise80'};

%  Create Image Data Storage
imds = imageDatastore(fullfile(rootFolder,categories),'LabelSource','foldernames');


% Count the number of images in each folder
tbl = countEachLabel(imds);

% If number of images per category is not the same, make them the same:
minSetCount = min(tbl{:,2});
imds = splitEachLabel(imds, minSetCount, 'randomize');
countEachLabel(imds);


%  Name the labels
Hyphen = find(imds.Labels == '-', 1);
Two = find(imds.Labels == '2', 1);
Three = find(imds.Labels == '3', 1);
Seven = find(imds.Labels == '7', 1);
Dhaka = find(imds.Labels == 'Dhaka', 1);
Metro = find(imds.Labels == 'Metro', 1);
Go = find(imds.Labels == 'Go', 1);
Noise = find(imds.Labels == 'Noise',1);
Noise1 = find(imds.Labels == 'Noise1', 1);
Noise2 = find(imds.Labels == 'Noise2', 1);
Noise3 = find(imds.Labels == 'Noise3', 1);
Noise4 = find(imds.Labels == 'Noise4', 1);
Noise5 = find(imds.Labels == 'Noise5', 1);
Noise6 = find(imds.Labels == 'Noise6', 1);

% Resnet(50) a neural network 
net = resnet50();
figure
plot(net)
title('Architecture of ResNet-50')
set(gca, 'YLim', [150,170]); %To resize the architecture figure
% gca loads the current axis
% YLim stands for YLimit
% Third argument is for the limits



% Inpection of the first layer allows to find out what kind of images this
% particular pre-trained convolutional neural network requires
net.Layers(1);

% Uncomment above line and you will find that this network will take a 
% [224 224 3] image size
net.Layers(end);
% This CNN model has been trained to solve 1000 classes problem
% It can also be found by this code:
numel(net.Layers(end).ClassNames);
% Uncomment above lines to find total number of classes

[trainingSet, testSet] = splitEachLabel(imds, 0.3, 'randomize');  %30 percent will be used for training and the rest for validation
% Since resnet(50) takes [224 224 3] input
imageSize = net.Layers(1).InputSize;

% Convert from Grayscale to RGB image
augmentedTrainingSet = augmentedImageDatastore(imageSize, trainingSet, 'ColorPreprocessing', 'gray2rgb');

augmentedTestSet = augmentedImageDatastore(imageSize, testSet,'ColorPreprocessing','gray2rgb');


w1 = net.Layers(2).Weights; %this line is getting the weights of the 2nd convolutional layer 
w1 = mat2gray(w1);

% figure
% montage(w1)
% title('First Convolutional Layer Weight')


% Feature Extraction
featureLayer = 'fc1000';
trainingFeatures = activations(net, augmentedTrainingSet, featureLayer, 'MiniBatchSize', 32, 'OutputAs', 'columns');

trainingLables = trainingSet.Labels;

% FitClassErrorCorrectingOutputCodec
classifier = fitcecoc(trainingFeatures, trainingLables, 'Learner','Linear','Coding','onevsall','ObservationsIn','columns');

testFeatures = activations(net, augmentedTestSet, featureLayer, 'MiniBatchSize', 32, 'OutputAs', 'columns');


predictLabels = predict(classifier, testFeatures, 'ObservationsIn', 'columns');

testLabels = testSet.Labels;
confMat = confusionmat(testLabels, predictLabels);
confMat = bsxfun(@rdivide, confMat, sum(confMat,2));

mean(diag(confMat))

imgTopCount = 0;
DeterminedTopPlateCharacterCount = 1;
% Create an empty string array using the strings function
DeterminedTopPlateCharacters = strings;

for count = 1 : NumberofTopImages

imgTopCount = imgTopCount+1;
imgCountString = num2str(imgTopCount);
imgName = 'Image_';
imgFullName = strcat(imgName,imgCountString,'.png');

newImage = imread(fullfile('C:\Users\HP\Desktop\Thesis_Folder\DecLetters\Detected Image Storage\DetectedTopImages', imgFullName));


ds = augmentedImageDatastore(imageSize, newImage, 'ColorPreprocessing', 'gray2rgb');

imageFeatures = activations(net, ds, featureLayer, 'MiniBatchSize', 32, 'OutputAs', 'columns');

label = predict(classifier, imageFeatures, 'ObservationsIn', 'columns');

sprintf('%s', label)
if(label ~= '0' && label ~= '1' && label ~= '2' && label ~= '3' && label ~= '4' && label ~= '5' && label ~= '6' && label~= '7' && label ~= '8' && label ~= '9' && label~= '-' && label ~= 'Noise' && label ~= 'Noise1' && label ~= 'Noise2' &&label ~= 'Noise3' && label ~= 'Noise4' && label ~= 'Noise5' && label ~= 'Noise6' && label ~= 'Noise7' && label ~= 'Noise8' && label ~= 'Noise9' && label ~= 'Noise10' && label ~= 'Noise11' && label~= 'Noise12' && label~= 'Noise13' && label ~= 'Noise14' && label ~= 'Noise16' && label ~= 'Noise17' && label ~= 'Noise18' && label ~= 'Noise19' && label ~= 'Noise20' && label ~= 'Noise21' && label ~= 'Noise22' && label ~= 'Noise23' && label ~= 'Noise24' && label ~= 'Noise25' && label ~= 'Noise26' && label ~= 'Noise27' && label ~= 'Noise28' && label ~= 'Noise29' && label ~= 'Noise30' && label ~= 'Noise31' && label ~= 'Noise32' && label ~= 'Noise33' && label ~= 'Noise34' && label ~= 'Noise35' && label ~= 'Noise36' && label ~= 'Noise37' && label ~= 'Noise38' && label ~= 'Noise39' && label ~= 'Noise40'  && label ~= 'Noise41' && label ~= 'Noise42' && label ~= 'Noise43' && label ~= 'Noise44' && label ~= 'Noise45' && label ~= 'Noise46' && label ~= 'Noise47' && label ~= 'Noise48' && label ~= 'Noise49' && label ~= 'Noise50' && label ~= 'Noise51' && label ~= 'Noise52' && label ~= 'Noise53' && label ~= 'Noise54' && label ~= 'Noise55'  && label ~= 'Noise56'  && label ~= 'Noise57' && label ~= 'Noise58' && label ~= 'Noise59' && label ~= 'Noise60' && label~= 'Noise61' && label~= 'Noise62' && label~= 'Noise63' && label~= 'Noise64' && label~= 'Noise65' && label~= 'Noise66' && label~= 'Noise67' && label~= 'Noise68' && label~= 'Noise69' && label~= 'Noise70' && label ~= 'Noise71' && label ~= 'Noise72' && label ~= 'Noise73' && label ~= 'Noise74' && label ~= 'Noise75' && label ~= 'Noise76' && label ~= 'Noise77' && label ~= 'Noise78' && label ~= 'Noise79' && label ~= 'Noise80')
    DeterminedTopPlateCharacters(DeterminedTopPlateCharacterCount) = label;
    DeterminedTopPlateCharacterCount = DeterminedTopPlateCharacterCount + 1;
end

end
% Join the cell array of inputs into a single string
DetTopChars = strjoin(DeterminedTopPlateCharacters);


imgBottomCount = 0;
DeterminedBottomPlateCharacterCount = 1;
% Create an empty string array using the strings function
DeterminedBottomPlateCharacters = strings;
for count = 1 : NumberofBottomImages

imgBottomCount = imgBottomCount+1;
imgCountString = num2str(imgBottomCount);
imgName = 'Image_';
imgFullName = strcat(imgName,imgCountString,'.png');

newImage = imread(fullfile('C:\Users\HP\Desktop\Thesis_Folder\DecLetters\Detected Image Storage\DetectedBottomImages', imgFullName));


ds = augmentedImageDatastore(imageSize, newImage, 'ColorPreprocessing', 'gray2rgb');

imageFeatures = activations(net, ds, featureLayer, 'MiniBatchSize', 32, 'OutputAs', 'columns');

label = predict(classifier, imageFeatures, 'ObservationsIn', 'columns');

sprintf('%s', label)

if (label ~= '-' && label ~= 'Dhaka' && label ~= 'Chotto' && label ~= 'Sylhet' && label ~= 'Metro' && label ~= 'Mo' && label ~= 'No' && label ~= 'Go' && label ~= 'Cho' && label ~= 'Bho' && label ~= 'Noise' && label ~= 'Noise1' && label ~= 'Noise2' &&label ~= 'Noise3' && label ~= 'Noise4' && label ~= 'Noise5' && label ~= 'Noise6' && label ~= 'Noise7' && label~= 'Noise8' && label ~= 'Noise9' && label ~= 'Noise10' && label ~= 'Noise11'  && label~= 'Noise12' && label~= 'Noise13' && label ~= 'Noise14' && label ~= 'Noise15' && label ~= 'Noise16' && label ~= 'Noise17' && label ~= 'Noise18' && label ~= 'Noise19' && label ~= 'Noise20' && label ~= 'Noise21' && label ~= 'Noise22' && label ~= 'Noise23' && label ~= 'Noise24' && label ~= 'Noise25' && label ~= 'Noise26' && label ~= 'Noise27' && label ~= 'Noise28' && label ~= 'Noise29' && label ~= 'Noise30' && label ~= 'Noise31' && label ~= 'Noise32' && label ~= 'Noise33' && label ~= 'Noise34' && label ~= 'Noise35' && label ~= 'Noise36' && label ~= 'Noise37' && label ~= 'Noise38' && label ~= 'Noise39' && label ~= 'Noise40'  && label ~= 'Noise41' && label ~= 'Noise42' && label ~= 'Noise43' && label ~= 'Noise44' && label ~= 'Noise45' && label ~= 'Noise46' && label ~= 'Noise47' && label ~= 'Noise48' && label ~= 'Noise49' && label ~= 'Noise50' && label ~= 'Noise51' && label ~= 'Noise52' && label ~= 'Noise53' && label ~= 'Noise54'  && label ~= 'Noise55'  && label ~= 'Noise56'  && label ~= 'Noise57' && label ~= 'Noise58' && label ~= 'Noise59' && label ~= 'Noise60' && label~= 'Noise61' && label~= 'Noise62' && label~= 'Noise63' && label~= 'Noise64' && label~= 'Noise65' && label~= 'Noise66' && label~= 'Noise67' && label~= 'Noise68' && label~= 'Noise69' && label~= 'Noise70' && label ~= 'Noise71' && label ~= 'Noise72' && label ~= 'Noise73' && label ~= 'Noise74' && label ~= 'Noise75' && label ~= 'Noise76' && label ~= 'Noise77' && label ~= 'Noise78' && label ~= 'Noise79' && label ~= 'Noise80')
    DeterminedBottomPlateCharacters(DeterminedBottomPlateCharacterCount) = label;
    DeterminedBottomPlateCharacterCount = DeterminedBottomPlateCharacterCount + 1;
end
end
% Join the cell array of inputs into a single string
DetBotChars = strjoin(DeterminedBottomPlateCharacters);

NET.addAssembly('System.Speech');
mySpeaker = System.Speech.Synthesis.SpeechSynthesizer;
mySpeaker.Rate = 1;
Speak(mySpeaker,'Character Recognition Complete');
disp(DetTopChars)
disp(DetBotChars)
FullPlate = strcat(DetTopChars,DetBotChars);
disp(FullPlate)
Speak(mySpeaker, 'The characters are ');
pause(1);
Speak(mySpeaker, FullPlate);
pause(1);
if (DetTopChars == 'Dhaka Metro Go' || DetTopChars == 'Dhaka Metro Mo' || DetTopChars == 'Dhaka Metro No' || DetTopChars == 'Chotto Metro Go' || DetTopChars == 'Dhaka Metro Cho' || DetTopChars == 'Sylhet Go' || DetTopChars == 'Chotto Metro Bho')
    Speak(mySpeaker, 'Plate Detection Complete.');
    top = DetTopChars;
    bot = string(DetBotChars);
    checkspreadsheet(top,bot);
    Speak(mySpeaker, 'Terminating Program.');
    clear all
    close all 
    clc
    result = 1;
else
result = 0;
end
end