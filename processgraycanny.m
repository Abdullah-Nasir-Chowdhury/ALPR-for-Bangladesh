function result = processgraycanny()
%     Re-try with a different code
close all;
clear;
clc
%//// Delete the contents of the folder in which the images will be saved
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

% If some area is missing try edge detection for gray
im = edge(imgray, 'canny');
figure
imshow(im)
title('grayedge')



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
im1 = im(1:(h/2) + 10, :);
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
NumberPlateTop_gray =NumberPlate_gray(1:a/2 + 10,:); 
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




%  The plate has been detected. Now the determination of the 
%  Characters is necessary

% /////// This is the section that requires deep learning //////







clear all
close all
clc
sprintf('Running Plate Text Classification');
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

categories = {'0','1','2','3','4','5','6','7','8','9','Dhaka','Chotto','Sylhet','Mo','Go','Bho','No','Cho','Metro','-',...
    'Dhaka Metro Go',...
    'Noise','Noise1','Noise2','Noise3','Noise4','Noise5','Noise6','Noise7',...
    'Noise8','Noise9','Noise10', 'Noise11','Noise12', 'Noise13', 'Noise14',...
    'Noise15', 'Noise16', 'Noise17', 'Noise18', 'Noise19', 'Noise20', 'Noise21',...
    'Noise22','Noise23','Noise24','Noise25','Noise26','Noise27','Noise28','Noise29','Noise30',...
    'Noise31','Noise32','Noise33','Noise34','Noise35','Noise36','Noise37', 'Noise38','Noise39',...
    'Noise40','Noise41','Noise42','Noise43','Noise44','Noise45','Noise46','Noise47','Noise48',...
    'Noise49','Noise50','Noise51','Noise52','Noise53','Noise54','Noise55','Noise56','Noise57',...
    'Noise58','Noise59','Noise60','Noise61','Noise62','Noise63',...
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

%  Test the figures
% figure
% subplot(2,2,1);
% imshow(readimage(imds,Two));
% subplot(2,2,2);
% imshow(readimage(imds,Three));
% subplot(2,2,3);
% imshow(readimage(imds,Seven));


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
if(label ~= '0' && label ~= '1' && label ~= '2' && label ~= '3' && label ~= '4' && label ~= '5' && label ~= '6' && label~= '7' && label ~= '8' && label ~= '9' && label~= '-' && label ~= 'Noise' && label ~= 'Noise1' && label ~= 'Noise2' &&label ~= 'Noise3' && label ~= 'Noise4' && label ~= 'Noise5' && label ~= 'Noise6' && label ~= 'Noise7' && label ~= 'Noise8' && label ~= 'Noise9' && label ~= 'Noise10' && label ~= 'Noise11' && label~= 'Noise12' && label~= 'Noise13' && label ~= 'Noise14' && label ~= 'Noise15' && label ~= 'Noise16' && label ~= 'Noise17' && label ~= 'Noise18' && label ~= 'Noise19' && label ~= 'Noise20' && label ~= 'Noise21' && label ~= 'Noise22' && label ~= 'Noise23' && label ~= 'Noise24' && label ~= 'Noise25' && label ~= 'Noise26' && label ~= 'Noise27' && label ~= 'Noise28' && label ~= 'Noise29' && label ~= 'Noise30' && label ~= 'Noise31' && label ~= 'Noise32' && label ~= 'Noise33' && label ~= 'Noise34' && label ~= 'Noise35' && label ~= 'Noise36' && label ~= 'Noise37' && label ~= 'Noise38' && label ~= 'Noise39' && label ~= 'Noise40'  && label ~= 'Noise41' && label ~= 'Noise42' && label ~= 'Noise43' && label ~= 'Noise44' && label ~= 'Noise45' && label ~= 'Noise46' && label ~= 'Noise47' && label ~= 'Noise48' && label ~= 'Noise49' && label ~= 'Noise50' && label ~= 'Noise51' && label ~= 'Noise52' && label ~= 'Noise53' && label ~= 'Noise54'  && label ~= 'Noise55'  && label ~= 'Noise56'  && label ~= 'Noise57' && label ~= 'Noise58' && label ~= 'Noise59' && label ~= 'Noise60' && label~= 'Noise61' && label~= 'Noise62' && label~= 'Noise63' && label~= 'Noise64' && label~= 'Noise65' && label~= 'Noise66' && label~= 'Noise67' && label~= 'Noise68' && label~= 'Noise69' && label~= 'Noise70' && label ~= 'Noise71' && label ~= 'Noise72' && label ~= 'Noise73' && label ~= 'Noise74' && label ~= 'Noise75' && label ~= 'Noise76' && label ~= 'Noise77' && label ~= 'Noise78' && label ~= 'Noise79' && label ~= 'Noise80')
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

if (label ~= '-' && label ~= 'Dhaka' && label ~= 'Chotto' && label ~= 'Sylhet' && label ~= 'Metro' && label ~= 'Mo' && label ~= 'Go' && label ~= 'Bho' && label ~= 'Cho' && label ~= 'No' && label ~= 'Noise' && label ~= 'Noise1' && label ~= 'Noise2' &&label ~= 'Noise3' && label ~= 'Noise4' && label ~= 'Noise5' && label ~= 'Noise6' && label ~= 'Noise7' && label ~= 'Noise8' && label ~= 'Noise9' && label ~= 'Noise10' && label ~= 'Noise11' && label~= 'Noise12' && label~= 'Noise13' && label ~= 'Noise14' && label ~= 'Noise15' && label ~= 'Noise16' && label ~= 'Noise17' && label ~= 'Noise18' && label ~= 'Noise19' && label ~= 'Noise20' && label ~= 'Noise21' && label ~= 'Noise22' && label ~= 'Noise23' && label ~= 'Noise24' && label ~= 'Noise25' && label ~= 'Noise26' && label ~= 'Noise27' && label ~= 'Noise28' && label ~= 'Noise29' && label ~= 'Noise30' && label ~= 'Noise31'&& label ~= 'Noise32' && label ~= 'Noise33' && label ~= 'Noise34' && label ~= 'Noise35' && label ~= 'Noise36' && label ~= 'Noise37' && label ~= 'Noise38' && label ~= 'Noise39' && label ~= 'Noise40' && label ~= 'Noise41' && label ~= 'Noise42' && label ~= 'Noise43' && label ~= 'Noise44' && label ~= 'Noise45' && label ~= 'Noise46' && label ~= 'Noise47' && label ~= 'Noise48' && label ~= 'Noise49' && label ~= 'Noise50' && label ~= 'Noise51' && label ~= 'Noise52' && label ~= 'Noise53' && label ~= 'Noise54'  && label ~= 'Noise55'  && label ~= 'Noise56'  && label ~= 'Noise57' && label ~= 'Noise58' && label ~= 'Noise59' && label ~= 'Noise60' && label~= 'Noise61' && label~= 'Noise62' && label~= 'Noise63' && label~= 'Noise64' && label~= 'Noise65' && label~= 'Noise66' && label~= 'Noise67' && label~= 'Noise68' && label~= 'Noise69' && label~= 'Noise70' && label ~= 'Noise71' && label ~= 'Noise72' && label ~= 'Noise73' && label ~= 'Noise74' && label ~= 'Noise75' && label ~= 'Noise76' && label ~= 'Noise77' && label ~= 'Noise78' && label ~= 'Noise79' && label ~= 'Noise80')
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
    
    result = 1;
    
else
    
    result = 0;
    
end