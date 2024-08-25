clc
clear all
%Store data in datastore
ADS = audioDatastore('D:\Alireza\University\DSP\Speech Emtion Detection\Dataset\AllDataSet','IncludeSubfolders',true,'LabelSource','foldernames');
[ADSTrain,ADSTest] = splitEachLabel(ADS,0.8);
% adsTrain
% adsTest
trainDatastoreCount = countEachLabel(ADSTrain);
%Pick a sample data and play it
[sampleTrain,dsInfo] = read(ADSTrain);
sound(sampleTrain,dsInfo.SampleRate);
%Feature extract with AFE and save it into 'afe' -->   MFCC,Pitch,ZCR,ShortTimeEnergy
fs = dsInfo.SampleRate;
windowLength = round(0.03*fs);
overlapLength = round(0.025*fs);
afe = audioFeatureExtractor(SampleRate=fs , Window=hamming(windowLength,"periodic"), OverlapLength=overlapLength, zerocrossrate=true,shortTimeEnergy=true,pitch=true,mfcc=true);
featureMap = info(afe);
%Store Features of all samples and label them until there is no sample left...
Features = [];
labels = [];
energyThreshold = 0.005;
zcrThreshold = 0.2; 

 
while hasdata(ADSTrain)
[audioIn,dsInfo] = read(ADSTrain);


feat = extract(afe,audioIn);

   isSpeech = feat(:,featureMap.shortTimeEnergy) > energyThreshold;
   isVoiced = feat(:,featureMap.zerocrossrate) < zcrThreshold;

   voicedSpeech = isSpeech & isVoiced;

feat(~voicedSpeech,:) = [];
    feat(:,[featureMap.zerocrossrate,featureMap.shortTimeEnergy]) = [];
    label = repelem(dsInfo.Label,size(feat,1));
    
    Features = [Features;feat];
    labels = [labels,label];
    dsInfo.FileName
end
% normalize and standardization...
M = mean(Features,1);
S = std(Features,[],1);
Features = (Features-M)./S;


