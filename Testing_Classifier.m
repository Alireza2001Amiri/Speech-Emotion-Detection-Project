%testing...

% Extract features from test samples and preproccess it...

features_test = [];
labels_test = [];
numVectorsPerFile = [];
while hasdata(ADSTest)
    [audioIn,dsInfo] = read(ADSTest);
    
    feat = extract(afe,audioIn);

    isSpeech = feat(:,featureMap.shortTimeEnergy) > energyThreshold;
    isVoiced = feat(:,featureMap.zerocrossrate) < zcrThreshold;

    voicedSpeech = isSpeech & isVoiced;

    feat(~voicedSpeech,:) = [];
    numVec = size(feat,1);
    feat(:,[featureMap.zerocrossrate,featureMap.shortTimeEnergy]) = [];
    
    label = repelem(dsInfo.Label,numVec);
    
    numVectorsPerFile = [numVectorsPerFile,numVec];
    features_test = [features_test;feat];
    labels_test = [labels_test,label];
    dsInfo.FileName
end
features_test = (features_test-M)./S;

%predict label according to trainedClassifier...

prediction = predict(trainedClassifier,features_test);
prediction = categorical(string(prediction));

% plot the result table...

figure(Units="normalized",Position=[0.4 0.4 0.4 0.4])
confusionchart(labels_test(:),prediction,title="Test Accuracy (Per Frame)", ...
    ColumnSummary="column-normalized",RowSummary="row-normalized");


