% 68 classes and 42 images


t=load('CMUPIEData'); 
num=0;
% t.CMUPIEData(1).label
% traindata = t.CMUPIEData(1).pixels;
newimg = zeros(1024,2380);%2856
testimg=zeros(1024,476);
trainlabel=zeros(2380,1);
testlabel=zeros(476,1);  %7 * 68 = 476
counttest=1;
counttrain=1;
for i=1:2856,
    if(mod(i,6)~=num)
        newimg(:,counttrain)=(t.CMUPIEData(i).pixels)';
        trainlabel(counttrain,1)=t.CMUPIEData(i).label;
        counttrain=counttrain+1;
    else
        testimg(:,counttest)=(t.CMUPIEData(i).pixels)';
        testlabel(counttest,1)=t.CMUPIEData(i).label;
        counttest=counttest+1;        
    end
end 
%  trainlabel
%  testlabel
% newismg
 

 mean_img = zeros(size(newimg,1),1);
 deviate_img = zeros(size(newimg,1),size(newimg,2));

for i=1:size(newimg,1),
     mean_img(i,1) = mean2(newimg(i,:));
     for j=1:size(newimg,2),
        deviate_img(i,j) = newimg(i,j) - mean_img(i,1);
     end
end
% deviate_img

[eigenvector,eigenvalue] = eig(deviate_img'*deviate_img);
new_eigenvalue = eig(deviate_img'*deviate_img);
new_eigenvalue;
v_i = deviate_img * eigenvector;

for i=1:size(v_i,2),
    v_i(:,i) = v_i(:,i)/norm(v_i(:,i));
end

[sorted index] = sort(new_eigenvalue,'descend');
sorted_eigenvector= v_i(:,index);

eigenfaces=zeros(size(sorted_eigenvector,1),10);
for j=3:12,
    for i=1:size(sorted_eigenvector,1),
        eigenfaces(i,j-2)=sorted_eigenvector(i,j);
    end
end
% eigenfaces;
trainweight = zeros(size(eigenfaces,2),size(deviate_img,2));

for i=1:size(deviate_img,2),
    trainweight(:,i) = eigenfaces'*deviate_img(:,i);
end

imgweight = zeros(size(eigenfaces,2),476);

% % sample = 1*10
% % training = 570*10
% % Group (training labels)  
k=5;

% accuracy=0;
for i=1:size(testimg,2),
    for j=1:size(mean_img,1),
        testimg(j,i)=testimg(j,i)-mean_img(j,1);
    end
    imgweight(:,i) = eigenfaces'*testimg(:,i);
%     assignedlabel=knnclassify(imgweight',trainweight',trainlabel,k);
%     assignedlabel;
%     if (assignedlabel==testlabel(i))
%         accuracy = accuracy+1;
%     end
end

probab=[];
addpath('libsvm/');
%     trainlabel
model = svmtrain(trainlabel,trainweight','-t 0');
%     model
[results,accuracy,probab] = svmpredict(testlabel,imgweight',model); 
results;
accuracy
% (accuracy*100)/size(testimg,2)
