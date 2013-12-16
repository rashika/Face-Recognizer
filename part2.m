% 68 classes and 42 images


t=load('CMUPIEData'); 
num=0;
% t.CMUPIEData(1).label
% traindata = t.CMUPIEData(1).pixels;
newimg = zeros(1024,2380);%2856
testimg=zeros(1024,476);%714
test_temp=zeros(1024,476);
trainlabel=zeros(2380,1);
testlabel=zeros(476,1);
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

eigen_arr=[5,10,15,20,25,30,35,40,45,50,60,70,80,90,100];
accuracy=[];
for iter=1:size(eigen_arr,2),
    eigenfaces=zeros(size(sorted_eigenvector,1),eigen_arr(iter));
    for j=3:eigen_arr(iter)+2,
        for i=1:size(sorted_eigenvector,1),
            eigenfaces(i,j-2)=sorted_eigenvector(i,j);
        end
    end
    % eigenfaces;
    trainweight = zeros(size(eigenfaces,2),size(deviate_img,2));

    for i=1:size(deviate_img,2),
        trainweight(:,i) = eigenfaces'*deviate_img(:,i);
    end

    imgweight = zeros(size(eigenfaces,2),1);

    % % sample = 1*10
    % % training = 570*10
    % % Group (training labels)  
%     k=5;

    

    % k_arr=[2,3,5,7,9];
    k_arr=[2];
    for kcount=1:size(k_arr,2),
        acc=0;
        for i=1:size(testimg,2),
            for j=1:size(mean_img,1),
                test_temp(j,i)=testimg(j,i)-mean_img(j,1);
            end
            imgweight = eigenfaces'*test_temp(:,i);
            assignedlabel=knnclassify(imgweight',trainweight',trainlabel,k_arr(kcount));
            assignedlabel;
            if (assignedlabel==testlabel(i))
                acc = acc+1;
            end
        end
        (acc*100)/size(testimg,2)
        accuracy=[accuracy,acc/size(testimg,2)];
    end
end
%  plot(k_arr,accuracy,'b');
%  xlabel('K value in K-NN classifier');
%  ylabel('Accuracy');
%  title('K values v/s Accuracy in CMU Dataset','FontSize',16);
%  axis([0 12 0 2])

 plot(eigen_arr,accuracy,'-b');
 xlabel('Number of Eigen Vectors');
 ylabel('Accuracy');
 title('Eigen Vectors v/s Accuracy in CMU Dataset','FontSize',16);
axis([0 100 0 2]) 
