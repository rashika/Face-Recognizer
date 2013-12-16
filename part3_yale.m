 dirname = 'Data';
 s = dir(dirname);
 isub=[s(:).isdir];
 namef={s(isub).name}';
 namef(ismember(namef,{'.','..'}))=[];
 fileID1 = fopen('images.txt','w');
 fileID2 = fopen('trainlabel.txt','w');
 fileID3 = fopen('testlabel.txt','w');
 trainlabel=cell(570,1);
 testlabel=cell(190,1);
 
 num=3;
 countnum=0;
 for i = 1:size(namef,1),
     imgfolder = strcat(dirname,'/',namef{i});
     fold=dir(imgfolder);
    for j=3:size(fold,1),
       re= regexp(fold(j).name,'_','split');
       imagename=strcat(imgfolder,'/',fold(j).name);
       fprintf(fileID1,'%s\n',imagename);
       if(mod(countnum,4)~=num)
           fprintf(fileID2,'%7s\n',re{1});
       else
           fprintf(fileID3,'%7s\n',re{1});
       end
       countnum=countnum+1;
     end
 end

 scale = 100;

 fid1 = fopen('trainlabel.txt');
 trlbl = fgetl(fid1);
 count=1;
 while ischar(trlbl)
     trainlabel{count,1} = trlbl;
     count=count+1;
     trlbl = fgetl(fid1);
 end

 count = 1;
 fid2 = fopen('testlabel.txt');
 telbl = fgetl(fid2);
  while ischar(telbl)
      testlabel{count,1} = telbl;
      count=count+1;
      telbl = fgetl(fid2);
  end

 newimg = zeros(scale*scale,570);
 testimg = zeros(scale*scale,190);
 fid = fopen('images.txt');
 img = fgetl(fid);
 count=0;
 count_img=1;
 count_test=1;
 
 while ischar(img)    
     if(mod(count,4)~=num)     
         A = imread(img);
         B = imresize(A, [scale scale]);
         for j=1:scale,
             for k=1:scale,
                 newimg((j-1)*scale+k,count_img)=B(j,k);                 
             end         
         end 
         count_img=count_img+1;
     else
         A = imread(img);
         B = imresize(A, [scale scale]);
         for j=1:scale,
             for k=1:scale,
                 testimg((j-1)*scale+k,count_test)=B(j,k);                 
             end         
         end 
         count_test=count_test+1;
     end
     count=count+1;
     img = fgetl(fid);
 end

 mean_img = zeros(scale*scale,1);
 deviate_img = zeros(size(newimg,1),size(newimg,2));
 
for i=1:size(newimg,1),
     mean_img(i,1) = mean2(newimg(i,:));
     for j=1:size(newimg,2),
        deviate_img(i,j) = newimg(i,j) - mean_img(i,1);
     end
end

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
eigenfaces;
trainweight = zeros(size(eigenfaces,2),size(deviate_img,2));

for i=1:size(deviate_img,2),
    trainweight(:,i) = eigenfaces'*deviate_img(:,i);
end

imgweight = zeros(size(eigenfaces,2),190);  % 10 x 1
 
k=5;
trainlabel1=[];
countnew=1;
for i=1:570,
    trainlabel1=[trainlabel1,countnew];
    if(mod(i,15)==0)
        countnew=countnew+1;
    end
end
countnew=1;
trainlabel1;
testlabel1=[];
for i=1:190,
    testlabel1=[testlabel1,countnew];
    if(mod(i,5)==0)
        countnew=countnew+1;
    end
end
accuracy=[];
for i=1:size(testimg,2),
    for j=1:size(mean_img,1),
        testimg(j,i)=testimg(j,i)-mean_img(j,1);
    end
    imgweight(:,i) = eigenfaces'*testimg(:,i); %10 x 190
    
%     for k=1:190,
%         if (strcmp(results(k),testlabel(i)) ~= 0)
%             accuracy = accuracy+1;
%         end
%     end
end
imgweight;
probab=[];
addpath('libsvm/');
%     trainlabel
model = svmtrain(trainlabel1',trainweight','-t 0');
%     model
[results,accuracy,probab] = svmpredict(testlabel1',imgweight',model); 
results;
accuracy
% (accuracy*100)/size(testimg,2)

