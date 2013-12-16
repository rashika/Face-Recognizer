 dirname = 'Data';
 testdir = 'imageset';
 s = dir(dirname);
 isub=[s(:).isdir];
 namef={s(isub).name}';
 namef(ismember(namef,{'.','..'}))=[];
 fileID1 = fopen('images.txt','w');
 fileID2 = fopen('trainlabel.txt','w');
 trainlabel=cell(760,1);

 
 for i = 1:size(namef,1),
     imgfolder = strcat(dirname,'/',namef{i});
     fold=dir(imgfolder);
    for j=3:size(fold,1),
       re= regexp(fold(j).name,'_','split');
       imagename=strcat(imgfolder,'/',fold(j).name);
       fprintf(fileID1,'%s\n',imagename);
       fprintf(fileID2,'%7s\n',re{1});
    end
 end

 scale = 80;
 
 newimg = zeros(scale*scale,760);
 testimg = zeros(scale*scale,1);
 fid = fopen('images.txt');
 img = fgetl(fid);
 count=0;
 count_img=1;
 count_test=1;
 
 for i =1:1,
    imgname= strcat(testdir,'/','a.jpg');
    A = imread(imgname);
    for j=1:scale,
             for k=1:scale,
                 testimg((j-1)*scale+k,i)=A(j,k);                 
             end         
    end 
 end
 
 while ischar(img)             
         A = imread(img);
         B = imresize(A, [scale scale]);
         for j=1:scale,
             for k=1:scale,
                 newimg((j-1)*scale+k,count_img)=B(j,k);                 
             end         
         end 
         count_img=count_img+1;
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
% sorted_eigenvector

eigenfaces=zeros(size(sorted_eigenvector,1),500);
for j=3:502,
    for i=1:size(sorted_eigenvector,1),
        eigenfaces(i,j-2)=sorted_eigenvector(i,j);
    end
end

trainweight = zeros(size(eigenfaces,2),size(deviate_img,2));

for i=1:size(deviate_img,2),
    trainweight(:,i) = eigenfaces'*deviate_img(:,i);
end

imgweight = zeros(size(eigenfaces,2),1);
 
k=5;

accuracy=0;
for i=1:size(testimg,2),
    for j=1:size(mean_img,1),
        testimg(j,i)=testimg(j,i)-mean_img(j,1);
    end
    imgweight = eigenfaces'*testimg(:,i);
    resultimg = eigenfaces*imgweight;
    
    for j=1:size(mean_img,1),
        resultimg(j,1)=resultimg(j,1)+mean_img(j,1);
    end
     finalimg = reshape(resultimg,[scale scale]);
%      imagesc(finalimg')
%      colormap('gray')
    imshow(finalimg',[]);
end

