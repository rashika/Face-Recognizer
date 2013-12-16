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
 fclose(fileID1);
  fclose(fileID2);
   fclose(fileID3);
 scale = 100;

 fid1 = fopen('trainlabel.txt');
%  [trainlabel,testlabel]=createlabel(fid1,fid2);
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
 
 % 38*20 = 760
 % testing=190
 % training= 570
 
 newimg = zeros(scale*scale,570);
 testimg = zeros(scale*scale,190);
 test_temp = zeros(scale*scale,190);
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
 fclose(fid);

 mean_img = zeros(scale*scale,1);
 deviate_img = zeros(size(newimg,1),size(newimg,2));
 
for i=1:size(newimg,1),
     mean_img(i,1) = mean2(newimg(i,:));
     for j=1:size(newimg,2),
        deviate_img(i,j) = newimg(i,j) - mean_img(i,1);
     end
end
 
% for i=1:size(deviate_img,1),
%     mean_img(i,1) = mean2(deviate_img(i,:));
% end

[eigenvector,eigenvalue] = eig(deviate_img'*deviate_img);
new_eigenvalue = eig(deviate_img'*deviate_img);
new_eigenvalue;
v_i = deviate_img * eigenvector;

for i=1:size(v_i,2),
    v_i(:,i) = v_i(:,i)/norm(v_i(:,i));
end

[sorted index] = sort(new_eigenvalue,'descend');
sorted_eigenvector= v_i(:,index);

accuracy=[];
% eigen_arr=[5,10,15,20,25,30,35,40,45,50,60,70,80,90,100];
eigen_arr=[60];
for iter=1:size(eigen_arr,2),
    eigenfaces=zeros(size(sorted_eigenvector,1),eigen_arr(iter));
    for j=3:eigen_arr(iter)+2,
        for i=1:size(sorted_eigenvector,1),
            eigenfaces(i,j-2)=sorted_eigenvector(i,j);
        end
    end
    eigenfaces;
    trainweight = zeros(size(eigenfaces,2),size(deviate_img,2));

    for i=1:size(deviate_img,2),
        trainweight(:,i) = eigenfaces'*deviate_img(:,i);
    end

    imgweight = zeros(size(eigenfaces,2),1);

    % sample = 1*10
    % training = 570*10
    % Group (training labels)  
    % accuracy=zeros(1,size(k,2));
    
    k_arr=[2,3,5,7,9];
%     k_arr=[2];
    
     for kcount=1:size(k_arr,2),
    %      k_arr(kcount)
         acc = 0;
        for i=1:size(testimg,2),
            for j=1:size(mean_img,1),
                test_temp(j,i)=testimg(j,i)-mean_img(j,1);
            end
            imgweight = eigenfaces'*test_temp(:,i);
            assignedlabel=knnclassify(imgweight',trainweight',trainlabel,k_arr(kcount));      
            assignedlabel;
            if (strcmp(assignedlabel,testlabel(i)) ~= 0)
                acc = acc+1;
            end
        end
        (acc*100)/size(testimg,2)
        accuracy=[accuracy,acc/size(testimg,2)];
     end
end
 
%  plot(eigen_arr,accuracy,'b');
%  xlabel('Number of Eigen Vectors');
%  ylabel('Accuracy');
%  title('Eigen Vectors v/s Accuracy in Yale Dataset','FontSize',16);
% axis([0 100 0 2]) 
 plot(k_arr,accuracy,'b');
 xlabel('K value in K-NN classifier');
 ylabel('Accuracy');
 title('K values v/s Accuracy in Yale Dataset','FontSize',16);
 axis([0 12 0 2])
% end




