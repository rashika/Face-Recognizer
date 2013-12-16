function [] = part1_verif(testimage,testlabel)
    
     scale=100;
     dirname = 'Data';
     s = dir(dirname);
     isub=[s(:).isdir];
     namef={s(isub).name}';
     namef(ismember(namef,{'.','..'}))=[];
     fileID1 = fopen('images.txt','w');
%      namef
     for i = 1:size(namef,1),
         imgfolder = strcat(dirname,'/',namef{i});
         fold=dir(imgfolder);
         for j=3:size(fold,1),
%            re= regexp(fold(j).name,'_','split');
           imagename=strcat(imgfolder,'/',fold(j).name);
           fprintf(fileID1,'%s\n',imagename);           
         end
     end
     fclose(fileID1);
     
     fid = fopen('images.txt');
     img1 = fgetl(fid);
     newimg = zeros(scale*scale,760);
     count_img=1;

     while ischar(img1)      
             A = imread(img1);
             B = imresize(A, [scale scale]);
             for j=1:scale,
                 for k=1:scale,
                     newimg((j-1)*scale+k,count_img)=B(j,k);                 
                 end         
             end 
             count_img=count_img+1;
             img1 = fgetl(fid);
     end
     
    tval=threshold(newimg);

      testimg = zeros(scale*scale,1);
      A = imread(testimage);
      B = imresize(A, [scale scale]);
        for j=1:scale,
            for k=1:scale,
               testimg((j-1)*scale+k,1)=B(j,k);                 
            end         
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

    trainweight = zeros(size(eigenfaces,2),size(deviate_img,2));

    for i=1:size(deviate_img,2),
        trainweight(:,i) = eigenfaces'*deviate_img(:,i);
    end
    
    distance=[];
    imgweight = zeros(size(eigenfaces,2),1);

     numk=0.1;
     for i=1:size(testimg,2),
         for j=1:size(mean_img,1),
             testimg(j,i)=testimg(j,i)-mean_img(j,1);
         end
         imgweight = eigenfaces'*testimg(:,i);
         for l=1:20,
            sum = 0;
            for k=1:size(imgweight,1),
                  if(str2num(testlabel)>13)
                      label=str2num(testlabel)-1;
                  else
                      label=str2num(testlabel);
                  end
                  sum=sum+(trainweight(k,(label-1)*20+l)-imgweight(k))*(trainweight(k,(label-1)*20+l)-imgweight(k));
            end
            distance = [distance,sqrt(sum)];
         end
         ct = 0;
         for l=1:20,
             if(distance(l) < numk*tval)
                 ct=ct+1;
             end
         end
        if (ct >= 3)
            disp('Yes they match.')
        else
            disp('They do not match.')
        end
     end

end

function[tval]=threshold(img)
    tval=0;
    euclid=[];
    labels=[];
   
    for i=1:38,
        sum=[];
        j=1;
        class=(i-1)*20;
        while(j<14)
            rand_num=randi(20,1,2);
            flag=0;
            numsum=rand_num(1)+rand_num(2);
            for k=1:j-1,
                if(numsum==sum(k))
                    flag=1;
                    break;
                end
            end
            if(flag~=1)
                sum=[sum,numsum]; 
                labels=[labels,0];  
                eusum = 0;
                for k=1:size(img,1),
                    eusum=eusum+(img(k,class+rand_num(1))-img(k,class+rand_num(2)))*(img(k,class+rand_num(1))-img(k,class+rand_num(2)));
                end
                euclid=[euclid,sqrt(eusum)];
                j=j+1;
            end
        end
    end    
    
    for i=1:38,
        class1=(i-1)*20;
        for j=1:13,
            rand_num=randi(20,1,2);
            class2 = randi(38,1,1);
            while (class2==i)
                class2 = randi(38,1,1);
            end
            class2=(class2-1)*20; 
            labels=[labels,1];  
            eusum = 0;
            for k=1:size(img,1),
                eusum=eusum+(img(k,class1+rand_num(1))-img(k,class2+rand_num(2)))*(img(k,class1+rand_num(1))-img(k,class2+rand_num(2)));
            end
            euclid=[euclid,sqrt(eusum)];
        end
    end
    
    [fp, tp, tval_arr, area, opt] = perfcurve(labels,euclid,1);
     
    plot(fp,tp,'b');
%      plot(fp,fp,'--r');
    xlabel('False Positive Rate');
    ylabel('True Positive Rate');
    title('ROC Curve','FontSize',16);
    axis([0 1 0 1]);
    opt
    
    for i=1:988,
        if(opt(1)==fp(i) && opt(2)==tp(i))
            tval = tval_arr(i-1);
            break;
        end
    end
    tval
end


