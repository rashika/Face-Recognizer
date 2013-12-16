function[]= part2_verif(testimage,testlabel)
% 68 classes and 42 image

re= regexp(testimage,'\.','split');
imageindex = str2num(re{1});

t=load('CMUPIEData'); 

newimg = zeros(1024,2856);%2856
testimg=zeros(1024,1);
testimg(:,1)= (t.CMUPIEData(imageindex).pixels)';

for i=1:2856,
        newimg(:,i)=(t.CMUPIEData(i).pixels)';
end 

 tval = threshold(newimg);

 mean_img = zeros(size(newimg,1),1);
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

eigenfaces=zeros(size(sorted_eigenvector,1),20);

for j=3:22,
    for i=1:size(sorted_eigenvector,1),
        eigenfaces(i,j-2)=sorted_eigenvector(i,j);
    end
end

trainweight = zeros(size(eigenfaces,2),size(deviate_img,2));

for i=1:size(deviate_img,2),
    trainweight(:,i) = eigenfaces'*deviate_img(:,i);
end

imgweight = zeros(size(eigenfaces,2),1);


    for j=1:size(mean_img,1),
        testimg(j,1)=testimg(j,1)-mean_img(j,1);
    end
    
    imgweight = eigenfaces'*testimg(:,1);
    distance=[];
    numk=0.3;
    for l=1:42,
            sum = 0;
            for k=1:size(imgweight,1),
                  label = str2num(testlabel);
                  sum=sum+(trainweight(k,(label-1)*42+l)-imgweight(k))*(trainweight(k,(label-1)*42+l)-imgweight(k));
            end
            distance = [distance,sqrt(sum)];
    end
    ct = 0;
    for l=1:42,
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

function[tval]=threshold(img)
    tval=0;
    euclid=[];
    labels=[];
   
    for i=1:68,
        sum=[];
        j=1;
        class=(i-1)*42;
        while(j<15)
            rand_num=randi(42,1,2);
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
    
    for i=1:68,
        class1=(i-1)*42;
        for j=1:14,
            rand_num=randi(42,1,2);
            class2 = randi(68,1,1);
            while (class2==i)
                class2 = randi(68,1,1);
            end
            class2=(class2-1)*42; 
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
    xlabel('False Positive Rate');
    ylabel('True Positive Rate');
    title('ROC Curve','FontSize',16);
    axis([0 1 0 1]);
    opt
    
    for i=1:1904, %68*14*2
        if(opt(1)==fp(i) && opt(2)==tp(i))
            tval = tval_arr(i-1);
            break;
        end
    end
    tval
end