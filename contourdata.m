function s=contourdata(c)
    if nargin<1 || ~isfloat(c) || size(c,1)~=2 || size(c,2)<4
       error('CONTOURDATA:rhs',...
             'Input Must be the 2-by-N Contour Matrix C.')
    end

    tol=1e-12;
    k=1;     %记录有几条等高线
    col=1;   %第几列

    while col<size(c,2)   % while less than total columns in c
       s(k).level = c(1,col); 
       s(k).number = c(2,col); 
       idx=col+1:col+c(2,col);%例如2：16
       s(k).x = c(1,idx); 
       s(k).y= c(2,idx); 
       s(k).isopen = abs( diff( c(1,idx([1 end])) ) ) >tol || ...  %或
                     abs( diff( c(2,idx([1 end])) ) ) >tol ; %diff差分函数   计算首位数值是否相同
       k=k+1;%等高线条数+1
       col=col+c(2,col)+1;
    end
end