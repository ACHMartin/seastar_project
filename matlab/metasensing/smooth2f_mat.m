function [out status] = smooth2f_mat(mat_in,nn, mm, tri, not_normalize_flag)


% smooths a matrix with a 'sliding box' of nn x mm dimensions using matlab
% filter2 function
%
% Adriano Meta 10/11/2015


if ((nn - floor(nn)) ~= 0 ) || (nn < 1) || ((mm - floor(mm)) ~= 0 ) || (mm < 1)
    disp('multilook factor must be a positive integer');
    status = 1;
    return;
end


[s1,s2]=size(mat_in);


if (nn > s1)
    disp('Warning: multilook factor reduced to match matrix size');
    nn = floor((s1-1)/2);
end

if (mm > s2)
    disp('Warning: multilook factor reduced to match matrix size');
    mm = floor((s2-1)/2);
end




if exist('tri', 'var')
    w1 = repmat(triang(nn), 1, mm) .* repmat(triang(mm).', nn, 1) ;
else
    w1 = ones(nn, mm);
end



if ~(exist('not_normalize_flag', 'var'))    
    norm_factor = sum(w1(:)); 
    out = filter2(w1/norm_factor, mat_in);
else
    out = filter2(w1, mat_in);
end



end