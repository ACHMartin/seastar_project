% Script to add potentially missing Incidence Angle, Look Angle and Squint
% fields to a folder of OSCAR netcdf datasets. Applies the Incidence Angle
% and Squint computations supplied in Matlab .m format by Metasensing.

file_path = [uigetdir(),'\'];
file_list = ls([file_path, '*.nc']);
num_files = size(file_list, 1);

for file = 1 : num_files
    file_name = [file_list(file, :)];
    info = ncinfo([file_path, file_name]);
    num_vars = size(info.Variables, 2);
    var_list=cell(num_vars,1);
    for i = 1 : num_vars
        var_list{i} = info.Variables(i).Name;
    end
    num_dims = size(info.Dimensions, 2);
    dim_list=cell(num_dims,1);
    for i = 1 : num_dims
        dim_list{i,1} = info.Dimensions(i).Length;

    end
    DEMImage=ncread([file_path, file_name],'DEMImage');
    disp('---------------------------------------------------------------')
    disp(['Processing file ',file_name]) 
    if sum(ismember(var_list,'IncidenceAngleImage')) == 1 &&...
            all(info.Variables(find(strcmp(var_list,'IncidenceAngleImage'))).Size ==...
            [info.Variables(find(strcmp(var_list,'GroundRange'))).Size,info.Variables(find(strcmp(var_list,'CrossRange'))).Size])
        disp('IncidenceAngleImage already present in netcdf file. Passing...')
    else
        [IncidenceAngleImage, alpha_s, LookAngleImage] = inc_from_netcdf([file_path, file_name]);
        IncidenceAngleImage(isnan(DEMImage))=NaN;
        IncidenceAngleImage(DEMImage==0)=NaN;
        IncidenceAngleImage=rad2deg(IncidenceAngleImage);
        LookAngleImage=rad2deg(LookAngleImage);
        [m,n] = size(IncidenceAngleImage);
        nccreate([file_path, file_name],'IncidenceAngleImage','Dimensions',...
            {info.Dimensions(find(cell2mat(dim_list(:,1)) == m)).Name, m,...
            info.Dimensions(find(cell2mat(dim_list(:,1)) == n)).Name, n},...
            'Format','netcdf4')
        ncwrite([file_path, file_name], 'IncidenceAngleImage', IncidenceAngleImage)
        ncwriteatt([file_path, file_name], 'IncidenceAngleImage','long_name','Incidence angle')
        ncwriteatt([file_path, file_name], 'IncidenceAngleImage','units','deg')
        ncwriteatt([file_path, file_name], 'IncidenceAngleImage','description','Incidence angle between nadir and the ground for each pixel in the image')
        nccreate([file_path, file_name],'LookAngleImage','Dimensions',...
            {info.Dimensions(find(cell2mat(dim_list(:,1)) == m)).Name, m,...
            info.Dimensions(find(cell2mat(dim_list(:,1)) == n)).Name, n},...
            'Format','netcdf4')
        ncwrite([file_path, file_name], 'LookAngleImage', LookAngleImage)
        ncwriteatt([file_path, file_name], 'LookAngleImage','long_name','Look angle')
        ncwriteatt([file_path, file_name], 'LookAngleImage','units','deg')
        ncwriteatt([file_path, file_name], 'LookAngleImage','description','Look angle between nadir and the beam pointing direction for each pixel in the image')
    end
    if sum(ismember(var_list,'SquintImage')) == 1 &&...
            all(info.Variables(find(strcmp(var_list,'SquintImage'))).Size ==...
            [info.Variables(find(strcmp(var_list,'GroundRange'))).Size,info.Variables(find(strcmp(var_list,'CrossRange'))).Size])
        disp('SquintImage already present in netcdf file. Passing...')
    else
        [fdc, Squint_slant, squintx, fdcg, Squint_ground, squintxg] = squint_from_netcdf_monostatic([file_path, file_name]);
        SquintImage = Squint_ground; 
        SquintMounted = Squint_ground;
        squint_tx = nan;
        squint_rx = nan;
        SquintImage = round(SquintImage.*pi/180*10000.)./10000;
        SquintMounted = round(SquintMounted.*pi/180*10000.)./10000;
        squint_tx = round(squint_tx.*pi/180*10000.)./10000;
        squint_rx = round(squint_rx.*pi/180*10000.)./10000;
        squintx = round(squintx.*pi/180*10000.)./10000;
        SquintImage=rad2deg(SquintImage);
        [m,n] = size(SquintImage);
        nccreate([file_path, file_name],'SquintImage','Dimensions',...
            {info.Dimensions(find(cell2mat(dim_list(:,1)) == m)).Name, m,...
            info.Dimensions(find(cell2mat(dim_list(:,1)) == n)).Name, n},...
            'Format','netcdf4')
        ncwrite([file_path, file_name], 'SquintImage', SquintImage)
        ncwriteatt([file_path, file_name], 'SquintImage','long_name','Ground squint')
        ncwriteatt([file_path, file_name], 'SquintImage','units','deg')
        ncwriteatt([file_path, file_name], 'SquintImage','description','Beam squint along the ground for each pixel in the image')
        
        SquintMounted=rad2deg(SquintMounted);
        [m,n] = size(SquintMounted);
        nccreate([file_path, file_name],'SquintMounted','Dimensions',...
            {info.Dimensions(find(cell2mat(dim_list(:,1)) == m)).Name, m,...
            info.Dimensions(find(cell2mat(dim_list(:,1)) == n)).Name, n},...
            'Format','netcdf4')
        ncwrite([file_path, file_name], 'SquintMounted', SquintImage)
        ncwriteatt([file_path, file_name], 'SquintMounted','long_name','Mounted squint')
        ncwriteatt([file_path, file_name], 'SquintMounted','units','deg')
        ncwriteatt([file_path, file_name], 'SquintMounted','description','Beam squint along the line of sight for each pixel in the image. Also called Mounted squint angle')

    end
    if sum(ismember(var_list,'OrbitHeadingImage')) == 1 &&...
            all(info.Variables(find(strcmp(var_list,'OrbitHeadingImage'))).Size ==...
            [info.Variables(find(strcmp(var_list,'GroundRange'))).Size,info.Variables(find(strcmp(var_list,'CrossRange'))).Size])
        disp('OrbitHeadingImage already present in netcdf file. Passing...')
    else
        [OrbitHeadingImage, OrbitYawImage] = yaw_and_heading_from_netcdf([file_path, file_name]);
        [m,n] = size(OrbitHeadingImage);
        nccreate([file_path, file_name],'OrbitHeadingImage','Dimensions',...
            {info.Dimensions(find(cell2mat(dim_list(:,1)) == m)).Name, m,...
            info.Dimensions(find(cell2mat(dim_list(:,1)) == n)).Name, n},...
            'Format','netcdf4')
        ncwrite([file_path, file_name], 'OrbitHeadingImage', OrbitHeadingImage)
        ncwriteatt([file_path, file_name], 'OrbitHeadingImage','long_name','Orbit heading')
        ncwriteatt([file_path, file_name], 'OrbitHeadingImage','units','deg')
        ncwriteatt([file_path, file_name], 'OrbitHeadingImage','description','Heading relative to North of the sensor for each pixel in the image')

        [m,n] = size(OrbitYawImage);
        nccreate([file_path, file_name],'OrbitYawImage','Dimensions',...
            {info.Dimensions(find(cell2mat(dim_list(:,1)) == m)).Name, m,...
            info.Dimensions(find(cell2mat(dim_list(:,1)) == n)).Name, n},...
            'Format','netcdf4')
        ncwrite([file_path, file_name], 'OrbitYawImage', OrbitYawImage)
        ncwriteatt([file_path, file_name], 'OrbitYawImage','long_name','Orbit yaw')
        ncwriteatt([file_path, file_name], 'OrbitYawImage','units','deg')
        ncwriteatt([file_path, file_name], 'OrbitYawImage','description','Gimbal yaw for each pixel in the image')
    end

end




