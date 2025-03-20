% Script to add potentially missing Incidence Angle, Look Angle and Squint
% fields to a folder of OSCAR netcdf datasets. Applies the Incidence Angle
% and Squint computations supplied in Matlab .m format by Metasensing.

L1A_file_path = [uigetdir(),'\'];
file_list = ls([L1A_file_path, '*.nc']);
num_files = size(file_list, 1);

for file = 1 : num_files
    L1A_file_name = [file_list(file, :)];
    % Decompose file path components
    if isscalar(split(L1A_file_path,'/'))
        OS = 'WINDOWS';
        file_path_components = split(L1A_file_path,'\'); % WINDOWS
    else
        OS = 'UNIX';
        file_path_components = split(L1A_file_path,'/'); % UNIX
    end
    % Build L1AP save path
    save_path_components = file_path_components;
    save_path_components{find(strcmp(file_path_components, 'L1A'))} = 'L1AP';
    if strcmp(OS, 'WINDOWS')
        save_path = cell2mat(join(save_path_components,'\'));
        file_path_split = split(L1A_file_path,'\');
    elseif strcmp(OS, 'UNIX')
        save_path = cell2mat(join(save_path_components,'/'));
        file_path_split = split(L1A_file_path,'/');
    end
    % copy L1A file to L1AP directory to be worked on
    if ~exist(save_path, 'dir')
        mkdir(save_path)
    end
    L1AP_file_name = ['L1AP_', L1A_file_name];
    copyfile([L1A_file_path, L1A_file_name], [save_path, L1AP_file_name], 'f')
    % Set file path to save path
    L1AP_file_path = save_path;


    info = ncinfo([L1AP_file_path, L1AP_file_name]);
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
    DEMImage=ncread([L1AP_file_path, L1AP_file_name],'DEMImage');
    disp('---------------------------------------------------------------')
    disp(['Processing file ', L1AP_file_name])

    % Read Python _version.py to retrieve __version__ parameter
    text = fileread('../../_version.py ');
    version_file_as_cells = regexp(text, '\n', 'split');
    mask = ~cellfun(@isempty, strfind(version_file_as_cells, '__version__ = '));
    version_string = string(erase(cell2mat(version_file_as_cells(mask)), '__version__ = '));
    version_string = regexp(version_string, '\D+', 'split');
    processing_version = join(version_string(~cellfun('isempty',version_string)),'.');

    % Extract data version from file path
    data_version_match = regexp(file_path_split,'\<v[0-9]\w*','match');
    data_version = cell2mat(data_version_match{~cellfun(@isempty,data_version_match)});

    % Extract campaign name from file path
    campaign_name_match = regexp(file_path_split,'\<[0-9]{6,6}_\w*','match');
    campaign_name = cell2mat(campaign_name_match{~cellfun(@isempty,a)});

    % Read track_names.ini to match Track name from DAR to track time
    track_time_for_ini_search = ['x',erase(ncreadatt([L1AP_file_path, L1AP_file_name],"/","Title"), 'Track : ')];
    ini_file = INI('File',['../../config/' campaign_name '_TrackNames.ini']);
    ini_struct = ini_file.read();
    track_name = ini_struct.(track_time_for_ini_search(1:9)).(track_time_for_ini_search);

    % Read start and end times, formatted YYYYMMDDTHHMM
    start_year = ncread([L1AP_file_path, L1AP_file_name],'StartYear');
    start_month = ncread([L1AP_file_path, L1AP_file_name],'StartMonth');
    start_day = ncread([L1AP_file_path, L1AP_file_name],'StartDay');
    start_hour = ncread([L1AP_file_path, L1AP_file_name],'StartHour');
    start_minute = ncread([L1AP_file_path, L1AP_file_name],'StartMin');
    start_time = sprintf('%02d%02d%02dT%02d%02d',[start_year,start_month,start_day, start_hour, start_minute]);

    end_year = ncread([L1AP_file_path, L1AP_file_name],'FinalYear');
    end_month = ncread([L1AP_file_path, L1AP_file_name],'FinalMonth');
    end_day = ncread([L1AP_file_path, L1AP_file_name],'FinalDay');
    end_hour = ncread([L1AP_file_path, L1AP_file_name],'FinalHour');
    end_minute = ncread([L1AP_file_path, L1AP_file_name],'FinalMin');
    end_time = sprintf('%02d%02d%02dT%02d%02d',[end_year,end_month,end_day, end_hour, end_minute]);

    % Write global attributes
    ncwriteatt([L1AP_file_path, L1AP_file_name], '/','Campaign', campaign_name)
    ncwriteatt([L1AP_file_path, L1AP_file_name], '/','Platform', 'OSCAR')
    ncwriteatt([L1AP_file_path, L1AP_file_name], '/','ProcessingLevel', 'L1AP')
    ncwriteatt([L1AP_file_path, L1AP_file_name], '/','Track',track_name)
    ncwriteatt([L1AP_file_path, L1AP_file_name], '/','StartTime',start_time)
    ncwriteatt([L1AP_file_path, L1AP_file_name], '/','EndTime',end_time)
    ncwriteatt([L1AP_file_path, L1AP_file_name], '/','Codebase','seastar_project');
    ncwriteatt([L1AP_file_path, L1AP_file_name], '/','CodeVersion', processing_version)
    ncwriteatt([L1AP_file_path, L1AP_file_name], '/','Comments', 'Processed on ' + string(today("datetime")))
    ncwriteatt([L1AP_file_path, L1AP_file_name], '/','Repository','https://github.com/NOC-EO/seastar_project');
    ncwriteatt([L1AP_file_path, L1AP_file_name], '/','DataVersion', data_version);

    if sum(ismember(var_list,'IncidenceAngleImage')) == 1 &&...
            all(info.Variables(find(strcmp(var_list,'IncidenceAngleImage'))).Size ==...
            [info.Variables(find(strcmp(var_list,'GroundRange'))).Size,info.Variables(find(strcmp(var_list,'CrossRange'))).Size])
        disp('IncidenceAngleImage already present in netcdf file. Passing...')
    else
        [IncidenceAngleImage, alpha_s, LookAngleImage] = inc_from_netcdf([L1AP_file_path, L1AP_file_name]);
        IncidenceAngleImage(isnan(DEMImage))=NaN;
        IncidenceAngleImage(DEMImage==0)=NaN;
        IncidenceAngleImage=rad2deg(IncidenceAngleImage);
        LookAngleImage=rad2deg(LookAngleImage);
        [m,n] = size(IncidenceAngleImage);
        nccreate([L1AP_file_path, L1AP_file_name],'IncidenceAngleImage','Dimensions',...
            {info.Dimensions(find(cell2mat(dim_list(:,1)) == m)).Name, m,...
            info.Dimensions(find(cell2mat(dim_list(:,1)) == n)).Name, n},...
            'Format','netcdf4')
        ncwrite([L1AP_file_path, L1AP_file_name], 'IncidenceAngleImage', IncidenceAngleImage)
        ncwriteatt([L1AP_file_path, L1AP_file_name], 'IncidenceAngleImage','long_name','Incidence angle')
        ncwriteatt([L1AP_file_path, L1AP_file_name], 'IncidenceAngleImage','units','deg')
        ncwriteatt([L1AP_file_path, L1AP_file_name], 'IncidenceAngleImage','description','Incidence angle between nadir and the ground for each pixel in the image')
        nccreate([L1AP_file_path, L1AP_file_name],'LookAngleImage','Dimensions',...
            {info.Dimensions(find(cell2mat(dim_list(:,1)) == m)).Name, m,...
            info.Dimensions(find(cell2mat(dim_list(:,1)) == n)).Name, n},...
            'Format','netcdf4')
        ncwrite([L1AP_file_path, L1AP_file_name], 'LookAngleImage', LookAngleImage)
        ncwriteatt([L1AP_file_path, L1AP_file_name], 'LookAngleImage','long_name','Look angle')
        ncwriteatt([L1AP_file_path, L1AP_file_name], 'LookAngleImage','units','deg')
        ncwriteatt([L1AP_file_path, L1AP_file_name], 'LookAngleImage','description','Look angle between nadir and the beam pointing direction for each pixel in the image')
    end
    if sum(ismember(var_list,'SquintImage')) == 1 &&...
            all(info.Variables(find(strcmp(var_list,'SquintImage'))).Size ==...
            [info.Variables(find(strcmp(var_list,'GroundRange'))).Size,info.Variables(find(strcmp(var_list,'CrossRange'))).Size])
        disp('SquintImage already present in netcdf file. Passing...')
    else
        [fdc, Squint_slant, squintx, fdcg, Squint_ground, squintxg] = squint_from_netcdf_monostatic([L1AP_file_path, L1AP_file_name]);
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
        nccreate([L1AP_file_path, L1AP_file_name],'SquintImage','Dimensions',...
            {info.Dimensions(find(cell2mat(dim_list(:,1)) == m)).Name, m,...
            info.Dimensions(find(cell2mat(dim_list(:,1)) == n)).Name, n},...
            'Format','netcdf4')
        ncwrite([L1AP_file_path, L1AP_file_name], 'SquintImage', SquintImage)
        ncwriteatt([L1AP_file_path, L1AP_file_name], 'SquintImage','long_name','Ground squint')
        ncwriteatt([L1AP_file_path, L1AP_file_name], 'SquintImage','units','deg')
        ncwriteatt([L1AP_file_path, L1AP_file_name], 'SquintImage','description','Beam squint along the ground for each pixel in the image')

        SquintMounted=rad2deg(SquintMounted);
        [m,n] = size(SquintMounted);
        nccreate([L1AP_file_path, L1AP_file_name],'SquintMounted','Dimensions',...
            {info.Dimensions(find(cell2mat(dim_list(:,1)) == m)).Name, m,...
            info.Dimensions(find(cell2mat(dim_list(:,1)) == n)).Name, n},...
            'Format','netcdf4')
        ncwrite([L1AP_file_path, L1AP_file_name], 'SquintMounted', SquintImage)
        ncwriteatt([L1AP_file_path, L1AP_file_name], 'SquintMounted','long_name','Mounted squint')
        ncwriteatt([L1AP_file_path, L1AP_file_name], 'SquintMounted','units','deg')
        ncwriteatt([L1AP_file_path, L1AP_file_name], 'SquintMounted','description','Beam squint along the line of sight for each pixel in the image. Also called Mounted squint angle')

    end
    if sum(ismember(var_list,'OrbitHeadingImage')) == 1 &&...
            all(info.Variables(find(strcmp(var_list,'OrbitHeadingImage'))).Size ==...
            [info.Variables(find(strcmp(var_list,'GroundRange'))).Size,info.Variables(find(strcmp(var_list,'CrossRange'))).Size])
        disp('OrbitHeadingImage already present in netcdf file. Passing...')
    else
        [OrbitHeadingImage, OrbitYawImage] = yaw_and_heading_from_netcdf([L1AP_file_path, L1AP_file_name]);
        [m,n] = size(OrbitHeadingImage);
        nccreate([L1AP_file_path, L1AP_file_name],'OrbitHeadingImage','Dimensions',...
            {info.Dimensions(find(cell2mat(dim_list(:,1)) == m)).Name, m,...
            info.Dimensions(find(cell2mat(dim_list(:,1)) == n)).Name, n},...
            'Format','netcdf4')
        ncwrite([L1AP_file_path, L1AP_file_name], 'OrbitHeadingImage', OrbitHeadingImage)
        ncwriteatt([L1AP_file_path, L1AP_file_name], 'OrbitHeadingImage','long_name','Orbit heading')
        ncwriteatt([L1AP_file_path, L1AP_file_name], 'OrbitHeadingImage','units','deg')
        ncwriteatt([L1AP_file_path, L1AP_file_name], 'OrbitHeadingImage','description','Heading relative to North of the sensor for each pixel in the image')

        [m,n] = size(OrbitYawImage);
        nccreate([L1AP_file_path, L1AP_file_name],'OrbitYawImage','Dimensions',...
            {info.Dimensions(find(cell2mat(dim_list(:,1)) == m)).Name, m,...
            info.Dimensions(find(cell2mat(dim_list(:,1)) == n)).Name, n},...
            'Format','netcdf4')
        ncwrite([L1AP_file_path, L1AP_file_name], 'OrbitYawImage', OrbitYawImage)
        ncwriteatt([L1AP_file_path, L1AP_file_name], 'OrbitYawImage','long_name','Orbit yaw')
        ncwriteatt([L1AP_file_path, L1AP_file_name], 'OrbitYawImage','units','deg')
        ncwriteatt([L1AP_file_path, L1AP_file_name], 'OrbitYawImage','description','Gimbal yaw for each pixel in the image')
    end

end




