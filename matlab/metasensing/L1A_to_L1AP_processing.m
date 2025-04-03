function L1A_to_L1AP_processing(L1A_file_path)
%L1A_TO_L1AP_PROCESSING Processes OSCAR L1A data to L1AP
%
%   Function to process OSCAR L1A data files to L1AP level. Adds
%   Incidence angle and Squint fields, as well as global attributes. Function
%   takes the path to L1A files as input and processes all L1A files within
%   the supplied directory, saving L1AP files in a new L1AP folder mirroring
%   the directory structure of the L1A files following the OSCAR file
%   structure specification.

[L1A_file_list, num_L1A_files] = build_L1A_file_list(L1A_file_path);

for file = 1 : num_L1A_files

    [L1A_file_name] = get_L1A_file_name(L1A_file_list, file);

    [L1AP_file_path, L1A_file_path_split] = build_L1AP_file_path(L1A_file_path);

    % copy L1A file to L1AP directory to be worked on
    if ~exist(L1AP_file_path, 'dir')
        mkdir(L1AP_file_path)
    end
    L1AP_file_name = ['L1AP_', L1A_file_name];
    copyfile([L1A_file_path, L1A_file_name], [L1AP_file_path, L1AP_file_name], 'f')

    disp('---------------------------------------------------------------')
    disp(['Processing file ', L1AP_file_name])

    [processing_version] = get_processing_version_from_file('../../_version.py');
    [data_version] = get_data_version_from_file_path_split(L1A_file_path_split);
    [campaign_name] = get_campaign_name_from_config(L1AP_file_name);
    [track_name] = get_track_name_from_campaign_config(campaign_name, L1AP_file_path, L1AP_file_name);
    [start_time, end_time] = read_start_end_time_from_L1A(L1AP_file_path, L1AP_file_name);
    write_global_attributes_to_L1AP_netcdf(L1AP_file_path, L1AP_file_name, ...
        campaign_name, track_name, start_time, end_time, processing_version, data_version)
    delete_title_attribute_from_L1AP_netcdf(L1AP_file_path, L1AP_file_name)
    add_incidence_angle_and_look_angle_to_L1AP_netcdf(L1AP_file_path, L1AP_file_name);

    add_squint_to_L1AP_netcdf(L1AP_file_path, L1AP_file_name);

    add_orbit_images_to_L1AP_netcdf(L1AP_file_path, L1AP_file_name)
end

end



function [L1A_file_list, num_L1A_files] = build_L1A_file_list(L1A_file_path)
%BUILD_L1A_FILE_LIST Constructs list of OSCAR L1A files for processing
%

if ispc
    L1A_file_list = ls([L1A_file_path, '*.nc']);
    num_L1A_files = size(L1A_file_list, 1);
else
    L1A_file_list_dir = dir([L1A_file_path, '*.nc']); %UNIX uses dir explicitly
    L1A_file_list_cells = struct2cell(L1A_file_list_dir);
    num_L1A_files = size(L1A_file_list_cells, 2);
    L1A_file_list = cell(num_L1A_files, 1);
    for file_index = 1:num_L1A_files
        L1A_file_list(file_index) = L1A_file_list_cells(1,file_index);
    end
end
end

function [L1AP_file_path, L1A_file_path_split] = build_L1AP_file_path(L1A_file_path)
%BUILD_L1AP_FILE_PATH Constructs OSCAR L1AP file path
%

if ispc
    L1A_file_path_components = split(L1A_file_path,'\'); % WINDOWS
else
    L1A_file_path_components = split(L1A_file_path,'/'); % UNIX
end
% Build L1AP save path
L1AP_file_path_components = L1A_file_path_components;
L1AP_file_path_components{find(strcmp(L1A_file_path_components, 'L1A'))} = 'L1AP';
if ispc
    L1AP_file_path = cell2mat(join(L1AP_file_path_components,'\'));
    L1A_file_path_split = split(L1A_file_path,'\');
else
    L1AP_file_path = cell2mat(join(L1AP_file_path_components,'/'));
    L1A_file_path_split = split(L1A_file_path,'/');
end
end


function [L1A_file_name] = get_L1A_file_name(L1A_file_list, file)
%GET_L1A_FILE_NAME Gets L1A file name to process
%

if ispc
    L1A_file_name = [L1A_file_list(file, :)];
else
    L1A_file_name = L1A_file_list{file};
end
end

function [processing_version] = get_processing_version_from_file(version_file)
%GET_PROCESSING_VERSION_FROM_FILE Gets SeaSTAR project version
%
%   Reads Python _version.py to retrieve __version__ parameter

text = fileread(version_file);
version_file_as_cells = regexp(text, '\n', 'split');
mask = ~cellfun(@isempty, strfind(version_file_as_cells, '__version__ = '));
version_string = string(erase(cell2mat(version_file_as_cells(mask)), '__version__ = '));
version_string = regexp(version_string, '\D+', 'split');
processing_version = join(version_string(~cellfun('isempty',version_string)),'.');
end

function [data_version] = get_data_version_from_file_path_split(L1A_file_path_split)
%GET_DATA_VERSION_FROM_FILE_PATH_SPLIT Get OSCAR data version
%
%   Extracts data version from file path information.

data_version_match = regexp(L1A_file_path_split,'\<v[0-9]\w*','match');
data_version = cell2mat(data_version_match{~cellfun(@isempty,data_version_match)});
end

function [campaign_name] = get_campaign_name_from_config(L1AP_file_name)
%GET_CAMPAIGN_NAME_FROM_CONFIG Get OSCAR campaign name from Campaign_name_lookup.ini
%

ini_file = INI('File','../../config/Campaign_name_lookup.ini');
ini_struct = ini_file.read();
expression = '\d*T\d*';
pat = regexpPattern(expression);
L1AP_file_name_split = split(L1AP_file_name, '_');
L1AP_file_time = unique(cell2mat(L1AP_file_name_split(cellfun(@(x) ~isempty(strfind(x,pat)),L1AP_file_name_split))), 'rows');
campaign_name = ini_struct.OSCAR_campaigns.(['x', L1AP_file_time(1:6)]);
end

function [track_name] = get_track_name_from_campaign_config(campaign_name, L1AP_file_path, L1AP_file_name)
%GET_TRACK_NAME_FROM_CAMPAIGN_CONFIG Gets OSCAR track name
%
% Reads track_names.ini from ../../config/ to match Track name from DAR to track time

track_time_for_ini_search = ['x',erase(ncreadatt([L1AP_file_path, L1AP_file_name],"/","Title"), 'Track : ')];
ini_file = INI('File',['../../config/' campaign_name '_TrackNames.ini']);
ini_struct = ini_file.read();
track_name = ini_struct.(track_time_for_ini_search(1:9)).(track_time_for_ini_search);
end

function [start_time, end_time] = read_start_end_time_from_L1A(L1AP_file_path, L1AP_file_name)
%READ_START_END_TIME_FROM_L1A Reads OSCAR L1A acquisition start and end time
%
%   Reads start and end times from L1A NetCDF, formatted YYYYMMDDTHHMM

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
end

function write_global_attributes_to_L1AP_netcdf(L1AP_file_path, L1AP_file_name, ...
    campaign_name, track_name, start_time, end_time, processing_version, data_version)
%WRITE_GLOBAL_ATTRIBUTES_TO_L1AP_NETCDF Writes global attributes
%
% Writes global attributes to L1AP file

ncwriteatt([L1AP_file_path, L1AP_file_name], '/','Campaign', campaign_name)
ncwriteatt([L1AP_file_path, L1AP_file_name], '/','Platform', 'OSCAR')
ncwriteatt([L1AP_file_path, L1AP_file_name], '/','ProcessingLevel', 'L1AP')
ncwriteatt([L1AP_file_path, L1AP_file_name], '/','Track',track_name)
ncwriteatt([L1AP_file_path, L1AP_file_name], '/','StartTime',start_time)
ncwriteatt([L1AP_file_path, L1AP_file_name], '/','EndTime',end_time)
[CrossRange_resolution, GroundRange_resolution, resolution_string] = compute_grid_resolution_string(L1AP_file_path, L1AP_file_name);
ncwriteatt([L1AP_file_path, L1AP_file_name], '/','Resolution',resolution_string);
ncwriteatt([L1AP_file_path, L1AP_file_name], '/','CrossRangeResolution',CrossRange_resolution);
ncwriteatt([L1AP_file_path, L1AP_file_name], '/','GroundRangeResolution',GroundRange_resolution);
ncwriteatt([L1AP_file_path, L1AP_file_name], '/','Codebase','seastar_project');
ncwriteatt([L1AP_file_path, L1AP_file_name], '/','CodeVersion', processing_version)
ncwriteatt([L1AP_file_path, L1AP_file_name], '/','Comments', 'Processed on ' + string(today("datetime")))
ncwriteatt([L1AP_file_path, L1AP_file_name], '/','Repository','https://github.com/NOC-EO/seastar_project');
ncwriteatt([L1AP_file_path, L1AP_file_name], '/','DataVersion', data_version);
end

function add_incidence_angle_and_look_angle_to_L1AP_netcdf(L1AP_file_path, L1AP_file_name)
%ADD_INCIDENCE_ANGLE_AND_LOOK_ANGLE_TO_L1AP_NETCDF Adds Incidence Angle and
%Look Angle to L1AP NetCDF file
%
%   Computes IncidenceAngleImage and LookAngleImage then adds them to the
%   L1AP NetCDF file along with variable attributes.

[info, num_vars, dim_list, var_list] = return_netcdf_info_and_dims(L1AP_file_path, L1AP_file_name);

DEMImage=ncread([L1AP_file_path, L1AP_file_name],'DEMImage');
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
end

function add_squint_to_L1AP_netcdf(L1AP_file_path, L1AP_file_name)
%ADD_SQUINT_TO_L1AP_NETCDF Adds Squint to L1AP NetCDF file
%
%   Computes SquintImage and SquintMounted then adds them to the L1AP
%   NetCDF file along with variable attributes.

[info, num_vars, dim_list, var_list] = return_netcdf_info_and_dims(L1AP_file_path, L1AP_file_name);

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
end
function add_orbit_images_to_L1AP_netcdf(L1AP_file_path, L1AP_file_name)
%ADD_ORBIT_IMAGES_TO_L1AP_NETCDF Computes and adds ObitHeadingImage and
%OrbitYawImage to L1AP netcdf file along with variable attributes
%

[info, num_vars, dim_list, var_list] = return_netcdf_info_and_dims(L1AP_file_path, L1AP_file_name);

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

function delete_title_attribute_from_L1AP_netcdf(L1AP_file_path, L1AP_file_name)
%DELETE_TITLE_ATTRIBUTE_FROM_L1AP_NETCDF Deletes redundant 'Title' global
%attribute
%

% Open netCDF file.
ncid = netcdf.open([L1AP_file_path, L1AP_file_name], 'NC_WRITE');
% Put file in define mode to delete an attribute.
netcdf.reDef(ncid);
% Delete the global attribute in the netCDF file.
netcdf.delAtt(ncid,netcdf.getConstant('GLOBAL'), 'Title');
% Return file to data mode.
netcdf.endDef(ncid)
% Verify that the global attribute was deleted.
[numdims, numvars, numatts, unlimdimID] = netcdf.inq(ncid);
global_attributes_list = cell(numatts, 1);
for attnum=1:numatts
    global_attributes_list{attnum} = netcdf.inqAttName(ncid,netcdf.getConstant('NC_GLOBAL'), attnum - 1);
end
if isempty(find(strcmp(global_attributes_list, 'Title')))
    disp('Title attribute successfully deleted...')
else
    disp('Unable to delete Title attribute.')
end
netcdf.close(ncid)

end

function [info, num_vars, dim_list, var_list] = return_netcdf_info_and_dims(L1AP_file_path, L1AP_file_name)
%RETURN_NETCDF_INFO_AND_DIMS Returns netcdf file information, variable list
%and dimensions
%

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

end

function [CrossRange_resolution, GroundRange_resolution, resolution_string] = compute_grid_resolution_string(L1AP_file_path, L1AP_file_name)
%COMPUTE_GRID_RESOLUTION_STRING Builds a grid resolution string
%
% Builds a grid resolution string for attributes writing using the
% CrossRange and GroundRange data arrays. Returns an exception if the grids
% have variable resolution.
%

CrossRange = ncread([L1AP_file_path, L1AP_file_name],'CrossRange');
GroundRange = ncread([L1AP_file_path, L1AP_file_name],'CrossRange');
CrossRange_resolution = unique(diff(CrossRange));
GroundRange_resolution = unique(diff(GroundRange));
% Construct an MException object to represent the error.
errID = 'compute_grid_resolution_string:VariableResolution';
msg = 'Unable to assign single grid resolution as grid spacing is not uniform';
baseException = MException(errID,msg);
if length(CrossRange_resolution) == 1 && length(GroundRange_resolution) == 1
    resolution_string = [sprintf('%03d', CrossRange_resolution),...
        'x', sprintf('%03d', GroundRange_resolution),'m'];
else
     throw(baseException)
end
end