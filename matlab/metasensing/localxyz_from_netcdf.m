
function [orbX1, orbY1, orb_img_Hei_1, orb_img_time_1, navX1, navY1, navZ1,  orbX2, orbY2, orb_img_Hei_2, orb_img_time_2, navX2, navY2, navZ2 ]=localxyz_from_netcdf(filename1)
%Function to convert orbit postion form lat and lon given in the netcdf file to the 
% groundrange (Y) crosrange (X) local coordinates  used to defien the global projection grid
% iinput:
% filename 1 = full name of the netcdf file
% output:
% orbX1: orbit data in X coordinates for each pixel of the SAR image
% orbY1:  orbit data in Y coordinates for each pixel of the SAR image
% orb_img_Hei_1:  orbit data in Z coordinates for each pixel of the SAR image
% orb_img_time_1: time each pixel of the SAR image
% navX1:  orbit data in X coordinates
% navY1:  orbit data in Y coordinates
% navZ1:  orbit data in Y coordinates
% orbX2: TX (bistatic case) orbit data in X coordinates for each pixel of the SAR image
% orbY2: TX (bistatic case) orbit data in Y coordinates for each pixel of the SAR image
% orb_img_Hei_2: TX (bistatic case) orbit data in Z coordinates for each pixel of the SAR image
% orb_img_time_2: TX (bistatic case) time each pixel of the SAR image
% navX2:  TX (bistatic case) orbit data in X coordinates
% navY2:  TX (bistatic case) orbit data in Y coordinates
% navZ2:  TX (bistatic case) orbit data in Y coordinates
mstvector=opennetcdf(filename1,'GBPGridInfo');

UTM_x0=mstvector(1);
UTM_y0=mstvector(2);
xheading=mstvector(9);

systype= opennetcdf(filename1,'SystemType');

orb_img_Lat=opennetcdf(filename1,'OrbLatImage');
orb_img_Lon=opennetcdf(filename1,'OrbLonImage');
orb_img_Hei_1=opennetcdf(filename1,'OrbHeightImage');
orb_Lat=opennetcdf(filename1,'OrbitLatitude');
orb_Lon=opennetcdf(filename1,'OrbitLongitude');
navZ1=opennetcdf(filename1,'OrbitHeight');
orb_img_time_1=opennetcdf(filename1,'OrbTimeImage');

orbX2=nan;
orbY2=nan;
orb_img_Hei_2=nan;
orb_img_time_2=nan;
navX2=nan;
navY2=nan;
navZ2=nan;

if systype=='B'
orb_img_Lat2=opennetcdf(filename1,'OrbLatImage2');
orb_img_Lon2=opennetcdf(filename1,'OrbLonImage2');
orb_img_Hei_2=opennetcdf(filename1,'OrbHeightImage2');
orb_Lat2=opennetcdf(filename1,'OrbitLatitude2');
orb_Lon2=opennetcdf(filename1,'OrbitLongitude2');
navZ2=opennetcdf(filename1,'OrbitHeight2');
orb_img_time_1=opennetcdf(filename1,'OrbTimeImage');
end
lookdirec=opennetcdf(filename1,'LookDirection');
Zone_UTM=double(opennetcdf(filename1,'UTMZone'));
Emisphere_UTM=double(opennetcdf(filename1,'Hemisphere'));

if Emisphere_UTM ==1
  Emisphere_UTMs='S';
else
  Emisphere_UTMs='N';  
end

if lookdirec == 'L'
look_direction='left';
end

if lookdirec == 'R'
look_direction='right';
end


[orb_Ex, orb_Nx] = wgs2utm_v3(orb_img_Lat, orb_img_Lon, Zone_UTM, Emisphere_UTMs);
[orbX1,orbY1]    = UTMEN2SARXY(orb_Ex,orb_Nx, xheading,  [UTM_x0, UTM_y0], look_direction);


% [orb_Ex, orb_Nx] = wgs2utm_v3(orb_Lat,orb_Lon, Zone_UTM, Emisphere_UTMs);
% [navX1, navY1]    = UTMEN2SARXY(orb_Ex,orb_Nx, xheading,  [UTM_x0, UTM_y0], look_direction, 1);

if systype=='B'

[orb_Ex2, orb_Nx2] = wgs2utm_v3(orb_img_Lat2, orb_img_Lon2, Zone_UTM, Emisphere_UTMs);
[orbX2,orbY2]    = UTMEN2SARXY(orb_Ex2,orb_Nx2, xheading,  [UTM_x0, UTM_y0], look_direction);

% 
% [orb_Ex2, orb_Nx2] = wgs2utm_v3(orb_Lat2,orb_Lon2, Zone_UTM, Emisphere_UTMs);
% [navX2, navY2]    = UTMEN2SARXY(orb_Ex2,orb_Nx2, xheading,  [UTM_x0, UTM_y0], look_direction, 1);


end

end