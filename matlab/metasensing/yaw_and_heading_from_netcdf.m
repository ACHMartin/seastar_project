%%%%%%%%%%%%%
% Copyright MetaSensing, B.V.
% Airborne SAR Processing
% Version: 1.0.0
% Release date: 24-10-2022
function [orbheading, orbyaw]=yaw_and_heading_from_netcdf(master_full_name) 
%master_full_name='V:\Capella\radiometric_problem\SAR_CPLX_20190822075219_9.6G_HH_12_pres_2_fdc_246.sar.rgo.sig.nc';
 
 
  orbimg=opennetcdf(master_full_name,'OrbitImage');
  tamimg=size(orbimg);
  orbtime=zeros(tamimg(1), tamimg(2)) + NaN;
  orbheading=zeros(tamimg(1), tamimg(2)) + NaN;
  
 [orbX1, orbY1, orb_img_Hei_1, orb_time]=localxyz_from_netcdf(master_full_name);
 
  orb=opennetcdf(master_full_name,'GPSTime'); 
  orbh=opennetcdf(master_full_name,'OrbitHeading');
  mstvector=opennetcdf(master_full_name,'GBPGridInfo');

 orbimg(find(orbimg(:) >= length(orb)))=0;%sometimes orbimge comes with values greater then the  navigation size!           

        valid   = find(orbimg > 1);
   
                
        azm = orbimg(valid)';
        az  = floor(azm);
       orb0=orb-orb(1);
       orbtime(valid)= orb0(az)' + (azm-az).*(orb0(az+1) - orb0(az))'; % Explicit linear interpolation   
       orbtime(valid)=orbtime(valid)+orb(1,1);
   
       orbheading(valid)= orbh(az)' + (azm-az).*(orbh(az+1) - orbh(az))'; % Explicit linear interpolation   
       
       orbtime(find(isnan(orbimg)))=NaN;
       orbheading(find(isnan(orbimg)))=NaN;
      
       %estimate velocities from positions (mono case)
[dt_x,dt_y] =gradient(orb_time);
clear orb_time;
dt=dt_x+dt_y;
%dt=smooth2f_mat(dt,7,7);
clear dt_x dt_y;
winfx=3;%50*3;
winfy=3;%50*3;

 [dvx_x, dvx_y]=gradient(orbX1);
v_x=(dvx_x+ dvx_y)./dt;
clear dvx_x dvx_y;
v_x(isinf(v_x))=nan;
velmedia=mean(v_x(:),'omitnan');
if velmedia > 0
findneg=find(v_x <= 1);
v_x(findneg)=nan;
end
if velmedia < 0
findpos=find(v_x >= -1);
v_x(findpos)=nan;
end
v_x=fillmissing(v_x, 'nearest');
%v_x=smooth2f_mat(v_x,winfx,winfy);
v_x=medfilt2(v_x,[winfx,winfy]);

[dvy_x, dvy_y]=gradient(orbY1);
v_y=(dvy_x+ dvy_y)./dt;
clear dvy_x dvy_y;
v_y(isinf(v_y))=nan;
if velmedia > 0
v_y(findneg)=nan;
end
if velmedia < 0
v_y(findpos)=nan;
end
v_y=fillmissing(v_y, 'nearest');
%v_y=smooth2f_mat(v_y,winfx,winfy);
v_y=medfilt2(v_y,[winfx,winfy]);
 
[dvz_x, dvz_y]=gradient(orb_img_Hei_1);
v_z=(dvz_x+ dvz_y)./dt;
clear dvz_x dvz_y;
v_z(isinf(v_z))=nan;
if velmedia > 0
v_z(findneg)=nan;
end
if velmedia < 0
v_z(findpos)=nan;
end
v_z=fillmissing(v_z, 'nearest');
%v_z=smooth2f_mat(v_z,winfx,winfy);
v_z=medfilt2(v_z,[winfx,winfy]);
 
clear dt
       
       
       
%  [gradorbtimex,gradorbtimey] =gradient(orbtime);
%  orbtime  = 0
%  gradorbtime=gradorbtimex+gradorbtimey;
%  gradorbtimex = 0
%  gradorbtimey = 0
%  [vox21, vox22]=gradient(ox2);
%   vox2=(vox21+ vox22)./gradorbtime;
%   vox21  = 0
%   vox22 = 0
%  [voy21, voy22]=gradient(oy2);
%   voy2=(voy21+ voy22)./gradorbtime;
%     voy21  = 0
%   voy22 = 0
%  [voz21, voz22]=gradient(oz2);
%   voz2=(voz21+ voz22)./gradorbtime;
%  voz21  = 0
%   voz22 = 0
%  
% 
%   gradorbtime= 0;
  
 
%   
%  
%  lon=opennetcdf(master_full_name, 'LonImage');
%  lat=opennetcdf(master_full_name, 'LatImage');
%  hei=opennetcdf(master_full_name, 'DEMImage');
%  
% 
%   [x, y, z] = llh2xyz(lat, lon, hei, 0);
%  
% 
% 
%  lon    = 0;
%  lat   = 0;
%  hei   = 0;
%  
%  
% 
% line_of_sightx = ox2 - x;
% line_of_sighty = oy2 - y;
% line_of_sightz = oz2 - z;
% 
% normi= sqrt(line_of_sightx.^2    +line_of_sighty.^2  + line_of_sightz.^2 );
%  
% u_line_of_sightx = line_of_sightx ./normi;
% line_of_sightx= 0;
% u_line_of_sighty = line_of_sighty ./ normi;
% line_of_sighty= 0;
% u_line_of_sightz = line_of_sightz ./ normi;
% line_of_sightz= 0;

normi= sqrt(v_x.^2    +v_y.^2  + v_z.^2 );


 UTM_Xdir=mstvector(6);
 UTM_Ydir=mstvector(7);
xheading=atan2(UTM_Xdir, UTM_Ydir);
%direction = atan2(nav(6,:), nav(5,:))+xheading;
direction = atan2(v_y, v_x)+xheading;

  

    orbyaw=angle(exp(complex(0,(direction-orbheading.*pi/180)))).*180./pi;      

% 
% u_velocityx = vox2 ./ normi;
% vox2= 0;
% u_velocityy = voy2 ./ normi;
% voy2= 0;
% u_velocityz = voz2 ./ normi;
% voz2= 0;
% 
% dotx  = u_velocityx  .* u_line_of_sightx  ;
% meandotx  = mean(u_velocityx(:), 'omitnan')  .* u_line_of_sightx  ;
% doty  = u_velocityy  .* u_line_of_sighty  ;
% meandoty =  mean(u_velocityy(:), 'omitnan')  .* u_line_of_sighty  ;
% dotz  = u_velocityz  .* u_line_of_sightz  ;
% meandotz  = mean(u_velocityz(:), 'omitnan')  .* u_line_of_sightz  ;
% 
% squint = acos(dotx +   doty + dotz);
%  squint= squint*180/pi-90;
% meansquint = acos(meandotx +   meandoty + meandotz);
% meansquint= meansquint*180/pi-90;
% 
%  wave = physconst('LightSpeed')./opennetcdf(master_full_name, 'CentralFreq');
%  fdc= 2.*normi.*sind(squint)./wave;
%  squint= squint*pi/180;
 end