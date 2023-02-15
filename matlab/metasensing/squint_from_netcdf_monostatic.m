function [fdc, squint, squintx, fdcg, squintg, squintxg] = squint_from_netcdf_monostatic(netcdf_fname)

fc = opennetcdf(netcdf_fname,'CentralFreq');
lookdirec = opennetcdf(netcdf_fname,'LookDirection');

[orbX1, orbY1, orb_img_Hei_1, orb_time]=localxyz_from_netcdf(netcdf_fname);

ox=opennetcdf(netcdf_fname,'CrossRange');
oy=opennetcdf(netcdf_fname,'GroundRange');
oz=opennetcdf(netcdf_fname,'DEMImage');

tam=size(oz);
ox=repmat(ox,1,tam(1))';
oy=repmat(oy,1,tam(2));

if lookdirec(1)== 'R'
oy=abs(oy);
end

%estimate velocities from positions
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

vnorm=sqrt(v_x.^2+v_y.^2+v_z.^2);
vnormg=sqrt(v_x.^2+v_y.^2);


%compute normalized LOS vector
vetorx=ox-orbX1;
vetory=oy-orbY1;
vetorz=oz-orb_img_Hei_1;
norm = sqrt(vetorx.^2+vetory.^2+vetorz.^2);
los_ux = vetorx./norm;
los_uy = vetory./norm;
los_uz = vetorz./norm;

normg=sqrt(vetorx.^2+vetory.^2);

%compute Doppler centroid frequency
wavelength = physconst('LightSpeed')./fc;
fdc = (v_x.*los_ux + v_y.*los_uy + v_z.*los_uz)*2/wavelength;

%fdc = reshape(fdc, size_tgt_lon);
squint=asin(fdc./(2.*vnorm).*wavelength).*180/pi;
squintx=-(acos(vetorx./(norm) )-pi/2).*180/pi;

%compute Doppler centroid frequency
fdcg = (v_x.*vetorx./normg + v_y.*vetory./normg)*2/wavelength;

%fdc = reshape(fdc, size_tgt_lon);
squintg=asin(fdcg./(2.*vnormg).*wavelength).*180/pi;
squintxg=-(acos(vetorx./(normg) )-pi/2).*180/pi;

end