function [inc_angle, alpha_s, look_angle, inc_angle_tx, look_angle_tx, inc_angle_rx, look_angle_rx]= inc_from_netcdf(netcdf_fname)
%
%input:
%netcdf_fname: Netcdf complete file name
%
%output:
%inc and look angle and slope angles

SystemType= opennetcdf(netcdf_fname,'SystemType');

if ~exist('speedup_factor', 'var')
    speedup_factor=[1,1];
end


if SystemType=='M'
    [inc_angle, alpha_s, look_angle]=incangle_from_netcdf(netcdf_fname,  speedup_factor);
    look_angle_tx=nan;
    inc_angle_tx=nan;
    look_angle_rx=nan;
    inc_angle_rx=nan;
end
if SystemType=='B'
     [inc_angle_rx, alpha_s, look_angle_rx]=incangle_from_netcdf(netcdf_fname,  speedup_factor);
     [inc_angle, alpha_s, look_angle]=incangle_from_netcdf_bistatic(netcdf_fname,  speedup_factor);
     [inc_angle_tx, alpha_s, look_angle_tx]=incangle_from_netcdf_tx(netcdf_fname,  speedup_factor);
end

inc_angle=round(inc_angle*10000.)./10000;
look_angle=round(look_angle*10000.)./10000;
inc_angle_tx=round(inc_angle_tx*10000.)./10000;
look_angle_tx=round(look_angle_tx*10000.)./10000;
inc_angle_rx=round(inc_angle_rx*10000.)./10000;
look_angle_rx=round(look_angle_rx*10000.)./10000;

inc_angle(inc_angle < 0)=nan;
look_angle(look_angle < 0)=nan;
inc_angle_tx(inc_angle_tx < 0)=nan;
look_angle_tx(look_angle_tx < 0)=nan;
inc_angle_rx(inc_angle_rx < 0)=nan;
look_angle_rx(look_angle_rx < 0)=nan;


% squint=squint.*pi/180;
% squint_tx=squint_tx.*pi./180;
% squint_rx=squint_rx.*pi./180;

end