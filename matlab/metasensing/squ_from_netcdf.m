function [fdc, squint, squintx, fdc_tx, fdc_rx, squint_tx, squint_rx] = squ_from_netcdf(netcdf_fname)
%
%input:
%netcdf_fname: Netcdf complete file name
%
%output:
%fdc : Doopler centroid in Hz
% squint: squint in radians

SystemType= opennetcdf(netcdf_fname,'SystemType');

if SystemType=='M'
    [fdc, squint, squintx]=squint_from_netcdf_monostatic(netcdf_fname);
    squint_tx=nan;
    squint_rx=nan;
end
if SystemType=='B'
    [fdc, squint, squintx, fdc_tx, fdc_rx, squint_tx, squint_rx]=squint_from_netcdf_bistatic(netcdf_fname);
end

squint=round(squint.*pi/180*10000.)./10000;
squint_tx=round(squint_tx.*pi/180*10000.)./10000;
squint_rx=round(squint_rx.*pi/180*10000.)./10000;

squintx=round(squintx.*pi/180*10000.)./10000;

% squint=squint.*pi/180;
% squint_tx=squint_tx.*pi./180;
% squint_rx=squint_rx.*pi./180;

end