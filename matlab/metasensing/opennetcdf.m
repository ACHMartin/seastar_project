function data= opennetcdf(filename, varname)


ncidx = netcdf.open(filename,'NOWRITE');
var=netcdf.inqVarID(ncidx,varname);
data   = netcdf.getVar(ncidx,var);
netcdf.close(ncidx);
  
end