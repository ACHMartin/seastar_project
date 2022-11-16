%clear
close all
% Fore
%filename="D:\data\SEASTAR\SEASTARex\Data\Metasensing\OSCAR\Brest_Sample_Oct_13_2022\SAR_CPLX_20220517T093239_13.5G_VV_33_pres_1_fdc_auto.sar.sig_INF_SAR_CPLX_20220517T093239_13.5G_VV_34_pres_1_fdc_auto.sar.sig.oph.nc";
%Aft
%filename="D:\data\SEASTAR\SEASTARex\Data\Metasensing\OSCAR\Brest_Sample_Oct_13_2022\SAR_CPLX_20220517T093239_13.5G_VV_77_pres_1_fdc_auto.sar.sig_INF_SAR_CPLX_20220517T093239_13.5G_VV_78_pres_1_fdc_auto.sar.sig.oph.nc";
%Mid
filename="D:\data\SEASTAR\SEASTARex\Data\Metasensing\OSCAR\Brest_Sample_Oct_13_2022\SAR_CPLX_20220517T093239_13.5G_VV_00_pres_1_fdc_auto.sar.sig.nc";

if ~exist('IncidenceAngleImage','var')
[IncidenceAngleImage, alpha_s, LookAngleImage]=inc_from_netcdf(filename);
end

[fdc, SquintImage, squintx]=squint_from_netcdf_monostatic(filename);
squint_tx=nan;
squint_rx=nan;
SquintImage=round(SquintImage.*pi/180*10000.)./10000;
squint_tx=round(squint_tx.*pi/180*10000.)./10000;
squint_rx=round(squint_rx.*pi/180*10000.)./10000;

squintx=round(squintx.*pi/180*10000.)./10000;