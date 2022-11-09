function [inc_angle, alpha_s, look_angle] = incangle_from_netcdf( master_full_name, speedup_factor)
%master_full_name='V:\Capella\radiometric_problem\SAR_CPLX_20190822075219_9.6G_HH_12_pres_2_fdc_246.sar.rgo.sig.nc';
%master_full_name='V:\Capella\radiometric_problem\SAR_CPLX_20190822065240_9.6G_HH_12_pres_2_fdc_246.sar.rgo.sig.nc.corr'


z_axis=opennetcdf(master_full_name, 'DEMImage');
x_axis=opennetcdf(master_full_name,'CrossRange');
y_axis=opennetcdf(master_full_name,'GroundRange');

lookdirec = opennetcdf(master_full_name,'LookDirection');
if lookdirec(1)== 'R'
y_axis=abs(y_axis);
end

[orb_x, orb_y, orb_z]=localxyz_from_netcdf(master_full_name);

if ~exist('speedup_factor', 'var')
    speedup_factor_x=1;%50;
    speedup_factor_y=1;%50;
else
    speedup_factor_x=speedup_factor(1);
    speedup_factor_y=speedup_factor(2);
end

xlen=length(x_axis);
ylen=length(y_axis);
xaxis= interp1( 1:xlen,x_axis, 1:(xlen-1)/ceil((xlen-1)/speedup_factor_x):xlen);
yaxis= interp1( 1:ylen,y_axis, 1:(ylen-1)/ceil((ylen-1)/speedup_factor_y):ylen);

[xaxis2, yaxis2]=meshgrid(xaxis, yaxis);
[x_axis2, y_axis2]=meshgrid(x_axis, y_axis);
yaxis22=yaxis2;

orbx= interp2( x_axis2, y_axis2 ,orb_x, xaxis2, yaxis2);
orby= interp2( x_axis2, y_axis2 ,orb_y, xaxis2, yaxis2);
orbz= interp2( x_axis2, y_axis2 ,orb_z, xaxis2, yaxis2);

zaxis= interp2( x_axis2, y_axis2 ,z_axis, xaxis2, yaxis2);

tamimg=size(zaxis);

dispstat('','init'); % One time only initialization
dispstat(sprintf('computing angles...'),'keepthis','timestamp');
for l=1: tamimg(1)
    for k=1: tamimg(2)
        
        
        
        
        
        
        
        lm1=l-1;
        lp1=l+1;
        km1=k-1;
        kp1=k+1;
        
        
        
        if lm1 == 0
            lm1=1;
        end
        if lp1 > tamimg(1)
            lp1=tamimg(1) ;
        end
        if km1 == 0
            km1=1;
        end
        if kp1 > tamimg(2)
            kp1=tamimg(2) ;
        end
        
        
        
        zaxis1=zaxis(lm1:lp1,km1:kp1);
        rim=sqrt((orbx(lm1:lp1,km1:kp1)-xaxis2(lm1:lp1,km1:kp1)).^2 + (orby(lm1:lp1,km1:kp1)-yaxis22(lm1:lp1,km1:kp1)).^2 + (orbz(lm1:lp1,km1:kp1)-zaxis(lm1:lp1,km1:kp1)).^2);
        rgr=sqrt((orbx(lm1:lp1,km1:kp1)-xaxis2(lm1:lp1,km1:kp1)).^2 + (orby(lm1:lp1,km1:kp1)-yaxis22(lm1:lp1,km1:kp1)).^2);
        
        
        vetorp=[xaxis2(l,k), yaxis22(l,k), zaxis(l,k)];
        vetoro=[(orbx(l,k))', (orby(l,k))', (orbz(l,k))'];
        %vetorv=[(orbvx)', (orbvy)', (orbvz)'];
        %vetorvu=[(orbvx)', (orbvy)', 0]/sqrt( (orbvx).^2 +(orbvy).^2);
        %vetorvu=[(orbvx)', (orbvy)', (orbvz)']/sqrt( (orbvx).^2 +(orbvy).^2+ (orbvz).^2 );
        vetor=vetorp-vetoro;
        vetoru=vetor;
        vetoru(:)=0;
        vetoruuu=vetor;
        vetoruuu(:)=0;
        vetoruuu(:,3)=1;%look angle suplement
        vetoru(:,1)=1;
        
        escalarprod=sum(vetor.*vetoru,2);
        magvetor=sqrt(vetor(:,1).^2+vetor(:,2).^2+vetor(:,3).^2);
        theta=acos(escalarprod./(magvetor) )-pi/2;
        escalarprod=sum(vetor.*vetoruuu,2);
        magvetor=sqrt(vetor(:,1).^2+vetor(:,2).^2+vetor(:,3).^2);
        look_angle(l,k)=pi-acos(escalarprod./(magvetor) );
        
        
        
        
        tamrgr=size(rgr);
        deltarggroundx=rgr-circshift(rgr,[0,1]);
        deltarggroundx(:,1)=NaN;
        %deltarggroundx=mean(deltarggroundx(:), 'omitnan');
        deltarggroundx=mean(deltarggroundx(isnan(deltarggroundx) == 0 ));
        deltarggroundy=rgr-circshift(rgr,[1,0]);
        deltarggroundy(1,:)=NaN;
        %deltarggroundy=mean(deltarggroundy(:), 'omitnan');
        deltarggroundy=mean(deltarggroundy(isnan(deltarggroundy) == 0));
        
        deltazaxisx=zaxis1-circshift(zaxis1,[0,1]);
        deltazaxisx(:,1)=NaN;
        %deltazaxisx=mean(deltazaxisx(:), 'omitnan');
        deltazaxisx=mean(deltazaxisx(isnan(deltazaxisx)==0));
        deltazaxisy=zaxis1-circshift(zaxis1,[1,0]);
        deltazaxisy(1,:)=NaN;
        %deltazaxisy=mean(deltazaxisy(:), 'omitnan');
        deltazaxisy=mean(deltazaxisy(isnan(deltazaxisy) == 0));
        deltargground=deltarggroundy*cos(theta)+deltarggroundx*sin(theta);
        deltazaxis= deltazaxisy*cos(theta)+deltazaxisx*sin(theta);
        alpha_s=atan(deltazaxis./deltargground);
        inc_angle(l,k)=look_angle(l,k)-alpha_s;
        
        
        
    end
    dispstat(sprintf('Progress %f%%',single(l)/single(tamimg(1))*100),'timestamp');
    
    
    
end
dispstat('.','keepprev');


[vers, versd]=version;
versiondate=char(strsplit(versd, ' '));
if versiondate(3,1) == '2' & versiondate(3,2) == '0' & versiondate(3,3) == '1' & versiondate(3,4) == '3'
    look_angle=inpaint_nans(look_angle,0);
else
    look_angle = fillmissing(look_angle,'nearest');
    look_angle = fillmissing(look_angle,'nearest',2);
end
look_angle= interp2( xaxis2, yaxis2 ,look_angle, x_axis2, y_axis2);

if versiondate(3,1) == '2' & versiondate(3,2) == '0' & versiondate(3,3) == '1' & versiondate(3,4) == '3'
    inc_angle=inpaint_nans(inc_angle,0);
else
    inc_angle = fillmissing(inc_angle,'nearest');
    inc_angle = fillmissing(inc_angle,'nearest',2);
end
inc_angle= interp2( xaxis2, yaxis2 ,inc_angle, x_axis2, y_axis2);




end
