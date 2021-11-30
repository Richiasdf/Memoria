function [dT,Ts]=dynamics_hvac_maborrelli_singapur_1zone_tch(x,u,params,ST)


cp=params.cp;
C1=params.C1;
C2=params.C2;
R=params.R;
Roa=params.Roa;
Vol=params.Vol;
rhoair=params.rhoair;

%disturbances
Toa=u(1);
Pd=u(2);
CO2amb = u(3);
CO2gen = u(4);
Humamb = u(5);
Humgen = u(6);

%inputs
ms=u(7);
DT=u(8);
DTH = u(9);
delta=u(10);

%current states
T=x;

%intermediate variable 
Tr=T(1);
%disp(delta*Tr + (1-delta)*Toa + DT)
Ts1 = delta*Tr+(1-delta)*Toa+DT;
Ts=delta*Tr+(1-delta)*Toa+DT+DTH;
COs = delta*T(3) + (1-delta)*CO2amb;
%based on this code
% http://woodshole.er.usgs.gov/operations/sea-mat/air_sea-html/qsat.html
Pa=1013.25; ew = 6.1121*(1.0007+3.46e-6*Pa).*exp((17.502*Ts1)./(240.97+Ts1)); % in mb
Hsat_Tchilled  = 0.62197*(ew./(Pa-0.378*ew));

Hums = min( delta*T(4) + (1-delta)*Humamb, Hsat_Tchilled);
%state dynamics

dT(1)=Tr+(ST*ms*cp*((Ts)-T(1))+ ST*(T(2)-T(1))/R+...
    ST*(Toa-T(1))/Roa+ST*Pd)/C1;
dT(2)=T(2)+ST*(T(1)-T(2))/(R*C2);
%CO2
CO2dif = T(3) + ST/(Vol*rhoair)*ms*(COs - T(3));
%+ aux;

if CO2dif < 0
    dT(3) = min(T(3),COs) + (ST/(Vol*rhoair)*CO2gen/(Vol));
else
    dT(3) = T(3) + ST/(Vol*rhoair)* (ms*(COs - T(3)) + CO2gen/(Vol));
end
if T(3)<=0 || dT(3)<=0
    dT(3) = 0;
end


hum_min = min(T(4),Hums);
dT(4) = T(4) + (ST*ms*(Hums-T(4))+ Humgen/(rhoair*Vol))/(rhoair*Vol);
if T(4)+ ST*ms*(Hums-T(4))/(rhoair*Vol) < hum_min
    dT(4) = hum_min + Humgen/(rhoair*Vol);
end
if T(4)<=0 && dT(4)<=0
    dT(4)=0;
end
dT(5) =  Hsat_Tchilled;

end



