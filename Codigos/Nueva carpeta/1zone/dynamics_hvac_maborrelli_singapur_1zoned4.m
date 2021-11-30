function [dT,Ts]=dynamics_hvac_maborrelli_singapur_1zoned4(x,u,params,ST)


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

%inputs
ms=u(5);
DT=u(6);
DTH = u(7);
delta=u(8);

%current states
T=x;

%intermediate variable 
Tr=T(1);
%disp(delta*Tr + (1-delta)*Toa + DT)
Ts=delta*Tr+(1-delta)*Toa+DT+DTH;
COs = delta*T(3) + (1-delta)*CO2amb;
%state dynamics

dT(1)=Tr+(ST*ms*cp*((Ts)-T(1))+ ST*(T(2)-T(1))/R+...
    ST*(Toa-T(1))/Roa+ST*Pd)/C1;
dT(2)=T(2)+ST*(T(1)-T(2))/(R*C2);
dT(3) = T(3) + ST/(Vol*rhoair)* (ms*(COs - T(3)) + CO2gen/(Vol));%+ aux;
if dT(3) <= 0 && T(3) <= 0
    dT(3) = 0;
end



