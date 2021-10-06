function [dT,Ts]=dynamics_hvac_maborrelli_singapur_1zoned(x,u,params,ST)

% nz=par.nz;
% Vert=par.Vert; 
% Neigh=par.Neigh;
% cp=par.cp;
% C1=par.C1;
% C2=par.C2;
% R=par.R;
% Rij=par.Rij;
% Roa=par.Roa;

cp=params.cp;
C1=params.C1;
C2=params.C2;
R=params.R;
Roa=params.Roa;

%disturbances
Toa=u(1);
Pd=u(2);

%inputs
ms=u(3);
DT=u(4);
delta=u(5);

%current states
T=x;

%intermediate variable 
Tr=T(1);
%disp(delta*Tr + (1-delta)*Toa + DT)
Ts=delta*Tr+(1-delta)*Toa+DT;

%state dynamics

dT(1)=Tr+(ST*ms*cp*((delta*Tr+(1-delta)*Toa+DT)-T(1))+ ST*(T(2)-T(1))/R+...
    ST*(Toa-T(1))/Roa+ST*Pd)/C1;
dT(2)=T(2)+ST*(T(1)-T(2))/R/C2;


