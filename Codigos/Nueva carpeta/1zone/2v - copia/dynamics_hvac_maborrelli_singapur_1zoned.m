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

%cp=params.cp;
%C1=params.C1;
%C2=params.C2;
%R=params.R;
%Roa=params.Roa;

%disturbances
%Toa=u(1);
%Pd=u(2);

%inputs
%ms=u(3);
%DT=u(4);
delta=u(5);

%current states

%intermediate variable 
%disp(delta*Tr + (1-delta)*Toa + DT)
Ts=delta*x(1)+(1-delta)*u(1)+u(4);

%state dynamics

dT(1)=x(1)+(ST*u(3)*params.cp*((delta*x(1)+(1-delta)*u(1)+u(4))-x(1))+ ST*(x(2)-x(1))/params.R+...
    ST*(u(1)-x(1))/params.Roa+ST*u(2))/params.C1;
dT(2)=x(2)+ST*(x(1)-x(2))/params.R/params.C2;


