function [dx,Ts]=dynamics_hvac_maborrelli_singapur_T(x,u,params,ST)

%Parameters
nz = params.nz; 
Neigh = params.Neigh;
cp = params.cp;
C1 = params.C1;
C2 = params.C2;
R = params.R;
Rij = params.Rij;
Roa = params.Roa;
Vol = params.Vol;
rhoair = params.rhoair;

%disturbances

%temp ambiente
Toa = u(1);
%calor generado por objetos interiores y personas
Pd = u(2:nz+1);
%CO2 ambiente
%CO2amb = u(nz+2);
%CO2 generado por personas
%CO2gen = u(nz+3:nz*2+2);
%Humedad ambiente
%Humamb = u(nz*2+3);
%Humedad generada por personas
%Humgen = u(nz*2+4:nz*3+3);

%inputs

%flujo masico del aire
ms = u(nz + 2:(1+2*nz));
%Enfriamiento
DTc = u(2*nz +2);
%Calefaccion
DTH = u(2*nz+3:3*nz+2);
%delta - valve
delta = u(3*nz+3);

%current states
T = x(1:2*nz);
%intermediate variable 
if sum(ms) == 0
    Tr = 0;
else
    aux = (ms.*T(1:nz));
    Tr=(sum(aux))/sum(ms);
end

%Supplies
for i = 1:nz
    Ts(i,1) = delta*Tr+(1-delta)*Toa+DTc+DTH(i);
end

for i=1:nz
    dtn = 0;
    for j = 1:nz
       if Neigh(i,j)==1
           dtn=dtn+(T(j)-T(i))/Rij(i,j);
       end        
    end
    %Air temperature
    dT(i,1) = T(i) + (ST*ms(i)*cp*(Ts(i,1)-T(i))+ST*(T(nz+i)-T(i))/R(i)+...
        ST*(Toa-T(i))/Roa(i)+ST*Pd(i)+ST*dtn)/C1(i);
    %Solids temperature
    dT(i+nz,1)=T(i+nz)+ ST*(T(i)-T(i+nz))/(R(i)*C2(i));

  

    
end

dx=dT;
end


