function [dT,Ts]=dynamics_hvac_maborrelli_singapur_nz(x,u,params)

% nz=par.nz;
% Vert=par.Vert; 
% Neigh=par.Neigh;
% cp=par.cp;
% C1=par.C1;
% C2=par.C2;
% R=par.R;
% Rij=par.Rij;
% Roa=par.Roa;

nz=params.nz; %nzones
Vert=params.Vert;
Neigh=params.Neigh;
cp=params.cp;
C1=params.C1;
C2=params.C2;
R=params.R;
Rij=params.Rij;
Roa=params.Roa;


%disturbances
%temp ambiente
Toa=u(1);
%calor generado por objetos interiores y personas
Pd=u(2:nz+1);

%inputs
%flujo masico del aire
%limites? 0 a 5
ms=u(2+nz:(1+2*nz));
%Lo posible sería agregar la parte de calentamiento
%limites?
DTc=u(2*nz+2);
%0 a 1
delta=u(2*nz+3);

%current states
T=x;
%intermediate variable 

Tr=Vert*(ms.*T(1:nz))/(Vert*ms);
for i=1:nz
    Ts(i,1)=delta*Tr+(1-delta)*Toa+DTc;
end


% output(1)=Tr;
%state dynamics
for i=1:nz
    dtn=0;
    for j=1:nz
       if Neigh(i,j)==1
           dtn=dtn+(T(j)-T(i))/Rij(i,j);
       end        
    end
    
    dT(i,1)=(ms(i)*cp*(Ts(i)-T(i))+(T(nz+i)-T(i))/R(i)+...
        (Toa-T(i))/Roa(i)+Pd(i)+dtn)/C1(i);
    
    dT(i+nz,1)=(T(i)-T(i+nz))/R(i)/C2(i);
end

