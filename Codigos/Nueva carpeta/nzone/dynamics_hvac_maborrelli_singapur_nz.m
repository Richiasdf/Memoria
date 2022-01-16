function [dx,Ts]=dynamics_hvac_maborrelli_singapur_nz(x,u,params,ST)

% nz=par.nz;
% Vert=par.Vert; 
% Neigh=par.Neigh;
% cp=par.cp;
% C1=par.C1;
% C2=par.C2;
% R=par.R;
% Rij=par.Rij;
% Roa=par.Roa;

nz=params.nz; 
Neigh=params.Neigh;
cp=params.cp;
C1=params.C1;
C2=params.C2;
R=params.R;
Rij=params.Rij;
Roa=params.Roa;

Vol=params.Vol;
rhoair=params.rhoair;


%disturbances
%temp ambiente

Toa=u(1);
%calor generado por objetos interiores y personas
Pd=u(2:nz+1);
%CO2 ambiente
CO2amb = u(nz+2);
%CO2 generado por personas
CO2gen = u(nz+3:nz*2+2);
%Humedad ambiente
Humamb = u(nz*2+3);
%Humedad generada por personas
Humgen = u(nz*2+4:nz*3+3);
%inputs
%flujo masico del aire
%limites? 0 a 5
ms=u(nz*3+4:(3+4*nz));
%Lo posible sería agregar la parte de calentamiento
%limites?
DTc=u(4*nz+4);
%0 a 1
DTH = u(4*nz+5:5*nz+4);
delta=u(5*nz+5);

%current states
T=x(1:2*nz);
CO2 = x(2*nz+1:3*nz);
hum = x(3*nz+1:4*nz);
%intermediate variable 
Pa=1013.25;
if sum(ms) == 0
    Tr = 0;
    TrCO2 = 0;
    Trhum = 0;
else
    Tr=(ms'.*T(1:nz))/sum(ms);
    TrCO2=(ms'.*CO2)/sum(ms);
    Trhum=(ms'.*hum)/sum(ms);
end


for i=1:nz
    if Tr == 0
        Ts1(i,1)=delta*Tr(1)+(1-delta)*Toa+DTc;
        Ts(i,1)=delta*Tr(1)+(1-delta)*Toa+DTc+DTH(i);
        CO2s(i,1)=delta*TrCO2(1)+(1-delta)*CO2amb;
        ew = 6.1121*(1.0007+3.46e-6*Pa).*exp((17.502*Ts1(i))./(240.97+Ts1(i)));
        Hsat_Tchilled  = 0.62197*(ew./(Pa-0.378*ew));
        Hums(i,1) = min( delta*Trhum(1) + (1-delta)*Humamb, Hsat_Tchilled);
    else
        Ts1(i,1)=delta*Tr(1,i)+(1-delta)*Toa+DTc;
        Ts(i,1)=delta*Tr(1,i)+(1-delta)*Toa+DTc+DTH(i);
        CO2s(i,1)=delta*TrCO2(1,i)+(1-delta)*CO2amb;
        ew = 6.1121*(1.0007+3.46e-6*Pa).*exp((17.502*Ts1(i))./(240.97+Ts1(i)));
        Hsat_Tchilled  = 0.62197*(ew./(Pa-0.378*ew));
        Hums(i,1) = min( delta*Trhum(1,i) + (1-delta)*Humamb, Hsat_Tchilled);
    end
end

%based on this code
% http://woodshole.er.usgs.gov/operations/sea-mat/air_sea-html/qsat.html
%Hums = min( delta*T(4) + (1-delta)*Humamb, Hsat_Tchilled);


% output(1)=Tr;
%state dynamics

for i=1:nz
    dtn=0;
    for j=1:nz
       if Neigh(i,j)==1
           dtn=dtn+(T(j)-T(i))/Rij(i,j);
       end        
    end
    
    dT(i,1)=T(i) + (ST*ms(i)*cp*(Ts(i,1)-T(i))+ST*(T(nz+i)-T(i))/R(i)+...
        ST*(Toa-T(i))/Roa(i)+ST*Pd(i)+ST*dtn)/C1(i);
    
    % Energy
    %E(i,1) = ms(i)*cp*(Ts(i)-T(i))/nc(?) + kf*ms(i)**2
    dT(i+nz,1)=T(i+nz)+ ST*(T(i)-T(i+nz))/(R(i)*C2(i));

    
    CO2dif = CO2(i) + ST/(Vol(i)*rhoair)*ms(i)*(CO2s(i,1) - CO2(i));
    if CO2dif < CO2s(i,1)
        dCO2(i,1) = CO2s(i,1) + ((ST*CO2gen(i))/Vol(i)*rhoair);
    else
        dCO2(i,1) = CO2(i) + ST/(Vol(i)*rhoair)* (ms(i)*(CO2s(i,1) - CO2(i)) + CO2gen(i));
    end
    
    %%%%%%%%% 
    dCO2(i,1)=ms(i)*(CO2s(i,1)-CO2(i))/(rhoair*Vol(i))+CO2gen(i)/Vol(i);
    if CO2(i)<=0 && dCO2(i,1)<=0
        dCO2(i,1)=0;
    end
    
    dHum(i,1)=(ms(i)*(Hums(i,1)-hum(i))+Humgen(i))/(rhoair*Vol(i));
    if hum(i)<=0 && dHum(i,1)<=0
        dHum(i,1)=0;
    end
    
    ew2 = 6.1121*(1.0007+3.46e-6*Pa).*exp((17.502*dT(i))./(240.97+dT(i))); % in mb
    Hsat_room  = 0.62197*(ew2./(Pa-0.378*ew2));

    hum_min = min(hum(i),Hums(i,1));
    dHum(i,1) = hum(i) + (ST*ms(i)*(Hums(i,1)-hum(i))+ Humgen(i)/(rhoair*Vol(i)))/(rhoair*Vol(i));
    if hum(i)+ ST*ms(i)*(Hums(i,1)-hum(i))/(rhoair*Vol(i)) < hum_min
        dHum(i,1) = hum_min + Humgen(i)/(rhoair*Vol(i));
    end
    if dHum(i,1) > Hsat_room
        dHum(i,1) = Hsat_room;
    end

    dHum(i+nz,1) =  dHum(i,1)/Hsat_room;

    
end

dx=[dT;dCO2;dHum];
end


