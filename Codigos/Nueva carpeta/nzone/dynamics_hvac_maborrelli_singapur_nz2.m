function [dx,Ts]=dynamics_hvac_maborrelli_singapur_nz(x,u,params,ST)

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
CO2amb = u(nz+2);
%CO2 generado por personas
CO2gen = u(nz+3:nz*2+2);
%Humedad ambiente
Humamb = u(nz*2+3);
%Humedad generada por personas
Humgen = u(nz*2+4:nz*3+3);

%inputs

%flujo masico del aire
ms = u(nz*3+4);
%delta flujo
delta_flujo = u(nz*3+5:(4+4*nz));
%Enfriamiento
DTc = u(4*nz+5);
%Calefaccion
DTH = u(4*nz+6:5*nz+5);
%delta - valve
delta = u(5*nz+6);

%current states
T = x(1:2*nz);
CO2 = x(2*nz+1:3*nz);
hum = x(3*nz+1:4*nz);
%intermediate variable 
Pa = 1013.25;

if sum(delta_flujo) == 0
    ms = zeros(4,1);
else
    ms = ms.*delta_flujo/sum(delta_flujo);
end

if sum(ms) == 0
    Tr = 0;
    TrCO2 = 0;
    Trhum = 0;
else
    aux = (ms.*T(1:nz));
    Tr=(sum(aux))/sum(ms);
    aux = (ms.*CO2);
    TrCO2=(sum(aux))/sum(ms);
    aux = (ms.*hum);
    Trhum=(sum(aux))/sum(ms);
end

%Supplies
Ts1 = delta*Tr+(1-delta)*Toa+DTc;
CO2s=delta*TrCO2+(1-delta)*CO2amb;
ew = 6.1121*(1.0007+3.46e-6*Pa).*exp((17.502*Ts1)./(240.97+Ts1));
Hsat_Tchilled  = 0.62197*(ew./(Pa-0.378*ew));
Hums = min( delta*Trhum + (1-delta)*Humamb, Hsat_Tchilled);
for i = 1:nz
    Ts(i,1) = delta*Tr+(1-delta)*Toa+DTc+DTH(i);
end

%based on this code
% http://woodshole.er.usgs.gov/operations/sea-mat/air_sea-html/qsat.html
%Hums = min( delta*T(4) + (1-delta)*Humamb, Hsat_Tchilled);


%state dynamics
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

    %CO2
    CO2dif = CO2(i) + ST/(Vol(i)*rhoair)*ms(i)*(CO2s - CO2(i));
    if CO2dif < CO2s
        dCO2(i,1) = CO2s + ((ST*CO2gen(i))/(Vol(i)*Vol(i)*rhoair));
    else
        dCO2(i,1) = CO2(i) + ST/(Vol(i)*rhoair)* (ms(i)*(CO2s - CO2(i)) + CO2gen(i)/Vol(i));
    end
    
    %Humidity
    %Sat hum in room
    ew2 = 6.1121*(1.0007+3.46e-6*Pa).*exp((17.502*dT(i,1))./(240.97+dT(i,1))); % in mb
    Hsat_room  = 0.62197*(ew2./(Pa-0.378*ew2));
    hum_min = min(hum(i),Hums);
    %Hum dynamics
    dHum(i,1) = hum(i) + (ST*ms(i)*(Hums-hum(i))+ Humgen(i)/(rhoair*Vol(i)))/(rhoair*Vol(i));
    if hum(i)+ ST*ms(i)*(Hums-hum(i))/(rhoair*Vol(i)) <= hum_min
        dHum(i,1) = hum_min + Humgen(i)/(rhoair*Vol(i));
    end
    if dHum(i,1) > Hsat_room
        dHum(i,1) = Hsat_room;
    end
    %Relative Hum
    dHum(i+nz,1) =  dHum(i,1)/Hsat_room;

    
end

dx=[dT;dCO2;dHum];
end


