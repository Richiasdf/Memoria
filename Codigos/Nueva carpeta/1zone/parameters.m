
%%params1zona
params1.cp=1012;
params1.C1=9.163*10^6; %9.163*10^6
params1.C2=1.694*10^8; %1.694*10^8
params1.R=1.7*10^(-3); % 1.7*10^(-3)
params1.Roa=57*10^(-3); % 57*10^(-3)

params1.kf=65;
params1.eta=4;
params1.eta_h = 0.7;
%params1.mode_map=[1 8 21;...
%                  2 0 8;...
%                  2 21 24]; 
%params1.t_offset=0; %in hours!!!!
params1.Vol=3*4*5;
params1.rhoair=1.225;

%%params n zonas

nz=4;
paramsnz.nxroom=4;
paramsnz.nz=nz;
paramsnz.Vert=[1:4];
paramsnz.Neigh=[0,1,0,0;...
       1,0,1,0;...
       0,1,0,1;...
       0,0,1,0];
paramsnz.cp=1012;
paramsnz.C1=9.163*10^6*ones(nz,1);
paramsnz.C2=1.694*10^8*ones(nz,1);
paramsnz.R=1.7*10^(-3)*ones(nz,1);
paramsnz.Rij=100*10^(-3)*ones(nz,nz).*paramsnz.Neigh;
paramsnz.Roa=57*10^(-3)*ones(nz,1);
paramsnz.kf=65;
paramsnz.eta=4*ones(nz,1);
paramsnz.mode_map=[1 8 21;...
                  2 0 8;...
                  2 21 24]; 
paramsnz.t_offset=0; %in hours!!!!
paramsnz.Vol=3*4*5*ones(nz,1);
paramsnz.rhoair=1.225;


tt=clock; jojo=tt(end); for i=1:floor(jojo*100),randn;end
sampling_time=60; %60 sec

nzones=1;
simulation_days=8; 

%% Temperatures
sig_Pd=300; %600
sig_T=1.5; %2
sig_w=200;%20


%Ambient Temp
npattern=4;
temp_pattern=[2,8,22,34];
time_pattern=[0,6,12,18]*3600;



temps=repmat(temp_pattern,1,simulation_days);
temps=temps+1.3*randn(1,npattern*simulation_days);
temps_w=temps+sig_T*randn(1,npattern*simulation_days);
times=[0:npattern*simulation_days-1]*3600*24/npattern;

sample_times=[0:3600/sampling_time*24*(simulation_days-1)]*sampling_time; %notice that due to spline interpolation the last 24/npattern hours do not work well, thus I just omit last due for simplicity
yy=spline(times,temps_w,sample_times);

Ts_struct.signals.dimensions=1;
Ts_struct.time=sample_times';
Ts_struct.signals.values=yy';


%Internal gains
npattern=24*7;
simulation_weeks=ceil(simulation_days/7);
pds_pattern=[0,0,0,0,0,0,0,0,1,1.1,1.5,1.9,1.96,2,2,2,2,1.96,1.9,1.5,1.1,0,0,0,...
             0,0,0,0,0,0,0,0,1,1.1,1.5,1.9,1.96,2,2,2,2,1.96,1.9,1.5,1.1,0,0,0,...
             0,0,0,0,0,0,0,0,1,1.1,1.5,1.9,1.96,2,2,2,2,1.96,1.9,1.5,1.1,0,0,0,...
             0,0,0,0,0,0,0,0,1,1.1,1.5,1.9,1.96,2,2,2,2,1.96,1.9,1.5,1.1,0,0,0,...
             0,0,0,0,0,0,0,0,1,1.1,1.5,1.9,1.96,2,2,2,2,1.96,1.9,1.5,1.1,0,0,0,...
             0,0,0,0,0,0,0,0,0.7,0.9,1.1,1.2,1.0,0.8,0.7,0.5,0.2,0,0,0,0,0,0,0,...
             0,0,0,0,0,0,0,0,0.7,0.9,1.1,1.2,1.0,0.8,0.7,0.5,0.2,0,0,0,0,0,0,0]*1000*2;
Pds=repmat(pds_pattern,nzones,simulation_weeks);
Pds=max(Pds+100*randn(nzones,npattern*simulation_weeks) ,0);
Pds_wprev=max(Pds+sig_Pd*randn(nzones,npattern*simulation_weeks) ,0);
Pds_w=[];
for i=1:nzones
    Pds_w=[Pds_w;max(interp(Pds_wprev(i,:),3600/sampling_time),0)];
end


Pds_struct.signals.dimensions=nzones;
Pds_struct.time=[0:3600/sampling_time*npattern*simulation_weeks-1]'*sampling_time;
Pds_struct.signals.values=Pds_w';

%Unmodelled uncertainties
npattern=4;
w_seq=sig_w*randn(nzones,npattern*simulation_days);
times=[0:npattern*simulation_days-1]*3600*24/npattern;
sample_times=[0:3600/sampling_time*24*(simulation_days-1)]*sampling_time; %notice that due to spline interpolation the last 24/npattern hours do not work well, thus I just omit last due for simplicity
yy=spline(times,w_seq,sample_times);

w_struct.signals.dimensions=nzones;
w_struct.time=sample_times';
w_struct.signals.values=yy';

%% CO2 ... basado en CO2genperson=0.1m^h=2.777*10^-5 m^3/s=27.77 *10^-6 m^3/s
%% Where the hell did I get the above from?????? CO2genperson=0.31Lt/min=5.166*10^-6 m3/s (ASHRAE)
sig_CO2=2/10^6;%20*10^6
sig_wCO2=0.1/10^6;%1*10^6

% Ambient CO2
npattern=6;
CO2_pattern=[350,350,450,400,420,420]*10^-6;
time_pattern=[0,4,8,12,16,20]*3600;



CO2s=repmat(CO2_pattern,1,simulation_days);
CO2s_w=CO2s+sig_CO2*randn(1,npattern*simulation_days);
times=[0:npattern*simulation_days-1]*3600*24/npattern;

sample_times=[0:3600/sampling_time*24*(simulation_days-1)]*sampling_time; %notice that due to spline interpolation the last 24/npattern hours do not work well, thus I just omit last due for simplicity
yy=spline(times,CO2s_w,sample_times);

CO2s_struct.signals.dimensions=1;
CO2s_struct.time=sample_times';
CO2s_struct.signals.values=yy';


%Internal gains
CO2genperson=5.166*10^-6;
sig_CO2gen=0.025/10^6;%0.25/10^6
npattern=24*7;
simulation_weeks=ceil(simulation_days/7);
CO2gens_pattern=[0,0,0,0,0,0,0,0,1,1.1,1.5,1.9,1.96,2,2,2,2,1.96,1.9,1.5,1.1,0,0,0,...
             0,0,0,0,0,0,0,0,1,1.1,1.5,1.9,1.96,2,2,2,2,1.96,1.9,1.5,1.1,0,0,0,...
             0,0,0,0,0,0,0,0,1,1.1,1.5,1.9,1.96,2,2,2,2,1.96,1.9,1.5,1.1,0,0,0,...
             0,0,0,0,0,0,0,0,1,1.1,1.5,1.9,1.96,2,2,2,2,1.96,1.9,1.5,1.1,0,0,0,...
             0,0,0,0,0,0,0,0,1,1.1,1.5,1.9,1.96,2,2,2,2,1.96,1.9,1.5,1.1,0,0,0,...
             0,0,0,0,0,0,0,0,0.7,0.9,1.1,1.2,1.0,0.8,0.7,0.5,0.2,0,0,0,0,0,0,0,...
             0,0,0,0,0,0,0,0,0.7,0.9,1.1,1.2,1.0,0.8,0.7,0.5,0.2,0,0,0,0,0,0,0]*CO2genperson*2;
CO2gens=repmat(CO2gens_pattern,nzones,simulation_weeks);
CO2gens_wprev=max(CO2gens+sig_CO2gen*randn(nzones,npattern*simulation_weeks) ,0);
CO2gens_w=[];
for i=1:nzones
    CO2gens_w=[CO2gens_w;max(interp(CO2gens_wprev(i,:),3600/sampling_time),0)];
end


CO2gens_struct.signals.dimensions=nzones;
CO2gens_struct.time=[0:3600/sampling_time*npattern*simulation_weeks-1]'*sampling_time;
CO2gens_struct.signals.values=CO2gens_w';

%Unmodelled uncertainties
npattern=4;
w_seq=sig_wCO2*randn(nzones,npattern*simulation_days);
times=[0:npattern*simulation_days-1]*3600*24/npattern;
sample_times=[0:3600/sampling_time*24*(simulation_days-1)]*sampling_time; %notice that due to spline interpolation the last 24/npattern hours do not work well, thus I just omit last due for simplicity
yy=spline(times,w_seq,sample_times);

CO2w_struct.signals.dimensions=nzones;
CO2w_struct.time=sample_times';
CO2w_struct.signals.values=yy';

Npeoplemax=2;
%% Humidity ... basado en 90% RH & 25 C=>17.68/10^3 kg/kg & 65% RH & 32 C=>19.19/10^3 kg/kg
% sig_H=0.1/10^3;%2/10^3
% sig_wH=0.001/10^3;%0.01/10^3
sig_H=0;
sig_wH=0;
ar=0.8;%0.55
br=0.4;%0.3
%Ambient Humidity
npattern=6;
hum_pattern=[17.5,17.5,18,21,21,19]/10^3;
time_pattern=[0,4,8,12,16,20]*3600;



Hs=repmat(hum_pattern,1,simulation_days);
Hs_w=Hs+sig_H*randn(1,npattern*simulation_days);
times=[0:npattern*simulation_days-1]*3600*24/npattern;

sample_times=[0:3600/sampling_time*24*(simulation_days-1)]*sampling_time; %notice that due to spline interpolation the last 24/npattern hours do not work well, thus I just omit last due for simplicity
yy=spline(times,Hs_w,sample_times);

Hs_struct.signals.dimensions=1;
Hs_struct.time=sample_times';
Hs_struct.signals.values=yy';


%Internal gains
Hgenperson=50/10^3/3600; %50g per hour per person
% sig_Hgen=0/10^3/3600;%20/10^3/3600
sig_Hgen=0;

npattern=24*7;
simulation_weeks=ceil(simulation_days/7);
Hgens_pattern=Npeoplemax/2*[0,0,0,0,0,0,0,0,1,1.1,1.5,1.9,1.96,2,2,2,2,1.96,1.9,1.5,1.1,0,0,0,...
             0,0,0,0,0,0,0,0,1,1.1,1.5,1.9,1.96,2,2,2,2,1.96,1.9,1.5,1.1,0,0,0,...
             0,0,0,0,0,0,0,0,1,1.1,1.5,1.9,1.96,2,2,2,2,1.96,1.9,1.5,1.1,0,0,0,...
             0,0,0,0,0,0,0,0,1,1.1,1.5,1.9,1.96,2,2,2,2,1.96,1.9,1.5,1.1,0,0,0,...
             0,0,0,0,0,0,0,0,1,1.1,1.5,1.9,1.96,2,2,2,2,1.96,1.9,1.5,1.1,0,0,0,...
             0,0,0,0,0,0,0,0,0.7,0.9,1.1,1.2,1.0,0.8,0.7,0.5,0.2,0,0,0,0,0,0,0,...
             0,0,0,0,0,0,0,0,0.7,0.9,1.1,1.2,1.0,0.8,0.7,0.5,0.2,0,0,0,0,0,0,0]*Hgenperson;
% Hgens=repmat(Hgens_pattern,nzones,simulation_weeks);
% Hgens_wprev=max(Hgens+sig_Hgen*randn(nzones,npattern*simulation_weeks) ,0);
% Hgens_w=[];
% for i=1:nzones
%     Hgens_w=[Hgens_w;max(interp(Hgens_wprev(i,:),3600/sampling_time),0)];
% end

Hgens=[];
Hgens_w=[];
for i=1:nzones
    Hgen=repmat(Hgens_pattern*(ar+(i-1)*br),1,simulation_weeks);
    Hgens=[Hgens;Hgen];
    Hgens_wprev=max(Hgen+sig_Hgen*randn(1,npattern*simulation_weeks) ,0);
    Hgens_w=[Hgens_w;max(interp(Hgens_wprev,3600/sampling_time),0)];
end


Hgens_struct.signals.dimensions=nzones;
Hgens_struct.time=[0:3600/sampling_time*npattern*simulation_weeks-1]'*sampling_time;
Hgens_struct.signals.values=Hgens_w';

%Unmodelled uncertainties
npattern=4;
w_seq=sig_wH*randn(nzones,npattern*simulation_days);
times=[0:npattern*simulation_days-1]*3600*24/npattern;
sample_times=[0:3600/sampling_time*24*(simulation_days-1)]*sampling_time; %notice that due to spline interpolation the last 24/npattern hours do not work well, thus I just omit last due for simplicity
yy=spline(times,w_seq,sample_times);

Hw_struct.signals.dimensions=nzones;
Hw_struct.time=sample_times';
Hw_struct.signals.values=yy';

