load_agent_name = "train_agent_td3_tch2.mat";
subcarpeta = "4v/";
restrictions = "TCH";
load(subcarpeta+load_agent_name);
if subcarpeta == "3v/" && restrictions == "T"
    load('values1z.mat');
    mdl = subcarpeta + "hvac_1zone_v3";
    ci = [28 ; 28 ];
    numObservations = 4;
    numActions = 3;
    limit_act_low = [0; -30; 0];
    limit_act_h = [5; 0; .9];
    obs_low = -40;
    obs_high = 1e7;
elseif subcarpeta == "2v/" && restrictions == "T"
    load('values1z.mat');
    mdl = subcarpeta + "hvac_1zone_v3";
    ci = [28 ; 34 ];
    numObservations = 4;
    numActions = 2;
    limit_act_low = [0; -30];
    limit_act_h = [5; 0];
    obs_low = -40;
    obs_high = 1e7;
elseif subcarpeta == "4v/" && restrictions == "TC"
    load('values1z_co2.mat');
    mdl = subcarpeta + "hvac_1zone_v3";
    ci = [4 ; 22 ;  .900*1e-3];
    numObservations = 7;
    numActions = 4;
    limit_act_low = [0; -30; 0 ; 0];
    limit_act_h = [5; 0; 8; .9];
    obs_low = -40;
    obs_high = 1e7;
    %obs_low = -inf;
    %obs_high = inf;
elseif subcarpeta == "4v/" && restrictions == "TCH"
    load('values1z_tch.mat');
    Hs_struct.signals.values = Hs_struct.signals.values/2;
    mdl = subcarpeta + "hvac_1zone_TCH";
    ci = [10 ; 22 ;  .900*1e-3 ;  0.0013 ; 0.5];
    numObservations = 11;
    numActions = 4;
    limit_act_low = [0; -30; 0 ; 0];
    limit_act_h = [5; 0; 8; .9];
    obs_low = -40;
    obs_high = 1e7;
end
    
%Range of temperature wanted
low = 22; %temperatura minima
high = 25; % temperatura máxima
CO2_max = 1200*1e-6;
beta = 0.05;% Peso de la energía en el reward
%Sample Time & Simulation Time
Ts = 60*1; %2 min - HVAC System sample Time
Ts2 = 60*10; %n min - Neural network Sample time
Tf = 3600*6; % n horas  - Simulation Time
maxepisodes = 30000;% max number of episodes to stop learning
maxsteps = ceil(Tf/Ts2); % Cantidad de pasos en un episodio
open_system(mdl)
params = params1;
[obsInfo,actInfo,env] = makeenv(mdl,numObservations,numActions,limit_act_low,limit_act_h,obs_low,obs_high);
rng(0)
warning('off','all')
sim(env,agent,simOpts);