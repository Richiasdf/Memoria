%% Load and open the system in simulink
mdl2 = 'hvac_maborrelli_singapur_nzones_with_MPC_v3_manualdist_3'
open_system(mdl2)

%% Define model
%for 1 zone
params1.cp=1012;
params1.C1=9.163*10^6; %9.163*10^6
params1.C2=1.694*10^8; %1.694*10^8
params1.R=1.7*10^(-3); % 1.7*10^(-3)
params1.Roa=57*10^(-3); % 57*10^(-3)

params1.kf=65;
params1.eta=4;

params1.mode_map=[1 8 20;...
                  2 0 8;...
                  2 20 24];
params1.t_offset=0; %in hours!!!!

%for nz zones
ci = [30;30;30;30;30;30;30;30];
nz=4;
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
paramsnz.mode_map=[1 8 20;...
                  2 0 8;...
                  2 20 24];
paramsnz.t_offset=0; %in hours!!!!
paramsnz.Vol=3*4*5*ones(nz,1);
paramsnz.rhoair=1.225;



% Extras? 
cp=params1.cp;
C1=params1.C1;
C2=params1.C2;
R=params1.R;
Roa=params1.Roa;


%% disturnance definitions
nz=4;
tt=clock; jojo=tt(end); for i=1:floor(jojo*100),randn;end
sampling_time=60; %60 sec
simulation_days=365; 

nzones=nz;

sig_Pd=600;
sig_T=2;
sig_w=200;


%Temperatures
npattern=4;
temp_pattern=[38,37,39,40];
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


%% Reinforcement learning
obsInfo = rlNumericSpec([4 1], 'LowerLimit', -20, 'UpperLimit', 40);
obsInfo.Name = 'observations';
obsInfo.Description = 'estado 1, estado 2, estado 3 and estado 4';
numObservations = obsInfo.Dimension(1);

actInfo = rlNumericSpec([6 1], 'LowerLimit', [0.01; 0.01; 0.01; 0.01; -15; 0], 'UpperLimit', [5; 5; 5; 5; 0; 1]);
actInfo.Name = 'actions';
numActions = actInfo.Dimension(1);

env = rlSimulinkEnv(mdl2,'hvac_maborrelli_singapur_nzones_with_MPC_v3_manualdist_3/RL Agent',...
    obsInfo,actInfo);

%in = Simulink.SimulationInput(mdl2);
%disp(in)
%Función para reiniciar el sistema
%env.ResetFcn = @(in) localResetFcn(in);

%Saturation, to keep real temps
min_temp = -10;
max_temp = 45;

%Sample Time & Simulation Time
Ts2 = 15;
Tf = 15200;

rng(0)

%Red neuronal Critico
%CAMBIAR

statePath = [
    featureInputLayer(numObservations,'Normalization','none','Name','State')
    fullyConnectedLayer(24,'Name','CriticStateFC1')
    reluLayer('Name','CriticRelu1')
    fullyConnectedLayer(12,'Name','CriticStateFC2')];
actionPath = [
    featureInputLayer(numActions,'Normalization','none','Name','Action')
    fullyConnectedLayer(12,'Name','CriticActionFC1')];
commonPath = [
    additionLayer(2,'Name','add')
    reluLayer('Name','CriticCommonRelu')
    fullyConnectedLayer(1,'Name','CriticOutput')];

criticNetwork = layerGraph();
criticNetwork = addLayers(criticNetwork,statePath);
criticNetwork = addLayers(criticNetwork,actionPath);
criticNetwork = addLayers(criticNetwork,commonPath);
criticNetwork = connectLayers(criticNetwork,'CriticStateFC2','add/in1');
criticNetwork = connectLayers(criticNetwork,'CriticActionFC1','add/in2');

criticOpts = rlRepresentationOptions('LearnRate',1e-02,'GradientThreshold',1);
critic = rlQValueRepresentation(criticNetwork,obsInfo,actInfo,'Observation',{'State'},'Action',{'Action'},criticOpts);


actorNetwork = [
    featureInputLayer(numObservations,'Normalization','none','Name','State')
    fullyConnectedLayer(12, 'Name','actorFC')
    reluLayer('Name','CriticCommonRelu')
    fullyConnectedLayer(numActions,'Name','Action')
    ];

actorOptions = rlRepresentationOptions('LearnRate',1e-02,'GradientThreshold',1);


%Ver mas en detalle
actor = rlDeterministicActorRepresentation(actorNetwork,obsInfo,actInfo,'Observation',{'State'},'Action',{'Action'},actorOptions);


agentOpts = rlDDPGAgentOptions(...
    'SampleTime',Ts2,...
    'TargetSmoothFactor',1e-3,...
    'DiscountFactor',1.0, ...
    'MiniBatchSize',64, ...
    'ExperienceBufferLength',1e6); 
    
agentOpts.NoiseOptions.StandardDeviation = 0.3;
agentOpts.NoiseOptions.StandardDeviationDecayRate = 1e-5;
% Averiguar***
agentOpts.NoiseOptions.MeanAttractionConstant = 0.2/Ts2;

agent = rlDDPGAgent(actor,critic,agentOpts);


maxepisodes = 5000;
maxsteps = ceil(Tf/Ts2);
StopReward = (maxsteps*0.85)*60;
trainOpts = rlTrainingOptions(...
    'MaxEpisodes',maxepisodes, ...
    'MaxStepsPerEpisode',maxsteps, ...
    'UseParallel',true, ...
    'ScoreAveragingWindowLength',20, ...
    'Verbose',false, ...
    'Plots','training-progress',...
    'StopTrainingCriteria','AverageReward',...
    'StopTrainingValue',StopReward);

trainingStats = train(agent,env,trainOpts);


simOpts = rlSimulationOptions('MaxSteps',maxsteps,'StopOnError','on');
experiences = sim(env,agent,simOpts);

%Save the train network
%save('train_agent.mat','agent','experiences')

