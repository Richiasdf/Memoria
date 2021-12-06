%& Clearing the workspace
clear all
% Set to true, to resume training from a saved agent
resumeTraining = false;
save_agent = true;
use_parallel = true;

%device for critic & actor
device = "gpu";
%Save & Load options
save_agent_name = "train_agent_td3.mat";
load_agent_name = "train_agent_td3_tch2.mat";
%folder where to load/save agents and models.
subcarpeta = "1v/";
%
numObservations = 4;
numActions = 1;
limit_act_low = 0;
limit_act_h = 5;
obs_low = -40;
obs_high = 1e7;
scale = limit_act_low + limit_act_h;
%Range of temperature wanted
low = 22; %temperatura minima
high = 25; % temperatura máxima
CO2_max = 1200*1e-6;
beta = 0.15;% Peso de la energía en el reward
%Sample Time & Simulation Time
Ts = 60*1; %2 min - HVAC System sample Time
Ts2 = 60*10; %n min - Neural network Sample time
Tf = 3600*6; % n horas  - Simulation Time
maxepisodes = 4000;% max number of episodes to stop learning
StopReward = 20; %Episode reward to stop learning
maxsteps = ceil(Tf/Ts2); % Cantidad de pasos en un episodio
%RL Layers
criticlayers = [2,1];
criticNeurons = [128, 128, 128];
actorlayers = 2;
actorNeurons = [128, 128];



%% Load and open the system in simulink
mdl = subcarpeta + "hvac_1zone_v3";
%open_system(mdl)

%% Define model parameters
load('values1z.mat');
params = params1;


%% Reinforcement learning
%Se definen las observaciones, rlNumericSpec es para observaciones de
%rangos continuos2
[obsInfo,actInfo,env] = makeenv(mdl,numObservations,numActions,limit_act_low,limit_act_h,obs_low,obs_high);

%Función para reiniciar el sistema
%Esta función es útil cuando se quiere usar distintas condiciones iniciales
%en cada inicio de un episodio.
%ci = [28 , 34 , 1500];
env.ResetFcn = @(in) localResetFcn(in);
%Set random seed.
rng(0)
%Opciones del agente
agentOpts = rlTD3AgentOptions(...
    'SampleTime',Ts2,...
    'TargetSmoothFactor',1e-3,...
    'DiscountFactor',0.95, ...
    'MiniBatchSize',256, ...
    'SaveExperienceBufferWithAgent',true, ...
    'ExperienceBufferLength',1e6); 
    
%Opciones del ruido aplicado a las acciones tomadas, si se tiene más de una
%acción, y estas no se encuentran en rangos similares, se recomienda
%utilizar un vector en vez de un escalar.
%agentOpts.TargetPolicySmoothModel.StandardDeviationMin = [0.000001; 0.00001] ;
%agentOpts.TargetPolicySmoothModel.StandardDeviation = [0.001; 0.03]; % target policy noise
%agentOpts.TargetPolicySmoothModel.StandardDeviationDecayRate = [1e-3; 1e-3];

agentOpts.ExplorationModel = rl.option.OrnsteinUhlenbeckActionNoise;
agentOpts.ExplorationModel.MeanAttractionConstant = 0.2/Ts2;
agentOpts.ExplorationModel.StandardDeviation = 0.5/sqrt(Ts2); %;0.2*sqrt(Ts2)]/sqrt(Ts2);
agentOpts.ExplorationModel.StandardDeviationDecayRate = 5e-4;% 5e-5];
agentOpts.ExplorationModel.StandardDeviationMin = 0.015/sqrt(Ts2);% 0.001]/sqrt(Ts2);


% Train the agent
% Set ResetExperienceBufferBeforeTraining to false to keep experience from the previous session
agentOpts.ResetExperienceBufferBeforeTraining = ~(resumeTraining);

if resumeTraining
    load(subcarpeta + load_agent_name);
    agent.AgentOptions.TargetPolicySmoothModel.StandardDeviationDecayRate = 1e-5;
    agent.AgentOptions.ExplorationModel.StandardDeviation = [0.5; 1; 0.7 ; 0.1 ]/sqrt(Ts2);
    agent.AgentOptions.ExplorationModel.StandardDeviationMin = [0.005; 0.007; 0.007 ; 0.05 ]/sqrt(Ts2);
    agent.AgentOptions.ExplorationModel.StandardDeviationDecayRate = 1e-5;

else
    %Red neuronal Critico
    criticOpts = rlRepresentationOptions('LearnRate',1e-03,'GradientThreshold',.5, 'UseDevice', device);
    criticNetwork = MakeCritic(numObservations,numActions,criticlayers,criticNeurons);
    critic = rlQValueRepresentation(criticNetwork,obsInfo,actInfo,'Observation',{'State'},'Action',{'Action'},criticOpts);
    criticNetwork2 = MakeCritic(numObservations,numActions,criticlayers,criticNeurons);
    critic2 = rlQValueRepresentation(criticNetwork2,obsInfo,actInfo,'Observation',{'State'},'Action',{'Action'},criticOpts);
    %Red neuronal Actor
    actorNetwork = MakeActor(numObservations,numActions, actorlayers , actorNeurons, scale);
    actorOptions = rlRepresentationOptions('LearnRate',2e-03,'GradientThreshold',.5, 'UseDevice', device);
    actor = rlDeterministicActorRepresentation(actorNetwork,obsInfo,actInfo,'Observation',{'State'},'Action',{'Action'},actorOptions);
    % Create a fresh new agent
    %createNetworks
    agent = rlTD3Agent(actor, [critic,critic2], agentOpts);
end


%    'Verbose',true, ...      'Plots','none',...
trainOpts = rlTrainingOptions(...
    'MaxEpisodes',maxepisodes, ...
    'MaxStepsPerEpisode',maxsteps, ...
    'UseParallel',use_parallel, ...
    'ScoreAveragingWindowLength',50, ...
    'Verbose',true, ...
    'StopTrainingCriteria','EpisodeReward',...
    'StopTrainingValue',StopReward);


simOpts = rlSimulationOptions('MaxSteps',maxsteps,'StopOnError','on');
if use_parallel
    parpool;
    parfevalOnAll( @warning, 0, 'off', 'all');
    clearvars use_parallel maxepisodes maxsteps StopReward resumeTraining device
else 
    warning('off','all')
    clearvars use_parallel maxepisodes maxsteps StopReward resumeTraining device
end

%Se comienza a entrenar el agente y se abre el gráfico de entrenamiento.
trainingStats = train(agent,env,trainOpts);

%Se carga un agente pre entrenado:
%load('train_agent2.mat')

%Se simula el agente entrenado

sim(env,agent,simOpts);

%Save the train network
%save('1stTD3_agent.mat','agent','simOpts')
if save_agent
    save(subcarpeta+save_agent_name,'agent','simOpts')
end

%% Funciones a utilizar

function criticNetwork = MakeCritic(obs,acts,layers,neurons)
    statePath = featureInputLayer(obs,'Normalization','none','Name','State');
    for i=1:layers(1)
        if mod(i,2) == 0
            statePath = [statePath; reluLayer('Name','StateTanh'+string(i/2))];
        end
        statePath = [statePath; fullyConnectedLayer(neurons(i),'Name','CriticStateFC'+string(i))];
        
    end
    actionPath = featureInputLayer(acts,'Normalization','none','Name','Action');
    for i=1:layers(2)
        if mod(i,2) == 0
            actionPath = [actionPath;reluLayer('Name','ActionTanh'+string(i/2))];
        end
        actionPath = [actionPath; fullyConnectedLayer(neurons(i+layers(1)),'Name','CriticActionFC'+string(i))]; 
    end
    commonPath = [
        additionLayer(2,'Name','add')
        reluLayer('Name','CommonTanh')
        fullyConnectedLayer(1,'Name','CriticOutput')];

    criticNetwork = layerGraph();
    criticNetwork = addLayers(criticNetwork,statePath);
    criticNetwork = addLayers(criticNetwork,actionPath);
    criticNetwork = addLayers(criticNetwork,commonPath);
    criticNetwork = connectLayers(criticNetwork,'CriticStateFC'+string(layers(1)),'add/in1');
    criticNetwork = connectLayers(criticNetwork,'CriticActionFC'+string(layers(2)),'add/in2');
    
end

function Network = MakeActor(obs,acts,layers,neurons,scale)
    Network = featureInputLayer(obs,'Normalization','none','Name','State');
    for i=1:layers(1)
        Network = [Network; fullyConnectedLayer(neurons(i),'Name','ActorFC'+string(i)); reluLayer('Name','ActorTanh'+string(i/2))];
    end
    Network = [Network  ;fullyConnectedLayer(acts,'Name','Action2');sigmoidLayer('Name','Sigmoid') ;scalingLayer('Name','Action','Scale',scale)];
end