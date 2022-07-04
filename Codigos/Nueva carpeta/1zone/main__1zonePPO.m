%& Clearing the workspace
clear all
% Set to true, to resume training from a saved agent
resumeTraining = false;
save_agent = true;
use_parallel = 0;

%device for critic & actor
device = "cpu";
%Save & Load options
save_agent_name = "train_agent_ppo_tc.mat";
load_agent_name = "train_agent2.mat";
%folder where to load/save agents and models.
subcarpeta = "4v/";
%
numObservations = 7;
numActions = 4;
obs_low = -40;
obs_high = 1e7;
limit_act_low = [0; -30 ;0 ; 0];
limit_act_h = [5; 0 ;8 ;0.9];
scale = limit_act_low + limit_act_h;
limit_act_low = [0; 0; 0; 0];
limit_act_h = [1; 1; 1; 1];
%Range of temperature wanted
low = 22; %temperatura minima
high = 25; % temperatura máxima
CO2_max = 1200*1e-6;
beta = 0.1;% Peso de la energía en el reward
%Sample Time & Simulation Time
Ts = 60*1; %2 min - HVAC System sample Time
Ts2 = 60*10; %n min - Neural network Sample time
Tf = 3600*48; % n horas  - Simulation Time
maxepisodes = 3000;% max number of episodes to stop learning
StopReward = -200; %Episode reward to stop learning
maxsteps = ceil(Tf/Ts2); % Cantidad de pasos en un episodio
%RL Layers
criticlayers = 2;
criticNeurons = [128, 128];
actorlayers = 2;
actorNeurons = [32, 32];



%% Load and open the system in simulink
%blk = "hvac_1zone_TCH_stocastic";
blk = "hvac_1zone_v3_stocastic";
mdl = subcarpeta + blk;
%open_system(mdl)

%% Define model parameters
%load('values1z.mat'); % Uncomennt this por T params
%load('values1z_co2.mat'); % Uncomment this for TC params
load('values1z_tch.mat'); % uncomment this for TCH (temp, co2, hum) params
params = params1;
%Hs_struct.signals.values = Hs_struct.signals.values/2;


%% Reinforcement learning
%Se definen las observaciones, rlNumericSpec es para observaciones de
%rangos continuos2
[obsInfo,actInfo,env] = makeenv(mdl,numObservations,numActions,limit_act_low,limit_act_h,obs_low,obs_high);

%Función para reiniciar el sistema
%Esta función es útil cuando se quiere usar distintas condiciones iniciales
%en cada inicio de un episodio.

env.ResetFcn = @(in) localResetFcn2(in, blk, T_s,H_sup,Hs_struct);

%Set random seed.
rng(0)
%Opciones del agente
agentOpts = rlPPOAgentOptions(...
    'SampleTime',Ts2,...
    'ClipFactor',0.2,...
    'DiscountFactor',0.99, ...
    'MiniBatchSize',256,...
    'ExperienceHorizon',280,... 
    'NumEpoch',10,...
    'GAEFactor',0.95,...
    'AdvantageEstimateMethod','gae'); 

if resumeTraining
    load(subcarpeta+load_agent_name);
else
    %Red neuronal Critico
    criticOpts = rlRepresentationOptions('LearnRate',3e-03,'GradientThreshold',1, 'UseDevice', device);
    criticNetwork = MakeCritic(numObservations,numActions,criticlayers,criticNeurons);
    critic = rlValueRepresentation(criticNetwork,obsInfo,'Observation',{'State'},criticOpts);
    %Red neuronal Actor
    actorNetwork = MakeActor(numObservations,numActions, actorlayers , actorNeurons,scale);
    actorOptions = rlRepresentationOptions('LearnRate',1e-03,'GradientThreshold',1, 'UseDevice', device);
    actor = rlStochasticActorRepresentation(actorNetwork,obsInfo,actInfo,'Observation',{'State'},actorOptions);
    % Create a fresh new agent
    %createNetworks
    agent = rlPPOAgent(actor, critic, agentOpts);
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
    %StepsUntilDataIsSent
    parpool;
    parfevalOnAll( @warning, 0, 'off', 'all');
    clearvars use_parallel maxepisodes maxsteps StopReward resumeTraining device
else 
    warning('off','all')
    clearvars use_parallel maxepisodes maxsteps StopReward resumeTraining device
end

%Se comienza a entrenar el agente y se abre el gráfico de entrenamiento.
%dbstop if naninf
%trainingStats = train(agent,env,trainOpts);

%Se carga un agente pre entrenado:
%load(subcarpeta+load_agent_name);
%load_agent_name = "train_agent_td3_2.mat";
%load_agent_name = "train_agent2.mat";


%Se simula el agente entrenado

%sim(env,agent,simOpts);

%Save the train network
if save_agent
    save(subcarpeta+save_agent_name,'agent','simOpts','env')
end


%% Funciones a utilizar

function statePath = MakeCritic(obs,acts,layers,neurons)
    statePath = featureInputLayer(obs,'Normalization','none','Name','State');
    for i=1:layers(1)
        if mod(i,2) == 0
            statePath = [statePath; reluLayer('Name','State ReLu '+string(i/2))];
        end
        statePath = [statePath; fullyConnectedLayer(neurons(i),'Name','CriticStateFC'+string(i))];
        
    end
    statePath = [statePath; reluLayer('Name','Common ReLu'); fullyConnectedLayer(1,'Name','CriticOutput')];
end

function Network0 = MakeActor(obs,acts,layers,neurons,scale)
    in = [featureInputLayer(obs,'Normalization','none','Name','State'); fullyConnectedLayer(acts,'Name','infc')]; 
    Network0 = layerGraph(in);
    outLayer = concatenationLayer(1,acts,'Name','out');
    Network0 = addLayers(Network0, outLayer);
    for j=1:acts
        Network{j} =[];
        for i=1:layers(1)
            Network{j} = [Network{j}; fullyConnectedLayer(neurons(i),'Name','ActorFC'+string(j)+ ' - ' + string(i)); reluLayer('Name','Actor ReLu '+string(j)+ ' - '+string(i))];
        end
        Network{j} = [Network{j} ;fullyConnectedLayer(2,'Name','Action '+string(j)); softmaxLayer('Name','actionProb'+string(j))];
        Network0 = addLayers(Network0,Network{j});
        Network0 = connectLayers(Network0,'infc','ActorFC'+string(j)+" - 1");  
        Network0 = connectLayers(Network0,'actionProb'+string(j),'out/in'+string(j));
    end
    
    
end