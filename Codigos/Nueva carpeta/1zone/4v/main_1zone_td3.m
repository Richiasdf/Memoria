%% Clearing the workspace & setting parameters
clear all
resumeTraining = false;% Set to true, to resume training from a saved agent
%Range of temperature wanted
low = 22; %temperatura minima
high = 25; % temperatura máxima
beta = 0.08; % peso de la utilización de energía en la recompensa.
%ci=[37;30]; %initial condition
Ts = 60*1; %2 min - HVAC System sample Time
Ts2 = 60*10; %n min - Neural network Sample time
Tf = 3600*24; % n horas  - Simulation Time
maxepisodes = 25000; %episodes to finish the training
%device for critic & actor
use_parallel = true;
device = "cpu";
%RL Layers
criticlayers = [2,1];
criticNeurons = [64, 32, 32];
actorlayers = 2;
actorNeurons = [32, 32];

%% Load and open the system in simulink
mdl2 = 'hvac_1zone_v3';
%open_system(mdl2)

%% Define model parameters
load('values1z.mat')


%% Reinforcement learning
%Se definen las observaciones, rlNumericSpec es para observaciones de
%rangos continuos2
obsInfo = rlNumericSpec([4 1]);
obsInfo.Name = 'observations';
obsInfo.Description = 'estado 1, estado 2, estado 3 and estado 4';
numObservations = obsInfo.Dimension(1);

%Se definen las acciones, rlNumericSpec es para acciones continuas entre
%los rangos definidos.
actInfo = rlNumericSpec([2 1], 'LowerLimit', [0;-30] , 'UpperLimit', [5;0]);
actInfo.Name = 'actions';
numActions = actInfo.Dimension(1);

%Se genera el ambiente, tomando en cuenta el modelo, el bloque del agente
%de RL en el modelo, y las informaciones de observación y acción.
env = rlSimulinkEnv(mdl2,'hvac_1zone_v3/RL Agent',...
    obsInfo,actInfo);
%Función para reiniciar el sistema
%Esta función es útil cuando se quiere usar distintas condiciones iniciales
%en cada inicio de un episodio.
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
agentOpts.TargetPolicySmoothModel.StandardDeviationMin = [0.000001; 0.00001] ;
agentOpts.TargetPolicySmoothModel.StandardDeviation = [0.001; 0.03]; % target policy noise
agentOpts.TargetPolicySmoothModel.StandardDeviationDecayRate = [1e-3; 1e-3];

%agentOpts.TargetPolicySmoothModel.LowerLimit = -0.7;
%agentOpts.TargetPolicySmoothModel.UpperLimit = 0.7;
agentOpts.ExplorationModel = rl.option.OrnsteinUhlenbeckActionNoise;
agentOpts.ExplorationModel.MeanAttractionConstant = 0.2/Ts2;
agentOpts.ExplorationModel.StandardDeviation = [0.4/sqrt(Ts2) ; 0.3];
agentOpts.ExplorationModel.StandardDeviationDecayRate = [5e-4; 5e-4]*3;
agentOpts.ExplorationModel.StandardDeviationMin = [0.05/sqrt(Ts2); 0.08];
agentOpts.ExplorationModel.Mean = [0; 0];

% Train the agent
% Set ResetExperienceBufferBeforeTraining to false to keep experience from the previous session
agentOpts.ResetExperienceBufferBeforeTraining = ~(resumeTraining);

if resumeTraining
    load('1stTD3_agent.mat');
    agentOpts.TargetPolicySmoothModel.StandardDeviationDecayRate = 1e-5;
    agentOpts.ExplorationModel = rl.option.OrnsteinUhlenbeckActionNoise;
    agentOpts.ExplorationModel.MeanAttractionConstant = 0.2/Ts2;
    agentOpts.ExplorationModel.StandardDeviation = [0.03;0.1]/sqrt(Ts2);
    agentOpts.ExplorationModel.StandardDeviationMin = 0.0015/sqrt(Ts2);
    agentOpts.ExplorationModel.StandardDeviationDecayRate = 1e-5;

else
    %Red neuronal Critico
    criticOpts = rlRepresentationOptions('LearnRate',1e-03,'GradientThreshold',.2, 'UseDevice', device);
    criticNetwork = MakeCritic(numObservations,numActions,criticlayers,criticNeurons);
    critic = rlQValueRepresentation(criticNetwork,obsInfo,actInfo,'Observation',{'State'},'Action',{'Action'},criticOpts);
    criticNetwork2 = MakeCritic(numObservations,numActions,criticlayers,criticNeurons);
    critic2 = rlQValueRepresentation(criticNetwork2,obsInfo,actInfo,'Observation',{'State'},'Action',{'Action'},criticOpts);
    %Red neuronal Actor
    actorNetwork = MakeActor(numObservations,numActions, actorlayers , actorNeurons);
    actorOptions = rlRepresentationOptions('LearnRate',2e-03,'GradientThreshold',.2, 'UseDevice', device);
    actor = rlDeterministicActorRepresentation(actorNetwork,obsInfo,actInfo,'Observation',{'State'},'Action',{'Action'},actorOptions);
    % Create a fresh new agent
    %createNetworks
    agent = rlTD3Agent(actor, [critic,critic2], agentOpts);
end


maxsteps = ceil(Tf/Ts2);
StopReward = 115;
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


%% Funciones a utilizar

function criticNetwork = MakeCritic(obs,acts,layers,neurons)
    statePath = featureInputLayer(obs,'Normalization','none','Name','State');
    for i=1:layers(1)
        if mod(i,2) == 0
            statePath = [statePath; tanhLayer('Name','StateTanh'+string(i/2))];
        end
        statePath = [statePath; fullyConnectedLayer(neurons(i),'Name','CriticStateFC'+string(i))];
        
    end
    actionPath = featureInputLayer(acts,'Normalization','none','Name','Action');
    for i=1:layers(2)
        if mod(i,2) == 0
            actionPath = [actionPath;tanhLayer('Name','ActionTanh'+string(i/2))];
        end
        actionPath = [actionPath; fullyConnectedLayer(neurons(i+layers(1)),'Name','CriticActionFC'+string(i))]; 
    end
    commonPath = [
        additionLayer(2,'Name','add')
        tanhLayer('Name','CommonTanh')
        fullyConnectedLayer(1,'Name','CriticOutput')];

    criticNetwork = layerGraph();
    criticNetwork = addLayers(criticNetwork,statePath);
    criticNetwork = addLayers(criticNetwork,actionPath);
    criticNetwork = addLayers(criticNetwork,commonPath);
    criticNetwork = connectLayers(criticNetwork,'CriticStateFC'+string(layers(1)),'add/in1');
    criticNetwork = connectLayers(criticNetwork,'CriticActionFC'+string(layers(2)),'add/in2');
    
end

function Network = MakeActor(obs,acts,layers,neurons)
    Network = featureInputLayer(obs,'Normalization','none','Name','State');
    for i=1:layers(1)
        Network = [Network; fullyConnectedLayer(neurons(i),'Name','ActorFC'+string(i)); tanhLayer('Name','ActorTanh'+string(i/2))];
    end
    Network = [Network  ;fullyConnectedLayer(acts,'Name','Action2');sigmoidLayer('Name','Sigmoid') ;scalingLayer('Name','Action','Scale',[5 -30]')];
end