%& Clearing the workspace
clear all
% Set to true, to resume training from a saved agent
resumeTraining = false;

%Range of temperature wanted
low = 22; %temperatura minima
high = 25; % temperatura máxima
beta = 0.07;
%Sample Time & Simulation Time
%ci=[37;30];
Ts = 60*3; %2 min - HVAC System sample Time
Ts2 = 60*10; %n min - Neural network Sample time
Tf = 3600*6; % n horas  - Simulation Time
maxepisodes = 25000;

%device for critic & actor
use_parallel = true;
device = "cpu";
%RL Layers
criticlayers = [2,1];
criticNeurons = [128, 128, 128];
actorlayers = 2;
actorNeurons = [128, 128];

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
agentOpts = rlDDPGAgentOptions(...
    'SampleTime',Ts2,...
    'TargetSmoothFactor',1e-3,...
    'DiscountFactor',0.8, ...
    'MiniBatchSize',128, ...
    'SaveExperienceBufferWithAgent',true, ...
    'ExperienceBufferLength',1e6); 
    
%Opciones del ruido aplicado a las acciones tomadas, si se tiene más de una
%acción, y estas no se encuentran en rangos similares, se recomienda
%utilizar un vector en vez de un escalar.
agentOpts.NoiseOptions.StandardDeviation = [0.5/sqrt(Ts2) ; 0.3];
agentOpts.NoiseOptions.StandardDeviationDecayRate =[5e-4; 1e-4];
agentOpts.NoiseOptions.MeanAttractionConstant = 0.2/Ts2;
agentOpts.NoiseOptions.StandardDeviationMin = [0.015; 0.1]/sqrt(Ts2);


% Train the agent
% Set ResetExperienceBufferBeforeTraining to false to keep experience from the previous session
agentOpts.ResetExperienceBufferBeforeTraining = ~(resumeTraining);

if resumeTraining
    load('train_agent1.mat');
    agent.AgentOptions.NoiseOptions.StandardDeviationMin = [0.001/sqrt(Ts2);0];
    agent.AgentOptions.NoiseOptions.StandardDeviation = [.012/sqrt(Ts2); 0];
    agent.AgentOptions.NoiseOptions.StandardDeviationDecayRate = [3e-3; 3e-3 ];
    
else
    %Red neuronal Critico
    criticOpts = rlRepresentationOptions('LearnRate',2e-03,'GradientThreshold',0.1, 'UseDevice', device);
    criticNetwork = MakeCritic(numObservations,numActions,criticlayers,criticNeurons);
    critic = rlQValueRepresentation(criticNetwork,obsInfo,actInfo,'Observation',{'State'},'Action',{'Action'},criticOpts);
    %Red neuronal Actor
    actorNetwork = MakeActor(numObservations,numActions, actorlayers , actorNeurons);
    actorOptions = rlRepresentationOptions('LearnRate',1e-03,'GradientThreshold',0.1, 'UseDevice', device);
    actor = rlDeterministicActorRepresentation(actorNetwork,obsInfo,actInfo,'Observation',{'State'},'Action',{'Action'},actorOptions);
    % Create a fresh new agent
    %createNetworks
    agent = rlDDPGAgent(actor, critic, agentOpts);
end


maxsteps = ceil(Tf/Ts2);
StopReward = 32;
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
%load('1stTD3_agent.mat')

%Se simula el agente entrenado

sim(env,agent,simOpts);

%Save the train network
%save('train_agent6.mat','agent','simOpts')


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
            actionPath = [actionPath;tanhLayer('Name','ActionTanh'+string(i/2))];
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

function Network = MakeActor(obs,acts,layers,neurons)
    Network = featureInputLayer(obs,'Normalization','none','Name','State');
    for i=1:layers(1)
        Network = [Network; fullyConnectedLayer(neurons(i),'Name','ActorFC'+string(i)); reluLayer('Name','ActorTanh'+string(i/2))];
    end
    Network = [Network  ; fullyConnectedLayer(acts,'Name','Action2');sigmoidLayer('Name','Sigmoid') ;scalingLayer('Name','Action','Scale',[5 -30]')];
end