%& Clearing the workspace
clear all


% Set to true, to resume training from a saved agent
resumeTraining = true;

%Range of temperature wanted
low = 22; %temperatura minima
high = 25; % temperatura máxima
beta = 0.07;
%Sample Time & Simulation Time
Ts = 60; %1 min - HVAC System sample Time
Ts2 = 10*60; %10 min - Neural network Sample time
Tf = 12*3600; % 6 horas  - Simulation Time
maxepisodes = 30000;

%device for critic & actor
device = "gpu";
%Use parallel compute
use_parallel = true;
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
actInfo = rlNumericSpec([3 1], 'LowerLimit', [0;-30;0] , 'UpperLimit', [5;0;0.9]);
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
    'DiscountFactor',0.95, ...
    'MiniBatchSize',256, ...
    'SaveExperienceBufferWithAgent',true, ...
    'ExperienceBufferLength',1e6); 
    
%Opciones del ruido aplicado a las acciones tomadas, si se tiene más de una
%acción, y estas no se encuentran en rangos similares, se recomienda
%utilizar un vector en vez de un escalar.
agentOpts.NoiseOptions.StandardDeviation = [.5; .3*sqrt(Ts2); .1]/sqrt(Ts2);
agentOpts.NoiseOptions.StandardDeviationDecayRate = [5e-4; 2e-4 ; 4e-5]*2;
agentOpts.NoiseOptions.MeanAttractionConstant = 0.2/Ts2;
agentOpts.NoiseOptions.StandardDeviationMin = [0.015; 0.1; 0.0002]/sqrt(Ts2);


% Train the agent
% Set ResetExperienceBufferBeforeTraining to false to keep experience from the previous session
agentOpts.ResetExperienceBufferBeforeTraining = ~(resumeTraining);

if resumeTraining
    load('train_agent4.mat');
    agent.AgentOptions.NoiseOptions.StandardDeviationMin = [0.01; 0.05; 0.2]/sqrt(Ts2);
    agent.AgentOptions.NoiseOptions.StandardDeviation = [.1; .5; .05]/sqrt(Ts2);
    agent.AgentOptions.NoiseOptions.StandardDeviationDecayRate = [1e-7; 1e-7 ; 1e-7];
else
    %Red neuronal Critico
    criticOpts = rlRepresentationOptions('LearnRate',2e-03,'GradientThreshold',1, 'UseDevice', device);
    criticNetwork = MakeCritic(numObservations,numActions,[2,1],[128, 128, 128]);
    critic = rlQValueRepresentation(criticNetwork,obsInfo,actInfo,'Observation',{'State'},'Action',{'Action'},criticOpts);
    %Red neuronal Actor
    actorNetwork = MakeActor(numObservations,numActions, 2 , [128, 64]);
    actorOptions = rlRepresentationOptions('LearnRate',1e-03,'GradientThreshold',1, 'UseDevice', device);
    actor = rlDeterministicActorRepresentation(actorNetwork,obsInfo,actInfo,'Observation',{'State'},'Action',{'Action'},actorOptions);
    % Create a fresh new agent
    agent = rlDDPGAgent(actor, critic, agentOpts);
end


maxsteps = ceil(Tf/Ts2);
StopReward = -10;

%    'Verbose',true, ...      'Plots','none',...
trainOpts = rlTrainingOptions(...
    'MaxEpisodes',maxepisodes, ...
    'MaxStepsPerEpisode',maxsteps, ...
    'UseParallel',use_parallel, ...
    'ScoreAveragingWindowLength',20, ...
    'Verbose',true, ...
    'Plots','none',...
    'StopTrainingCriteria','AverageReward',...
    'StopTrainingValue',StopReward);

simOpts = rlSimulationOptions('MaxSteps',maxsteps,'StopOnError','on');

if use_parallel
    parpool;
    parfevalOnAll( @warning, 0, 'off', 'all');
    clearvars use_parallel maxepisodes maxsteps StopReward resumeTraining device
else 
    clearvars use_parallel maxepisodes maxsteps StopReward resumeTraining device
end
%Se comienza a entrenar el agente y se abre el gráfico de entrenamiento.
trainingStats = train(agent,env,trainOpts);

%Se carga un agente pre entrenado:
%load('train_agent.mat')

%Se simula el agente entrenado

sim(env,agent,simOpts);

%Save the train network
%save('train_agent7.mat','agent','simOpts')


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

function Network = MakeActor(obs,acts,layers,neurons)
    Network = featureInputLayer(obs,'Normalization','none','Name','State');
    for i=1:layers(1)
        Network = [Network; fullyConnectedLayer(neurons(i),'Name','ActorFC'+string(i)); reluLayer('Name','ActorTanh'+string(i))];
    end
    Network = [Network ; fullyConnectedLayer(acts,'Name','Action2') ;sigmoidLayer('Name','Sigmoid') ;scalingLayer('Name','Action','Scale',[5 -30 0.9]')];
end