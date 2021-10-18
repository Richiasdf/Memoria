%& Clearing the workspace
clear all


% Set to true, to resume training from a saved agent
resumeTraining = true;

%Range of temperature wanted
low = 22; %temperatura minima
high = 25; % temperatura máxima

%Sample Time & Simulation Time
Ts = 60; %1 min - HVAC System sample Time
Ts2 = 600; %10 min - Neural network Sample time
Tf = 21600; % 6 horas  - Simulation Time

%device for critic & actor
device = "cpu";

%% Load and open the system in simulink
mdl2 = 'hvac_1zone_v3';
%open_system(mdl2)
warning('off','all')
%% Define model parameters
load('values1z.mat')


%% Reinforcement learning
%Se definen las observaciones, rlNumericSpec es para observaciones de
%rangos continuos2
obsInfo = rlNumericSpec([3 1], 'LowerLimit', [-20; -20; 0], 'UpperLimit', [40; 55; 1e6]);
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
    'MiniBatchSize',128, ...
    'SaveExperienceBufferWithAgent',true, ...
    'ExperienceBufferLength',1e6); 
    
%Opciones del ruido aplicado a las acciones tomadas, si se tiene más de una
%acción, y estas no se encuentran en rangos similares, se recomienda
%utilizar un vector en vez de un escalar.
agentOpts.NoiseOptions.StandardDeviation = [3.35; 3.4; 3.2];
agentOpts.NoiseOptions.StandardDeviationDecayRate = [1e-4; 5e-4 ; 1e-3];
agentOpts.NoiseOptions.MeanAttractionConstant = 0.15/Ts2;
agentOpts.NoiseOptions.StandardDeviationMin = [0.015; 0.01; 0.01];


% Train the agent
% Set ResetExperienceBufferBeforeTraining to false to keep experience from the previous session
agentOpts.ResetExperienceBufferBeforeTraining = ~(resumeTraining);

if resumeTraining
    agentOpts.NoiseOptions.StandardDeviation = [2.25; 1.2; 1.15];
    load('train_agent.mat');
else
    %Red neuronal Critico
    criticOpts = rlRepresentationOptions('LearnRate',1e-03,'GradientThreshold',1, 'UseDevice', device);
    criticNetwork = MakeCritic(numObservations,numActions,[2,1],[80, 64, 64]);
    critic = rlQValueRepresentation(criticNetwork,obsInfo,actInfo,'Observation',{'State'},'Action',{'Action'},criticOpts);
    %Red neuronal Actor
    actorNetwork = MakeActor(numObservations,numActions, 1 , 80);
    actorOptions = rlRepresentationOptions('LearnRate',1e-03,'GradientThreshold',1, 'UseDevice', device);
    actor = rlDeterministicActorRepresentation(actorNetwork,obsInfo,actInfo,'Observation',{'State'},'Action',{'Action'},actorOptions);
    % Create a fresh new agent
    agent = rlDDPGAgent(actor, critic, agentOpts);
end
maxepisodes = 55000;
maxsteps = ceil(Tf/Ts2);
StopReward = -7;
%    'Verbose',true, ...      'Plots','none',...
trainOpts = rlTrainingOptions(...
    'MaxEpisodes',maxepisodes, ...
    'MaxStepsPerEpisode',maxsteps, ...
    'UseParallel',true, ...
    'ScoreAveragingWindowLength',25, ...
    'Verbose',true, ...
    'StopTrainingCriteria','AverageReward',...
    'StopTrainingValue',StopReward);
%Se comienza a entrenar el agente y se abre el gráfico de entrenamiento.
trainingStats = train(agent,env,trainOpts);

%Se carga un agente pre entrenado:
%load('train_agent2.mat')

%Se simula el agente entrenado
simOpts = rlSimulationOptions('MaxSteps',maxsteps,'StopOnError','on');
sim(env,agent,simOpts);

%Save the train network
%save('train_agent.mat','agent','simOpts')


%% Funciones a utilizar

function criticNetwork = MakeCritic(obs,acts,layers,neurons)
    statePath = featureInputLayer(obs,'Normalization','none','Name','State');
    for i=1:layers(1)
        if even(i)
            statePath = [statePath; tanhLayer('Name','StateTanh'+string(i/2))];
        end
        statePath = [statePath; fullyConnectedLayer(neurons(i),'Name','CriticStateFC'+string(i))]; 
        
    end
    actionPath = featureInputLayer(acts,'Normalization','none','Name','Action');
    for i=1:layers(2)
        if even(i)
            actionPath = [actionPath;tanhLayer('Name','ActionTanh'+string(i/2))];
        end
        actionPath = [actionPath; fullyConnectedLayer(neurons(i+layers(1)),'Name','CriticActionFC'+string(i))]; 
    end
    commonPath = [
        additionLayer(2,'Name','add')
        tanhLayer('Name','CriticTanh')
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
        Network = [Network; fullyConnectedLayer(neurons(i),'Name','ActorFC'+string(i)); tanhLayer('Name','ActorTanh'+string(i/2)) ]; 
    end
    Network = [Network; fullyConnectedLayer(acts,'Name','Action')];
end