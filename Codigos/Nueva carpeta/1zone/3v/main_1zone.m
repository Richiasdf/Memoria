%& Clearing the workspace
clear all

%% Load and open the system in simulink
mdl2 = 'hvac_1zone_v3';
open_system(mdl2)

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
%env.ResetFcn = @(in) localResetFcn(in);

%Range of temperature wanted
min_temp = 22;
max_temp = 25;

%Sample Time & Simulation Time
Ts2 = 600; %10 min
Tf = 21600; % 6 horas

%Set random seed.
rng(0)

%Red neuronal Critico
%CAMBIAR
%fullyConnectedLayer(84,'Name','CriticStateFC1')
%tanhLayer('Name','CriticTanh1')
%14 6 6
statePath = [
    featureInputLayer(numObservations,'Normalization','none','Name','State')
    fullyConnectedLayer(40,'Name','CriticStateFC1')
    tanhLayer('Name','CriticTanh1')
    fullyConnectedLayer(25,'Name','CriticStateFC2')];
actionPath = [
    featureInputLayer(numActions,'Normalization','none','Name','Action')
    fullyConnectedLayer(25,'Name','CriticActionFC1')];
commonPath = [
    additionLayer(2,'Name','add')
    tanhLayer('Name','CriticTanh')
    fullyConnectedLayer(1,'Name','CriticOutput')];

criticNetwork = layerGraph();
criticNetwork = addLayers(criticNetwork,statePath);
criticNetwork = addLayers(criticNetwork,actionPath);
criticNetwork = addLayers(criticNetwork,commonPath);
criticNetwork = connectLayers(criticNetwork,'CriticStateFC2','add/in1');
criticNetwork = connectLayers(criticNetwork,'CriticActionFC1','add/in2');

criticOpts = rlRepresentationOptions('LearnRate',1e-03,'GradientThreshold',1);
critic = rlQValueRepresentation(criticNetwork,obsInfo,actInfo,'Observation',{'State'},'Action',{'Action'},criticOpts);

%Red neuronal Actor
actorNetwork = [
    featureInputLayer(numObservations,'Normalization','none','Name','State')
    fullyConnectedLayer(9, 'Name','actorFC2')
    tanhLayer('Name','ActorCommonRelu2')
    fullyConnectedLayer(numActions,'Name','Action')
    ];

actorOptions = rlRepresentationOptions('LearnRate',1e-03,'GradientThreshold',1);


%Ver mas en detalle
actor = rlDeterministicActorRepresentation(actorNetwork,obsInfo,actInfo,'Observation',{'State'},'Action',{'Action'},actorOptions);


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
agentOpts.NoiseOptions.StandardDeviation = [0.35; 0.4; 0.2];
agentOpts.NoiseOptions.StandardDeviationDecayRate = [1e-4; 5e-4 ; 1e-3];
% Averiguar***
agentOpts.NoiseOptions.MeanAttractionConstant = 0.15/Ts2;
agentOpts.NoiseOptions.StandardDeviationMin = [0.015; 0.01; 0.01];


% Train the agent
% Set to true, to resume training from a saved agent
resumeTraining = false;
% Set ResetExperienceBufferBeforeTraining to false to keep experience from the previous session
agentOpts.ResetExperienceBufferBeforeTraining = ~(resumeTraining);

if resumeTraining
    agentOpts.NoiseOptions.StandardDeviation = [0.25; 0.2; 0.15];
    load('train_agent2.mat');
else
    % Create a fresh new agent
    agent = rlDDPGAgent(actor, critic, agentOpts);
end



maxepisodes = 16000;
maxsteps = ceil(Tf/Ts2);
StopReward = -7;
%    'Verbose',false, ...
trainOpts = rlTrainingOptions(...
    'MaxEpisodes',maxepisodes, ...
    'MaxStepsPerEpisode',maxsteps, ...
    'UseParallel',true, ...
    'ScoreAveragingWindowLength',25, ...
    'Plots','training-progress',...
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
save('train_agent2.mat','agent','simOpts')

