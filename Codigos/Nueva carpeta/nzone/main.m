%% Load and open the system in simulink
mdl2 = 'hvac_maborrelli_singapur_nzones_with_MPC_v3_manualdist_3';
open_system(mdl2)

%% Define model parameters
load('values.mat')


%% Reinforcement learning
obsInfo = rlNumericSpec([4 1], 'LowerLimit', -20, 'UpperLimit', 40);
obsInfo.Name = 'observations';
obsInfo.Description = 'estado 1, estado 2, estado 3 and estado 4';
numObservations = obsInfo.Dimension(1);

actInfo = rlNumericSpec([6 1], 'LowerLimit', [0.001; 0 ; 0; 0 ; -15; 0], 'UpperLimit', [5; 5; 5; 5; 0; 1]);
actInfo.Name = 'actions';
numActions = actInfo.Dimension(1);

env = rlSimulinkEnv(mdl2,'hvac_maborrelli_singapur_nzones_with_MPC_v3_manualdist_3/RL Agent',...
    obsInfo,actInfo);

%in = Simulink.SimulationInput(mdl2);
%disp(in)
%Función para reiniciar el sistema
%env.ResetFcn = @(in) localResetFcn(in);

%Range of temperature wanted
min_temp = 22;
max_temp = 25;

%Sample Time & Simulation Time
Ts2 = 900; %15 min
Tf = 21600; % 6 horas

rng(0)

%Red neuronal Critico
%CAMBIAR

statePath = [
    featureInputLayer(numObservations,'Normalization','none','Name','State')
    fullyConnectedLayer(128,'Name','CriticStateFC1')
    fullyConnectedLayer(128,'Name','CriticStateFC3')
    tanhLayer('Name','CriticTanh1')
    fullyConnectedLayer(64,'Name','CriticStateFC2')];
actionPath = [
    featureInputLayer(numActions,'Normalization','none','Name','Action')
    fullyConnectedLayer(128,'Name','CriticActionFC1')
    fullyConnectedLayer(64,'Name','CriticActionFC3')];
commonPath = [
    additionLayer(2,'Name','add')
    tanhLayer('Name','CriticTanh')
    fullyConnectedLayer(1,'Name','CriticOutput')];

criticNetwork = layerGraph();
criticNetwork = addLayers(criticNetwork,statePath);
criticNetwork = addLayers(criticNetwork,actionPath);
criticNetwork = addLayers(criticNetwork,commonPath);
criticNetwork = connectLayers(criticNetwork,'CriticStateFC2','add/in1');
criticNetwork = connectLayers(criticNetwork,'CriticActionFC3','add/in2');

criticOpts = rlRepresentationOptions('LearnRate',1e-02,'GradientThreshold',1);
critic = rlQValueRepresentation(criticNetwork,obsInfo,actInfo,'Observation',{'State'},'Action',{'Action'},criticOpts);


actorNetwork = [
    featureInputLayer(numObservations,'Normalization','none','Name','State')
    fullyConnectedLayer(128, 'Name','actorFC')
    fullyConnectedLayer(64, 'Name','actorFC2')
    tanhLayer('Name','ActorCommonRelu2')
    fullyConnectedLayer(numActions,'Name','Action')
    ];

actorOptions = rlRepresentationOptions('LearnRate',1e-02,'GradientThreshold',1);


%Ver mas en detalle
actor = rlDeterministicActorRepresentation(actorNetwork,obsInfo,actInfo,'Observation',{'State'},'Action',{'Action'},actorOptions);


agentOpts = rlDDPGAgentOptions(...
    'SampleTime',Ts2,...
    'TargetSmoothFactor',1e-3,...
    'DiscountFactor',1, ...
    'MiniBatchSize',128, ...
    'ExperienceBufferLength',1e6); 
    
agentOpts.NoiseOptions.StandardDeviation = 0.3;
agentOpts.NoiseOptions.StandardDeviationDecayRate = 1e-4;
% Averiguar***
agentOpts.NoiseOptions.MeanAttractionConstant = 0.2/Ts2;

agent = rlDDPGAgent(actor,critic,agentOpts);


maxepisodes = 2000;
maxsteps = ceil(Tf/Ts2);
StopReward = (maxsteps*0.85)*2;
%    'Verbose',false, ...
trainOpts = rlTrainingOptions(...
    'MaxEpisodes',maxepisodes, ...
    'MaxStepsPerEpisode',maxsteps, ...
    'UseParallel',true, ...
    'ScoreAveragingWindowLength',25, ...
    'Plots','training-progress',...
    'StopTrainingCriteria','AverageReward',...
    'StopTrainingValue',StopReward);

trainingStats = train(agent,env,trainOpts);


simOpts = rlSimulationOptions('MaxSteps',maxsteps,'StopOnError','on');
experiences = sim(env,agent,simOpts);

%Save the train network
%save('train_agent2.mat','agent','experiences')

