mdl = 'rlFlyingRobotEnv';
open_system(mdl)
integratedMdl = 'IntegratedFlyingRobot';
[~,agentBlk,observationInfo,actionInfo] = createIntegratedEnv(mdl,integratedMdl);