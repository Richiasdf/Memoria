function [obsInfo,actInfo,env] = makeenv(mdl,obs,acts,low,upp,obs_low,obs_h)
    sub = (split(mdl,"/"));
    mdl2=sub(2);
    sub = sub(1);
    cd(sub)
    %Se definen las observaciones, rlNumericSpec es para observaciones de
    %rangos continuos2
    obsInfo = rlNumericSpec([obs 1], 'LowerLimit', obs_low , 'UpperLimit', obs_h);
    obsInfo.Name = 'observations';
    obsInfo.Description = 'estado 1, estado 2, estado 3 and estado 4';
    %Se definen las acciones, rlNumericSpec es para acciones continuas entre
    %los rangos definidos.
    actInfo = rlNumericSpec([acts 1], 'LowerLimit', low , 'UpperLimit', upp);
    actInfo.Name = 'actions';
    %Se genera el ambiente, tomando en cuenta el modelo, el bloque del agente
    %de RL en el modelo, y las informaciones de observación y acción.
    env = rlSimulinkEnv(mdl2,mdl2+"/RL Agent",obsInfo,actInfo);
    clearvars sub mdl2
    cd ..
   