function [sys,x0,str,ts,simStateCompliance]=maborrelli_sfunc_1zone_4(t,x,u,flag,ic,params, ST)
%LIMINTM Limited integrator implementation.
%   Example MATLAB file S-function implementing a continuous limited integrator
%   where the output is bounded by lower bound (LB) and upper bound (UB)
%   with initial conditions (XI).
%   
%   See sfuntmpl.m for a general S-function template.
%
%   See also SFUNTMPL.
    
%   Copyright 1990-2009 The MathWorks, Inc.

switch flag

  %%%%%%%%%%%%%%%%%%
  % Initialization %
  %%%%%%%%%%%%%%%%%%
  case 0         
    [sys,x0,str,ts,simStateCompliance] = mdlInitializeSizes(ic, ST);

  %%%%%%%%%%%%%%%
  % Derivatives %
  %%%%%%%%%%%%%%%
  case 1
    %sys = mdlDerivatives(t,x,u,params);
    disp('derivatives')

  %%%%%%%%%%%%%%%%%%%%%%%%
  % Update %
  %%%%%%%%%%%%%%%%%%%%%%%%
  
  case 2
    sys=mdlUpdate(x,u,params,ST);
    %disp(x)
    
  %%%%%%%%%%%%%
  % Terminate %
  %%%%%%%%%%%%%
  
  case 9
    sys = []; % do nothing

  %%%%%%%%%%
  % Output %
  %%%%%%%%%%
  case 3
    sys = mdlOutputs(t,x,u); 

  otherwise
    DAStudio.error('Simulink:blocks:unhandledFlag', num2str(flag));
end

% end limintm

%
%=============================================================================
% mdlInitializeSizes
% Return the sizes, initial conditions, and sample times for the S-function.
%=============================================================================
%
function [sys,x0,str,ts,simStateCompliance] = mdlInitializeSizes(ic,ST)

sizes = simsizes;
sizes.NumContStates  = 0;
sizes.NumDiscStates  = 3;
sizes.NumOutputs     = 3;
sizes.NumInputs      = 8;
sizes.DirFeedthrough = 0;
sizes.NumSampleTimes = 1;

sys = simsizes(sizes);
str = [];
x0  = ic;
ts  = [ST 0];   % sample time: [period, offset]

% speicfy that the simState for this s-function is same as the default
simStateCompliance = 'DefaultSimState';

% end mdlInitializeSizes

%
%=============================================================================
% mdlDerivatives
% Compute derivatives for continuous states.
%=============================================================================
%
function sys = mdlDerivatives(t,x,u,params)
%disp(u)
[sys,~]=dynamics_hvac_maborrelli_singapur_1zone(x,u,params);


% end mdlDerivatives


%=============================================================================
% mdlUpdate
% Handle discrete state updates, sample time hits, and major time step
% requirements.
%=============================================================================
%
function sys=mdlUpdate(x,u,params,ST)
[sys,~]=dynamics_hvac_maborrelli_singapur_1zoned4(x,u,params,ST);
%sys = x;

% end mdlUpdate

%
%=============================================================================
% mdlOutputs
% Return the output vector for the S-function
%=============================================================================
%
function sys = mdlOutputs(t,x,u)
%disp(u)
%e = (params.cp/params.eta)*u(3)*u(4)+ params.kf*(u(3))^2;
%if isnan(e)
%    e = 0;
%end
%x(3) = e;
%disp(x)
sys = x;


% end mdlOutputs
