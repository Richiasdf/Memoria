function [sys,x0,str,ts,simStateCompliance]=maborrelli_sfunc_nzones(t,x,u,flag,ic,params,ST)
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
    [sys,x0,str,ts,simStateCompliance] = mdlInitializeSizes(ic,params,ST);
    %disp('flag 0')

  %%%%%%%%%%%%%%%
  % Derivatives %
  %%%%%%%%%%%%%%%
  case 1
    sys = mdlDerivatives(t,x,u,params);
    %end
    %disp('flag 1')

  %%%%%%%%%%%%%%%%%%%%%%%%
  % Update %
  %%%%%%%%%%%%%%%%%%%%%%%%
  
  case 2
    sys=mdlUpdate(x,u,params,ST);
    
  %%%%%%%%%%%%%
  % Terminate %
  %%%%%%%%%%%%%
  
  case 9
    sys = []; % do nothing

  %%%%%%%%%%
  % Output %
  %%%%%%%%%%
  case 3
    sys = mdlOutputs(x); 


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
function [sys,x0,str,ts,simStateCompliance] = mdlInitializeSizes(ic,params,ST)

nz=size(params.Neigh,1);

sizes = simsizes;
sizes.NumContStates  = 0;
sizes.NumDiscStates  = 5*nz;
sizes.NumOutputs     = 5*nz;
sizes.NumInputs      = nz*3+3+10;
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
[sys,~]=dynamics_hvac_maborrelli_singapur_nz(x,u,params);

% end mdlDerivatives

function sys=mdlUpdate(x,u,params,ST)
[sys,~]=dynamics_hvac_maborrelli_singapur_nz(x,u,params,ST);

%
%=============================================================================
% mdlOutputs
% Return the output vector for the S-function
%=============================================================================
%
function sys = mdlOutputs(x)
sys = x;

% end mdlOutputs
