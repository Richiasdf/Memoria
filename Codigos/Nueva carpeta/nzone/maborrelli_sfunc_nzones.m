function [sys,x0,str,ts,simStateCompliance]=maborrelli_sfunc_nzones(t,x,u,flag,ic,params)
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
    [sys,x0,str,ts,simStateCompliance] = mdlInitializeSizes(ic,params);
    %disp('flag 0')

  %%%%%%%%%%%%%%%
  % Derivatives %
  %%%%%%%%%%%%%%%
  case 1
    sys = mdlDerivatives(t,x,u,params);
    %end
    %disp('flag 1')

  %%%%%%%%%%%%%%%%%%%%%%%%
  % Update and Terminate %
  %%%%%%%%%%%%%%%%%%%%%%%%
  case {2,9}
    %disp('aca')
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
function [sys,x0,str,ts,simStateCompliance] = mdlInitializeSizes(ic,params)

nz=size(params.Neigh,1);

sizes = simsizes;
sizes.NumContStates  = 2*nz;
sizes.NumDiscStates  = 0;
sizes.NumOutputs     = 2*nz;
sizes.NumInputs      = 2+nz+1+nz;
sizes.DirFeedthrough = 0;
sizes.NumSampleTimes = 1;

sys = simsizes(sizes);
str = [];
x0  = ic;
ts  = [0 0];   % sample time: [period, offset]

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

%
%=============================================================================
% mdlOutputs
% Return the output vector for the S-function
%=============================================================================
%
function sys = mdlOutputs(x)
sys = x;

% end mdlOutputs
