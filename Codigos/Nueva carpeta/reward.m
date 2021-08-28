function reward(block)
% Level-2 MATLAB file S-Function for inherited sample time demo.

%   Copyright 1990-2009 The MathWorks, Inc.

  setup(block);
  
%endfunction

function setup(block)
  
  %% Register number of input and output ports
  block.NumInputPorts  = 1;
  block.NumOutputPorts = 1;

  %% Setup functional port properties to dynamically
  %% inherited.
  block.SetPreCompInpPortInfoToDynamic;
  block.SetPreCompOutPortInfoToDynamic;
 
  block.InputPort(1).Dimensions        = [1 4];
  block.InputPort(1).DirectFeedthrough = false;
  
  block.OutputPort(1).Dimensions       = 1;
  
  %% Set block sample time to inherited
  block.SampleTimes = [-1 0];
  
  %% Set the block simStateCompliance to default (i.e., same as a built-in block)
  block.SimStateCompliance = 'DefaultSimState';

  %% Register methods
  block.RegBlockMethod('Outputs',                 @Output);  
  

function Output(block)

 low = 20;
 max = 25;
 reward = 0; 
 %disp(length(block.InputPort(1).Data))
 for i=1:length(block.InputPort(1).Data)
     %disp(i)
     aux = block.InputPort(1).Data(i);
     %disp(aux)
     if (aux < low || aux > max) && (abs(aux - low) >= 1 && abs(aux-max) >= 1)
        %disp(abs(aux - low))
        reward = reward - log(abs(min( abs(aux - low), abs(aux - max) )));
        %reward = reward + 0;
        %disp('if13')
     end
     %disp('if1')
     if (aux >= low && aux <= max)
        reward = reward + 15;
     end
     %disp('if2')
     if (aux < low || aux > max) && (abs(aux - low) < 1 && abs(aux-max) < 1)
        reward = reward + 0.7;
     end
     %disp('if3')
 end
  
 block.OutputPort(1).Data = reward;
  
%endfunction


