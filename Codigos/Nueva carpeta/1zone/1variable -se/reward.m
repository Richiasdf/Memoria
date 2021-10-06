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
 
  block.InputPort(1).Dimensions        = [2 1];
  block.InputPort(1).DirectFeedthrough = false;
  
  block.OutputPort(1).Dimensions       = 1;
  
  %% Set block sample time to inherited
  block.SampleTimes = [-1 0];
  
  %% Set the block simStateCompliance to default (i.e., same as a built-in block)
  block.SimStateCompliance = 'DefaultSimState';

  %% Register methods
  block.RegBlockMethod('Outputs',                 @Output);  
  

function Output(block)

 low = 22;
 high = 25;
 %Añadir energía:
 reward = 0; %-0.8 * block.InputPort(1).Data(length(block.InputPort(1).Data)); 
 %disp(length(block.InputPort(1).Data))
 for i=1:1
     %disp(i)
     aux = block.InputPort(1).Data(i);
     %disp(aux)
     if (aux < low || aux > high) && (abs(aux - low) >= 1 && abs(aux-high) >= 1)
        %disp(abs(aux - low))
        reward = reward - min( abs(aux - low), abs(aux - high));
        %reward = reward + 0;
        %disp('if13')
     end
     %disp('if1')
     if (aux >= low && aux <= high)
        reward = reward + 0.5;
     end

 end
  
 block.OutputPort(1).Data = reward;
  
%endfunction


