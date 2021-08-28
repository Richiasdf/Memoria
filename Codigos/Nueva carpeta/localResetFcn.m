function in = localResetFcn(in)

%nz=size(params.Neigh,1);

%sizes = simsizes;
%sizes.NumContStates  = 2*nz;
%sizes.NumDiscStates  = 0;
%sizes.NumOutputs     = 2*nz;
%sizes.NumInputs      = 2+nz+1+nz;
%sizes.DirFeedthrough = 0;
%sizes.NumSampleTimes = 1;

%sys = simsizes(sizes);
%str = [];
%x0  = ic;
%ts  = [0 0];   % sample time: [period, offset]

%in = setVariable(in,'flag', 0);

% speicfy that the simState for this s-function is same as the default
blk = 'hvac_maborrelli_singapur_nzones_with_MPC_v3_manualdist_3/System';
%blk = 'rlwatertank/Water-Tank System/H';
%in = set_param(blk,'Parameters','ci,paramsnz,1');
in = setBlockParameter(in,blk,'Parameters','ci,paramsnz,1');