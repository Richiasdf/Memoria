function in = localResetFcn(in)

%nz=size(params.Neigh,1);

disp('reset')
ci = 20 + randi(20,1,2);
ci = ci';
blk = 'hvac_1zone_v3';
in = setVariable(in,'ci',ci,'Workspace',blk);