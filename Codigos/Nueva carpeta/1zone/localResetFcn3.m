function in = localResetFcn3(in)

%nz=size(params.Neigh,1);

ci = [10 + randi(30,1,1), 34 , .900*1e-3, 0.0013 , 0.0022];
ci = ci';
blk = 'hvac_1zone_TCH';
in = setVariable(in,'ci',ci,'Workspace',blk);