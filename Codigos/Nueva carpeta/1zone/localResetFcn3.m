function in = localResetFcn3(in)

%nz=size(params.Neigh,1);

ci = [10 + randi(12,1,1), 18 + randi(9,1,1)  , .900*1e-3, 0.0013 , 0.5];
ci = ci';
blk = 'hvac_1zone_TCH';
in = setVariable(in,'ci',ci,'Workspace',blk);