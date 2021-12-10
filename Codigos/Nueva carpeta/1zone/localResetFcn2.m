function in = localResetFcn2(in)

%nz=size(params.Neigh,1);

ci = [10 + randi(30,1,1), 22 + randi(6,1,1) , .900*1e-3];
ci = ci';
blk = 'hvac_1zone_v3';
in = setVariable(in,'ci',ci,'Workspace',blk);