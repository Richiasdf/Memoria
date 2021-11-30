function in = localResetFcn(in)

%nz=size(params.Neigh,1);

ci = [25 + randi(15,1,1), 34 ];
ci = ci';
blk = 'hvac_1zone_v3';
in = setVariable(in,'ci',ci,'Workspace',blk);