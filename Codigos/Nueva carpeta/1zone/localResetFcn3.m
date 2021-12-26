function in = localResetFcn3(in, block,T_s,H_sup,struct)

%nz=size(params.Neigh,1);
pick_day = randi([0 2],1,1);
T_supply = T_s(1 + pick_day*2881: 2881 + pick_day*2881);
H_supply = H_sup(1 + pick_day*2881: 2881 + pick_day*2881);
pick_hour = randi([1 720],1,1);
T_supply = [T_supply(pick_hour:2881), T_supply(1:pick_hour - 1)]';
H_supply = [H_supply(pick_hour:2881), H_supply(1:pick_hour - 1)]';
t = struct.time;
T_supply = [t, T_supply];
H_supply = [t, H_supply];
if pick_day == 0
    ci = [min(max(T_supply(1) + randi([-4 4],1,1),3),49), 16 + randi(6,1,1)  , .900*1e-3, 0.0020 , 0.35];
    ci = ci';
elseif pick_day == 1
    ci = [min(max(T_supply(1) + randi([-4 4],1,1),3),49), 14 + randi(6,1,1)  , .900*1e-3, 0.0020 , 0.35];
    ci = ci';
else
    ci = [min(max(T_supply(1) + randi([-4 4],1,1),3),49), 20 + randi(6,1,1)  , .900*1e-3, 0.0020 , 0.45];
    ci = ci';
end
blk = block;
in = setVariable(in,'ci',ci,'Workspace',blk);
in = setVariable(in,'T_supply',T_supply,'Workspace',blk);
in = setVariable(in,'H_supply',H_supply,'Workspace',blk);