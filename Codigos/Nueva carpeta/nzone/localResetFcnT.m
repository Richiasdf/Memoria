function in = localResetFcnT(in, block,T_s,H_sup,struct)

%nz=size(params.Neigh,1);
pick_day = randi([0 2],1,1);
T_supply = T_s(1 + pick_day*2881: 2881 + pick_day*2881);
%H_supply = H_sup(1 + pick_day*2881: 2881 + pick_day*2881);
%pick_hour = randi([1 720],1,1);
%T_supply = [T_supply(pick_hour:2881), T_supply(1:pick_hour - 1)]';
%H_supply = [H_supply(pick_hour:2881), H_supply(1:pick_hour - 1)]';
t = struct.time;
T_supply = [t, T_supply'];
%H_supply = [t, H_supply'];
%ci = [0, 0 ,0 ,0 ,0];
if pick_day == 0
    t1 = min(max(T_supply(1,2) + randi([-1 10],1,1),3),49);
    t2 = 16 + randi(6,1,1);
    ci = [t1, t1, t1, t1, t2, t2, t2, t2];
elseif pick_day == 1
    t1 = min(max(T_supply(1,2) + randi([-1 10],1,1),3),49);
    t2 = 14 + randi(6,1,1);
    ci = [t1, t1, t1, t1, t2, t2, t2, t2];
else
    t1 = min(max(T_supply(1,2) + randi([-4 4],1,1),3),49);
    t2 = 16 + randi(6,1,1);
    ci = [t1, t1, t1, t1, t2, t2, t2, t2];
end
blk = block;
ci = ci';
in = setVariable(in,'ci',ci,'Workspace',blk);
in = setVariable(in,'T_supply',T_supply,'Workspace',blk);