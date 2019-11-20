files = dir('data/');
files = files(3:end);

% discrete timesteps
dt = 0.01;

% u parameters
m = 17;
L = 0.1;
I = 20;
b = 0.1;

u = [-1/m, -1/m, 0, 0, 1/m, 1/m, 0, 0;
    0, 0, -1/m -1/m, 0, 0, 1/m, 1/m;
    -L/(2*I), L/(2*I), -b/(2*I), b/(2*I), -L/(2*I), L/(2*I), -b/(2*I), b/(2*I)];

B = @(theta) rotz(rad2deg(theta))';

% first block matrix
A1 = zeros(6);
A1(1:3, 4:6) = eye(3);

for k=1:length(files)
    data = load(join(['C:\Users\conno\Dropbox\caltech\g2\fall\acm154\project\data\', files(k).name]));
    data = get(data.statedata, 'Data');
    
    num = files(k).name(2:end-4);
    
    F = zeros(8,1);
    for j=1:length(num)
        F(str2num(num(j))) = 1;
    end
    
    phi = zeros(length(data), 3)
    for i=2:length(data)
        
        theta_t1 = data(i-1,3);
        v_t1 = data(i-1,1:6)';
        
        v_t2 = data(i,1:6)';
        
        % Last block matrix
        A3 = zeros(6,3);
        A3(4:6,:) = B(theta_t1);
        
        % discretized dynamics
        phi_mat = (v_t2 - (eye(6) + A1*dt)*v_t1 - dt*A3*u*F)/dt;
        phi(i,:) = phi_mat(4:6);
        
        ans = v_t2;
        ans(4:6)'*0.1
        phi(i,:)
        
    end
    data = horzcat(data, phi);
    %csvwrite(join(['csv/', files(k).name(1:end-3), 'csv']), data);
end
