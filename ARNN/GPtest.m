clc; clear
close all
n0 = 1;
n1 = 1;
l = 20;
Nmax = 1e3;
Nsession = 1;
y = nan(Nmax,Nsession);
rng(0);

for i=1:Nsession
  y(:,i)=RSGPgenerator(Nmax,n0,n1,1,l,.1,1,0,0);
end

figure;
plot(y,'.');