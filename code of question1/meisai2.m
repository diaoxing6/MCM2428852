clc
clear
[t1,x1]=ode45('lam_2',[0 40],[150 50 10]);
prey1=x1(:,1);lam1=x1(:,2);pred1=x1(:,3);
subplot(2,2,1)
plot(t1,prey1,'-r',t1,lam1,'-b',t1,pred1,'-g')
legend('Prey','Lamprey','Predator')
%title('Prey Lamprey Predator population dynamics plot-1')
xlabel('generation');   ylabel('Population')
%%
[t2,x2]=ode45('lam_1',[0 40],[150 50 10]);
prey2=x2(:,1);lam2=x2(:,2);pred2=x2(:,3);
subplot(2,2,2)
plot(t2,prey2,'-r',t2,lam2,'-b',t2,pred2,'-g')
legend('Prey','Lamprey','Predator')
%title('Prey Lamprey Predator population dynamics plot-1')
xlabel('generation');   ylabel('Population')
%%
[t3,x3]=ode45('lam1',[0 40],[150 50 10]);
prey3=x3(:,1);lam3=x3(:,2);pred3=x3(:,3);
subplot(2,2,3)
plot(t3,prey3,'-r',t3,lam3,'-b',t3,pred3,'-g')
legend('Prey','Lamprey','Predator')
%title('Prey Lamprey Predator population dynamics plot-3')
xlabel('generation');   ylabel('Population')
%%
[t4,x4]=ode45('lam2',[0 40],[150 50 10]);
prey4=x4(:,1);lam4=x4(:,2);pred4=x4(:,3);
subplot(2,2,4)
plot(t4,prey4,'-r',t4,lam4,'-b',t4,pred4,'-g')
legend('Prey','Lamprey','Predator')
%title('Prey Lamprey Predator population dynamics plot-4')
xlabel('generation');   ylabel('Population')
