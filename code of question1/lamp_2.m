function dx=lamp_2(t,x)
    dx=zeros(3,1); 
    dx(1)=x(1)*(0.5-0.035*x(2));
    dx(2)=x(2)*(-0.3+0.020*x(1)-0.005*x(3));
    dx(3)=x(3)*(-0.1+0.01*x(2));
end