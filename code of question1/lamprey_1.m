%
function dx=lamprey_1(t,x)
    dx=zeros(3,1); 
    dx(1)=x(1)*(0.5-0.030*x(2));
    dx(2)=x(2)*(-0.3+0.015*x(1)-0.011*x(3));
    dx(3)=x(3)*(-0.1+0.012*x(2));
end