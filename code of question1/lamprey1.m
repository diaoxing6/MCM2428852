%%%
function dx=lamprey1(t,x)
    dx=zeros(3,1); 
    dx(1)=x(1)*(0.5-0.055*x(2));
    dx(2)=x(2)*(-0.3+0.030*x(1)-0.009*x(3));
    dx(3)=x(3)*(-0.1+0.008*x(2));
end