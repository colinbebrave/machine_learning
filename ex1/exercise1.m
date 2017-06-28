
for iterations = 1:1500
    alpha = 0.01
    theta = zeros(2,1)
    suma=sum(X*theta-y)
    theta = theta-alpha*suma*X/m
    iterations = iterations + 1
    J = sum((suma^2)/(2m)
end

    