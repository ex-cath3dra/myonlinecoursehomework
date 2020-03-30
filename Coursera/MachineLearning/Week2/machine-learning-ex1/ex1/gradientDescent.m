function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %

n=length(theta);
for i=1:m
  h(i,:)=theta'*X(i,:)';   %computes h vector
end
for j=1:n
    theta(j,:)=theta(j,:)-alpha*(sum((h.-y)'*X(:,j))/m);
end

  
% disp's used for debugging:
  % J(iter,:)=computeCost(X,y,theta); %alternative J (for no reason)
  % disp(['For iteration ' num2str(iter) ':'])
  % disp(['Cost is: J=' num2str(J(iter,:))])
  % disp('theta is:')
  % disp(theta)


    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);

end

end
