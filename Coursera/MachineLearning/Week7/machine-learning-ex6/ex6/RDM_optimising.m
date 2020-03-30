range=[0.01,0.03,0.1,0.3,1,3,10,30];

% SVM Parameters
C = 1;
sigma = 0.1;

for i=1:length(range)
  C=range(i);
  for j=1:length(range)
    sigma=range(j);
% We set the tolerance and max_passes lower here so that the code will run
% faster. However, in practice, you will want to run the training to
% convergence.
model= svmTrain(X, y, C, @(x1, x2) gaussianKernel(x1, x2, sigma));

predictions = svmPredict(model, Xval);
E(i,j)=mean(double(predictions ~= yval));
end

end
