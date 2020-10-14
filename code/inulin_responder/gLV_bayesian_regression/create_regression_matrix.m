function [reformed_X, reformed_Y] = create_regression_matrix(topX_bacteria)

load('processed_data.mat');

%% select the most abundant bacteria
var_bacteria = normalized_data_w_pseudocount.Properties.VariableNames(7:6+topX_bacteria);
var_scfa = cell({'Acetate';'Butyrate';'Propionate'});
var_scfa_bacteria = {var_scfa{:} var_bacteria{:}};

%% create raw Y matrix: [n time points] x [scfa + bacteria]
% each time point represent each day of each mice
raw_Y = log_deriv_normalized_data_w_pseudocount{:,var_scfa_bacteria};

%% create raw X matrix: [n time points] x [1,0/1,scfa,bacteria]
% the first 1 represents growth rate
% the second 0/1 represents diet effect: 0 or cellulose, 1 for inulin
raw_X = zeros(height(normalized_data_w_pseudocount), 2+length(var_scfa_bacteria));
raw_X(:,1) = 1;
raw_X(:,2) = normalized_data_w_pseudocount.Diet;
raw_X(:,3:end) = normalized_data_w_pseudocount{:,var_scfa_bacteria};

%% reform Y matrix
reformed_Y = raw_Y(:);

%% reform X matrix
dimX1 = size(raw_X,1);
dimX2 = size(raw_X,2);
dimY2 = size(raw_Y,2);
reformed_X = zeros(dimX1*dimY2, dimX2*dimY2);
for i=1:dimY2
    reformed_X(1+(i-1)*dimX1:i*dimX1,1+(i-1)*dimX2:i*dimX2) = raw_X;
end

end

