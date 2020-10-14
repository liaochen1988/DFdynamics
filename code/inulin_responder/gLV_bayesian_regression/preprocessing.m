clear all;
clc;

%% Read file
original_data = readtable('../DaiLab_data/SCFA_16S_combined.csv');
original_data.Var1 = [];

%% normalize bacteria and SCFA seperately
var_scfa = cell({'Acetate';'Butyrate';'Propionate'});
var_bacteria = original_data.Properties.VariableNames(7:end);
normalized_data_w_pseudocount = original_data;
normalized_data_w_pseudocount{:,var_scfa} = normalized_data_w_pseudocount{:,var_scfa}/max(normalized_data_w_pseudocount{:,var_scfa}(:));
normalized_data_w_pseudocount{:,var_bacteria} = normalized_data_w_pseudocount{:,var_bacteria}/max(normalized_data_w_pseudocount{:,var_bacteria}(:));

%% add pseudocount (minimum of all non-zero values)
for i=1:height(normalized_data_w_pseudocount)
    sample = normalized_data_w_pseudocount{i,var_bacteria};
    pseudocount = min(sample(sample > 0));
    sample(sample==0) = pseudocount;
    normalized_data_w_pseudocount{i,var_bacteria} = sample;
end

%% calculate log-derivative
log_normalized_data_w_pseudocount = normalized_data_w_pseudocount;
log_normalized_data_w_pseudocount{:,var_bacteria} = log(normalized_data_w_pseudocount{:,var_bacteria});

all_mice_ID = unique(original_data.Mice_ID);
log_deriv_normalized_data_w_pseudocount = log_normalized_data_w_pseudocount;
var_scfa = cell({'Acetate';'Butyrate';'Propionate'});
var_scfa_bacteria = {var_scfa{:} var_bacteria{:}};
for i=1:length(all_mice_ID)
    curr_mice = all_mice_ID{i};
    curr_logic = strcmp(log_normalized_data_w_pseudocount.Mice_ID,curr_mice);
    curr_log_data = log_normalized_data_w_pseudocount(curr_logic,:);
    for j=1:length(var_scfa_bacteria)
        curr_var = var_scfa_bacteria{j};
        xdata = curr_log_data{:,'Day'};
        ydata = curr_log_data{:,curr_var};
        
        % find first-order derivative
        pp          = spline(xdata, ydata);
        pder        = fnder(pp, 1);
        log_deriv_normalized_data_w_pseudocount{curr_logic,curr_var} = ppval(pder, xdata);
    end
end

save('processed_data.mat','original_data','normalized_data_w_pseudocount','log_deriv_normalized_data_w_pseudocount');