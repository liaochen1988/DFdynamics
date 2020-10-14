function [lb, ub] = create_constraints(fn_evidence)

% fn_evidence is filename for acetate, butyrate, propionate producers +
% inulin degraders

% coefficients are ordered in the following way:
% for SCFA:       0,        0,      0,      0,      0,      whether a given SCFA can be produced/uptaken by bacteria
% for Bacteria:   growth,   >0,  positive if the bacteria can use the SCFA,
% otherwise negative, bacteria-bacteria interactions


end

