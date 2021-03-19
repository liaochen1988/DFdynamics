data {
	int<lower=0> N;
	vector[N] dlogX;
	vector[N] growth_rate_Muribaculaceae;
	vector[N] inulin_response_Muribaculaceae;
	vector[N] pairwise_interaction_Muribaculaceae_Muribaculaceae;
	vector[N] pairwise_interaction_Muribaculaceae_Bacteroides_dash_acidifaciens;
	vector[N] pairwise_interaction_Muribaculaceae_Akkermansia_dash_muciniphila;
	vector[N] growth_rate_Bacteroides_dash_acidifaciens;
	vector[N] inulin_response_Bacteroides_dash_acidifaciens;
	vector[N] pairwise_interaction_Bacteroides_dash_acidifaciens_Muribaculaceae;
	vector[N] pairwise_interaction_Bacteroides_dash_acidifaciens_Bacteroides_dash_acidifaciens;
	vector[N] pairwise_interaction_Bacteroides_dash_acidifaciens_Akkermansia_dash_muciniphila;
	vector[N] growth_rate_Akkermansia_dash_muciniphila;
	vector[N] inulin_response_Akkermansia_dash_muciniphila;
	vector[N] pairwise_interaction_Akkermansia_dash_muciniphila_Muribaculaceae;
	vector[N] pairwise_interaction_Akkermansia_dash_muciniphila_Bacteroides_dash_acidifaciens;
	vector[N] pairwise_interaction_Akkermansia_dash_muciniphila_Akkermansia_dash_muciniphila;
}
parameters {
	real<lower=0,upper=1> sigma;
	real alpha__Muribaculaceae;
	real epsilon__Muribaculaceae;
	real beta__Muribaculaceae_Muribaculaceae;
	real beta__Muribaculaceae_Bacteroides_dash_acidifaciens;
	real beta__Muribaculaceae_Akkermansia_dash_muciniphila;
	real alpha__Bacteroides_dash_acidifaciens;
	real epsilon__Bacteroides_dash_acidifaciens;
	real beta__Bacteroides_dash_acidifaciens_Muribaculaceae;
	real beta__Bacteroides_dash_acidifaciens_Bacteroides_dash_acidifaciens;
	real beta__Bacteroides_dash_acidifaciens_Akkermansia_dash_muciniphila;
	real alpha__Akkermansia_dash_muciniphila;
	real epsilon__Akkermansia_dash_muciniphila;
	real beta__Akkermansia_dash_muciniphila_Muribaculaceae;
	real beta__Akkermansia_dash_muciniphila_Bacteroides_dash_acidifaciens;
	real beta__Akkermansia_dash_muciniphila_Akkermansia_dash_muciniphila;
}
model {
	sigma ~ uniform(0,1);
	alpha__Muribaculaceae ~ normal(0,1);
	epsilon__Muribaculaceae ~ normal(0,1);
	beta__Muribaculaceae_Muribaculaceae ~ normal(0,1);
	beta__Muribaculaceae_Bacteroides_dash_acidifaciens ~ normal(0,1);
	beta__Muribaculaceae_Akkermansia_dash_muciniphila ~ normal(0,1);
	alpha__Bacteroides_dash_acidifaciens ~ normal(0,1);
	epsilon__Bacteroides_dash_acidifaciens ~ normal(0,1);
	beta__Bacteroides_dash_acidifaciens_Muribaculaceae ~ normal(0,1);
	beta__Bacteroides_dash_acidifaciens_Bacteroides_dash_acidifaciens ~ normal(0,1);
	beta__Bacteroides_dash_acidifaciens_Akkermansia_dash_muciniphila ~ normal(0,1);
	alpha__Akkermansia_dash_muciniphila ~ normal(0,1);
	epsilon__Akkermansia_dash_muciniphila ~ normal(0,1);
	beta__Akkermansia_dash_muciniphila_Muribaculaceae ~ normal(0,1);
	beta__Akkermansia_dash_muciniphila_Bacteroides_dash_acidifaciens ~ normal(0,1);
	beta__Akkermansia_dash_muciniphila_Akkermansia_dash_muciniphila ~ normal(0,1);
	dlogX ~ normal(alpha__Muribaculaceae*growth_rate_Muribaculaceae+epsilon__Muribaculaceae*inulin_response_Muribaculaceae+beta__Muribaculaceae_Muribaculaceae*pairwise_interaction_Muribaculaceae_Muribaculaceae+beta__Muribaculaceae_Bacteroides_dash_acidifaciens*pairwise_interaction_Muribaculaceae_Bacteroides_dash_acidifaciens+beta__Muribaculaceae_Akkermansia_dash_muciniphila*pairwise_interaction_Muribaculaceae_Akkermansia_dash_muciniphila+alpha__Bacteroides_dash_acidifaciens*growth_rate_Bacteroides_dash_acidifaciens+epsilon__Bacteroides_dash_acidifaciens*inulin_response_Bacteroides_dash_acidifaciens+beta__Bacteroides_dash_acidifaciens_Muribaculaceae*pairwise_interaction_Bacteroides_dash_acidifaciens_Muribaculaceae+beta__Bacteroides_dash_acidifaciens_Bacteroides_dash_acidifaciens*pairwise_interaction_Bacteroides_dash_acidifaciens_Bacteroides_dash_acidifaciens+beta__Bacteroides_dash_acidifaciens_Akkermansia_dash_muciniphila*pairwise_interaction_Bacteroides_dash_acidifaciens_Akkermansia_dash_muciniphila+alpha__Akkermansia_dash_muciniphila*growth_rate_Akkermansia_dash_muciniphila+epsilon__Akkermansia_dash_muciniphila*inulin_response_Akkermansia_dash_muciniphila+beta__Akkermansia_dash_muciniphila_Muribaculaceae*pairwise_interaction_Akkermansia_dash_muciniphila_Muribaculaceae+beta__Akkermansia_dash_muciniphila_Bacteroides_dash_acidifaciens*pairwise_interaction_Akkermansia_dash_muciniphila_Bacteroides_dash_acidifaciens+beta__Akkermansia_dash_muciniphila_Akkermansia_dash_muciniphila*pairwise_interaction_Akkermansia_dash_muciniphila_Akkermansia_dash_muciniphila, sigma);
}