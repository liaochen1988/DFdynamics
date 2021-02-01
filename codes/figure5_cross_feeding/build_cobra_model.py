import pandas as pd
import cobra
import warnings
from termcolor import colored
from copy import copy,deepcopy
import re
from optlang.symbolics import Zero
from sympy import Add
import pandas as pd
from utils import use_diet, add_enzymatic_constraints
import os
import numpy as np

def build_community_model(taxonomy, # a dict with agora model as key and relative abundance as value
                          AGORA_path, # directory where AGORA models are stored
                          model_id='com', # identifier of the model
                          medium=None, # culture medium that represents diet flux
                          enzyme_cost_profile=None, # if true, total enzymatic fluxes are limited
                          steady_com=False, # if true, growth rates of all species are constrained to be equal
                          biomass_coupling=False, # if true, couple any flux with biomass
                          biomass_coupling_coefficient=400, # only useful when biomass_coupling is true
                          solver='cplex' # we only allow cplex or gurobi
                         ):

    # create a new community model
    com = cobra.Model(model_id)
    com.species = []
    com.solver =solver
    if solver == 'cplex':
        com.solver.configuration.lp_method = "barrier"
        com.solver.configuration.qp_method = "barrier"

    # define community-level growth rate
    if steady_com == True:
    	obj = com.problem.Variable('community_growth', lb=0, ub=1000)
    else:
    	obj = Zero

	# add each community component to the community model
    print("building new community model {}.".format(model_id))
    for model_file, genus_fraction in taxonomy.items():
        model_name = model_file.replace('.xml','')
        genus_taxid = int(model_file.replace('.xml','').split('_')[1])
        if model_file not in os.listdir(AGORA_path):
            print('model file %s cannot found.' % (model_name))
            raise

        # add to species list
        com.species.append(model_name)
        print('adding reactions for %s (fraction = %2.3f)'%(model_name, genus_fraction))

        # read Cobra model
        model = cobra.io.read_sbml_model(AGORA_path + model_name + '.xml', f_replace={})
        model.solver = 'cplex'

        # append organism name to reactions of this organism
        suffix = "__" + model_name.replace(" ", "_").strip()
        #print(suffix)
        for m in model.metabolites:
            m.global_id = re.sub("__\\d+__", "_", m.id).strip(" _-")
            m.id = m.global_id + suffix
            m.compartment += suffix
            m.community_id = model_name
        for r in model.reactions:
            r.global_id = re.sub("__\\d+__", "_", r.id).strip(" _-")
            r.id = r.global_id + suffix
            r.community_id = model_name
        com.add_reactions(model.reactions)

        # identify biomass reaction and make it the objective
        biomrxns = [rxn.id for rxn in model.reactions if rxn.id.startswith(('R_biomass','biomass'))]
        #print(biomrxns)
        assert len(biomrxns)==1
        model.objective = model.reactions.get_by_id(biomrxns[0]).flux_expression

        # add coupling between biomass flux and the community growth rate: V_biomass^k = X^k * mu
        o = com.solver.interface.Objective.clone(model.objective, model=com.solver)
        V_bm = o.expression
        if steady_com==True:
            community_coupling = model.problem.Constraint((V_bm - genus_fraction*obj).expand(), name='community_coupling'+suffix, lb=0,ub=0)
            com.add_cons_vars([community_coupling])
            com.solver.update()
        else:
            obj += V_bm

        # convert all reaction bounds (including exchange reactions) to be abundance-scaled
        # V_j^k >= LB_j^k * X^k
        # V_j^k <= UB_j^k * X^k
        for reaction in com.reactions:
            if reaction.id.endswith(suffix):
                reaction.lower_bound = genus_fraction*reaction.lower_bound
                reaction.upper_bound = genus_fraction*reaction.upper_bound

        # add coupling between flux and biomass (the goal is to avoid situations when  at no biomass production)
        # |V_j^k| <= biomass_coupling_coefficient * V_biomass^k
        if biomass_coupling==True:
            for r in com.reactions:
                if r.community_id == model_name:
                    if r.lower_bound==0.0 and r.upper_bound==0.0:
                        continue
                    if r.lower_bound >= 0.0:
                        biomass_coupling_f = model.problem.Constraint((biomass_coupling_coefficient * V_bm - r.flux_expression).expand(), name='forward_biomass_coupling__'+r.id+suffix, lb=0)
                        com.add_cons_vars([biomass_coupling_f])
                    elif r.upper_bound <= 0.0:
                        biomass_coupling_r = model.problem.Constraint((biomass_coupling_coefficient * V_bm + r.flux_expression).expand(), name='reverse_biomass_coupling__'+r.id+suffix, lb=0)
                        com.add_cons_vars([biomass_coupling_r])
                    else:
                        biomass_coupling_f = model.problem.Constraint((biomass_coupling_coefficient * V_bm - r.flux_expression).expand(), name='forward_biomass_coupling__'+r.id+suffix, lb=0)
                        biomass_coupling_r = model.problem.Constraint((biomass_coupling_coefficient * V_bm + r.flux_expression).expand(), name='reverse_biomass_coupling__'+r.id+suffix, lb=0)
                        com.add_cons_vars([biomass_coupling_f, biomass_coupling_r])
                    com.solver.update()

        # add enzyme constraints
        if enzyme_cost_profile is not None:
            enzyme_cost_profile_tax = deepcopy(enzyme_cost_profile[enzyme_cost_profile.NCBIID==genus_taxid])
            enzyme_cost_profile_tax.drop('NCBIID',axis=1,inplace=True)
            enzyme_cost_profile_tax.set_index('EC',inplace=True,drop=True)
            com = add_enzymatic_constraints(com, tblenzymecost=enzyme_cost_profile_tax, reaction_suffix=suffix, suffix_weight=genus_fraction)

        # A list of sub-strings in reaction IDs that usually indicate that the reaction is *not* an exchange reaction."""
        exclude = ["biosynthesis", "transcription", "replication", "sink", "demand", "DM_", "SN_", "SK_"]

        for r in model.reactions:
            # Some sanity checks for whether the reaction is an exchange
            ex = "e" + "__" + r.community_id
            if (not r.boundary or any(bad in r.id for bad in exclude)): # or ex not in r.compartments
                continue
            if not r.id.lower().startswith(("r_ex","ex")):
                raise RuntimeError("Reaction %s seems to be an exchange " % r.id + "reaction but its ID does not start with 'EX_'...")

            # switch the lower and upper bounds if the exchange reaction is written in form of [ <=> reactantant ]
            if len(r.reactants) == 0:
                r.add_metabolites({r.products[0]: -2})
                lb = r.lower_bound
                ub = r.upper_bound
                r.lower_bound = -ub
                r.upper_bound = -lb

            lb = r.lower_bound
            ub = r.upper_bound

            if lb < 0.0 and lb > -1e-6:
                warnings.warn("lower bound for %r below numerical accuracy -> adjusting to stabilize model.")
                lb = -1e-6
            if ub > 0.0 and ub < 1e-6:
                warnings.warn("upper bound for %r below numerical accuracy -> adjusting to stabilize model.")
                ub = 1e-6

            met = r.reactants[0]
            medium_id = re.sub("_{}$".format(met.compartment.replace('[','\[').replace(']','\]')), "", met.id)
            #print(met.id, met.compartment.replace('[','\[').replace(']','\]'), medium_id)
            if medium_id in exclude:
                continue
            medium_id += "_m"
            if medium_id not in com.metabolites:
                # If metabolite does not exist in medium add it to the model and also add an exchange reaction for the medium
                medium_met = met.copy()
                medium_met.id = medium_id
                medium_met.compartment = "m"
                medium_met.global_id = medium_id
                medium_met.community_id = "medium"

                # add reaction: Medium_Met <=>
                ex_medium = cobra.Reaction(
                    id="EX_" + medium_met.id,
                    name=medium_met.id + " medium exchange",
                    lower_bound=lb,
                    upper_bound=ub)
                ex_medium.add_metabolites({medium_met: -1})
                ex_medium.global_id = ex_medium.id
                ex_medium.community_id = "medium"
                com.add_reactions([ex_medium])
            else:
                medium_met = com.metabolites.get_by_id(medium_id)
                ex_medium = com.reactions.get_by_id("EX_" + medium_met.id)
                ex_medium.lower_bound = min(lb, ex_medium.lower_bound)
                ex_medium.upper_bound = max(ub, ex_medium.upper_bound)

            # add right-hand-side metabolite to each exchange reaction of single organism
            # e.g., convert Met_id_c <=> to Met_id_c <=> Met_id_m
            r.add_metabolites({medium_met: 1})
        com.solver.update()  # to avoid dangling refs due to lazy add

    # define objective
    com.objective = com.problem.Objective(obj, direction="max")
    com.solver.update()

    # apply medium
    if medium is not None:
        com = use_diet(model=com, medium=medium, model_type='community')

    return com
