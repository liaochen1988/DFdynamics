import pandas as pd
from copy import copy, deepcopy
from os.path import isfile
from cobra import Reaction, Model, Metabolite
from cobra.io import read_sbml_model
from optlang.symbolics import Zero
import re
import numpy as np

def build_community_model(
    taxonomy,   # dataframe, columns = [model name, biomass]
    AGORA_model_dir, # directory of AGORA models
    community_id='cmm',
    steady_com=False, # if true, growth rates of all species are constrained to be equal
    biomass_coupling=False, # if true, couple any flux with biomass
    biomass_coupling_coefficient=400, # only useful when biomass_coupling is true
    solver='cplex'):

    # create a new model
    comm = Model(community_id)
    comm.species = []

    # adjust the optlang solver configuration for larger problems
    if solver=='cplex':
        comm.solver = 'cplex'
        comm.solver.configuration.lp_method = "barrier"
        comm.solver.configuration.qp_method = "barrier"
        comm.solver.problem.parameters.threads.set(1)
        comm.solver.problem.parameters.barrier.convergetol.set(1e-9)
        comm.solver.configuration.tolerances.integrality = 1e-9
        comm.solver.problem.parameters.mip.tolerances.integrality.set(1e-9)
    elif solver == "gurobi":
        comm.solver = 'gurobi'
        comm.solver.configuration.lp_method = "barrier"
        comm.solver.problem.Params.BarConvTol = 1e-9
        comm.solver.problem.Params.BarIterLimit = 1001
    else:
        print('solver must be cplex or gurobi.')
        raise

    # define objective
    if steady_com==True:
        obj = comm.problem.Variable('community_growth_rate', lb=0, ub=1000)
    else:
        obj = Zero

    # add each species to the community
    for index in taxonomy.index:
        # read taxonomy information
        model_name = taxonomy.loc[index, 'model_name']
        biomass = taxonomy.loc[index, 'biomass']
        relative_abundance = biomass/sum(list(taxonomy['biomass']))

        # read model
        model = read_sbml_model(AGORA_model_dir+model_name+'.xml')
        model.solver = solver

        # check if model_name is unique
        if len(taxonomy[taxonomy.model_name==model_name].index)>1:
            model_name = model_name + '__' + str(index)

        # suffix is used to reactions from differentiate organisms
        suffix = "__" + model_name.replace(" ", "_").strip()
        for m in model.metabolites:
            m.global_id = re.sub("__\\d+__", "_", m.id).strip(" _-")
            m.id = m.global_id + suffix
            m.compartment = '['+m.compartment+']'+suffix       # change compartment from 'c' (for example) to 'c + suffix'
            m.community_id = model_name
        for r in model.reactions:
            r.global_id = re.sub("__\\d__", "_", r.id).strip(" _-")
            r.id = r.global_id + suffix
            r.community_id = model_name
        comm.add_reactions(model.reactions)
        comm.species.append(model_name)

        o = comm.solver.interface.Objective.clone(model.objective, model=comm.solver)
        if steady_com==True:
            # couple flux through biomass with the community growth rate: V_biomass^k = X^k * mu
            community_coupling = model.problem.Constraint((o.expression - obj * relative_abundance).expand(), name='community_coupling'+suffix, lb=0,ub=0)
            comm.add_cons_vars([community_coupling])
            comm.solver.update()

            # convert all reaction bounds (including exchange reactions) to be relative_abundance-scaled
            # V_j^k >= LB_j^k * X^k
            # V_j^k <= UB_j^k * X^k
            for reaction in comm.reactions:
                if reaction.id.endswith(suffix):
                    reaction.lower_bound = reaction.lower_bound * relative_abundance
                    reaction.upper_bound = reaction.upper_bound * relative_abundance
        else:
            obj += o.expression * relative_abundance

        # add coupling between flux and bioimass
        # the goal is to avoid situations when  at no biomass production
        # |V_j^k| <= biomass_coupling_coefficient * V_biomass^k
        if biomass_coupling==True:
            for r in comm.reactions:
                if r.community_id == model_name:
                    if r.lower_bound==0.0 and r.upper_bound==0.0:
                        continue
                    if r.lower_bound >= 0.0:
                        biomass_coupling_f = model.problem.Constraint((biomass_coupling_coefficient * o.expression - r.flux_expression).expand(), name='forward_biomass_coupling__'+r.id+suffix, lb=0)
                        comm.add_cons_vars([biomass_coupling_f])
                    elif r.upper_bound <= 0.0:
                        biomass_coupling_r = model.problem.Constraint((biomass_coupling_coefficient * o.expression + r.flux_expression).expand(), name='reverse_biomass_coupling__'+r.id+suffix, lb=0)
                        comm.add_cons_vars([biomass_coupling_r])
                    else:
                        biomass_coupling_f = model.problem.Constraint((biomass_coupling_coefficient * o.expression - r.flux_expression).expand(), name='forward_biomass_coupling__'+r.id+suffix, lb=0)
                        biomass_coupling_r = model.problem.Constraint((biomass_coupling_coefficient * o.expression + r.flux_expression).expand(), name='reverse_biomass_coupling__'+r.id+suffix, lb=0)
                        comm.add_cons_vars([biomass_coupling_f, biomass_coupling_r])
                    comm.solver.update()

        # A list of sub-strings in reaction IDs that usually indicate that
        # the reaction is *not* an exchange reaction."""
        exclude = ["biosynthesis", "transcription", "replication", "sink", "demand", "DM_", "SN_", "SK_"]

        for r in model.reactions:
            # Some sanity checks for whether the reaction is an exchange
            ex = "e" + "__" + r.community_id
            if (not r.boundary or any(bad in r.id for bad in exclude)): # or ex not in r.compartments
                continue
            if not r.id.lower().startswith("ex"):
                print("Reaction %s seems to be an exchange " % r.id +
                      "reaction but its ID does not start with 'EX_'...")
                assert False

            # Make sure that exchange reactions only have one reactant or product
            assert len(r.reactants)+len(r.products) == 1

            # switch the lower and upper bounds if the exchange reaction
            # is written in form of [ <=> reactantant ]
            if len(r.reactants) == 0:
                r.add_metabolites({r.products[0]: -2})
                lb = r.lower_bound
                ub = r.upper_bound
                r.lower_bound = -ub
                r.upper_bound = -lb

            assert len(r.reactants) == 1
            assert len(r.products)  == 0

            lb = r.lower_bound
            ub = r.upper_bound

            if lb < 0.0 and lb > -1e-6:
                print("lower bound for %r below numerical accuracy -> adjusting to stabilize model.")
                lb = -1e-6
            if ub > 0.0 and ub < 1e-6:
                print("upper bound for %r below numerical accuracy -> adjusting to stabilize model.")
                ub = 1e-6

            # met.id:  12dgr180_e__Clostridium_saccharoperbutylacetonicum
            # medium_id: 12dgr180_m
            # met = (r.reactants + r.products)[0]
            met = r.reactants[0]

            # re.sub(pattern, repl, string, count=0, flags=0), $ match the end of a string
            # note that
            #medium_id = re.sub("_{}$".format(met.compartment), "", met.id)
            medium_id = met.id.replace(met.compartment,'')
            if medium_id in exclude:
                continue
            medium_id += "_m"

            if medium_id not in comm.metabolites:
                # If metabolite does not exist in medium add it to the model
                # and also add an exchange reaction for the medium
                #print("adding metabolite %s to external medium" % medium_id)
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
                comm.add_reactions([ex_medium])
            else:
                #print("updating import rate for external metabolite %s" %  medium_id)
                medium_met = comm.metabolites.get_by_id(medium_id)
                ex_medium = comm.reactions.get_by_id("EX_" + medium_met.id)
                ex_medium.lower_bound = min(lb, ex_medium.lower_bound)
                ex_medium.upper_bound = max(ub, ex_medium.upper_bound)

            # add right-hand-side metabolite to each exchange reaction of single organism
            # e.g., convert Met_id_c <=> to Met_id_c <=> Met_id_m
            r.add_metabolites({medium_met: 1})

        comm.solver.update()  # to avoid dangling refs due to lazy add

    # update objective function
    comm.objective = comm.problem.Objective(obj, direction="max")

    return comm

# set culture medium
def set_culture_medium(model, medium):
    for rxn in model.exchanges:
        rxn.lower_bound = 0
    for nutr, flux in medium.items():
        if nutr in model.reactions:
            model.reactions.get_by_id(nutr).lower_bound = flux
    return model

# Pareto front analysis of for 2-species model
def pareto_front_2species(model,
                          npts=50 # number of points between 0 and max growth rate
                          ):
    # this algorithm only allows 2-species community
    assert len(model.species) == 2

    # get biomass reaction of each species
    biom_rid = {}
    for sp in model.species:
        biom_rid[sp] = [r.id for r in model.reactions if r.id.startswith('biomass') and r.id.endswith(sp)][0]

    # fix growth rate of the current species and maximize the other
    pareto_front = []
    for sp_to_fix in model.species:
        sp_to_maximize = [sp for sp in model.species if sp != sp_to_fix][0]

        # fix growth rate of the current species and maximize the other
        model.objective = model.reactions.get_by_id(biom_rid[sp_to_fix]).flux_expression
        max_gr = model.slim_optimize()
        xdata = np.linspace(0,max_gr,npts)

        model.objective = model.reactions.get_by_id(biom_rid[sp_to_maximize]).flux_expression
        ydata=[]
        for x in xdata:
            model.reactions.get_by_id(biom_rid[sp_to_fix]).lower_bound=x
            model.reactions.get_by_id(biom_rid[sp_to_fix]).upper_bound=x
            ydata.append(model.slim_optimize())

        # reset bounds of biomass reaction
        model.reactions.get_by_id(biom_rid[sp_to_fix]).lower_bound = 0
        model.reactions.get_by_id(biom_rid[sp_to_fix]).upper_bound = 1000

        pareto_front.append([sp_to_fix,sp_to_maximize,list(xdata),list(ydata)])

    return pareto_front

