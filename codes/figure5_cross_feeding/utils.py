import pandas as pd
from copy import deepcopy
from optlang.symbolics import add, mul, One, Zero
from termcolor import colored
from cobra.flux_analysis.loopless import loopless_fva_iter
from numpy import zeros
from cobra.util import solver as sutil
import timeit
import json
import numpy as np
#from sympy import Add
import warnings
import sys

def set_cplex_mip_parameters(model,
                             advance=None,
                             barrier_algorithm=None,
                             threads=None,
                             tolint=None,
                             emphasis=None,
                             submipnodelim=None,
                             sollim=None,
                             probe=None,
                             rinsheur=None,
                             nodeselect=None,
                             heuristicfreq=None,
                             covers=None,
                             cliques=None,
                             disjunctive=None,
                             liftproj=None,
                             localimplied=None,
                             tolmipgap=None,
                             polishafter_mipgap=None,
                             symmetry=None,
                             repeatpresolve=None,
                             zerohalfcut=None,
                             gomory=None
                            ):
    if advance is not None:
        model.solver.problem.parameters.advance.set(advance) # 2
    if barrier_algorithm is not None:
        model.solver.problem.parameters.barrier.algorithm.set(barrier_algorithm) # 3
    if threads is not None:
        model.solver.problem.parameters.threads.set(threads) # 1
    if tolint is not None:
        model.solver.problem.parameters.mip.tolerances.integrality.set(tolint) # 1e-6
    if emphasis is not None:
        model.solver.problem.parameters.emphasis.mip.set(emphasis) # 1
    if submipnodelim is not None:
        model.solver.problem.parameters.mip.limits.submipnodelim.set(submipnodelim) # 10,000
    if sollim is not None:
        model.solver.problem.parameters.mip.limits.solutions.set(sollim) # 1
    if probe is not None:
        model.solver.problem.parameters.mip.strategy.probe.set(probe) # 3
    if rinsheur is not None:
        model.solver.problem.parameters.mip.strategy.rinsheur.set(rinsheur) # 20
    if nodeselect is not None:
        model.solver.problem.parameters.mip.strategy.nodeselect.set(nodeselect) # 2
    if heuristicfreq is not None:
        model.solver.problem.parameters.mip.strategy.heuristicfreq.set(heuristicfreq) # 10
    if covers is not None:
        model.solver.problem.parameters.mip.cuts.covers.set(covers) # 3
    if cliques is not None:
        model.solver.problem.parameters.mip.cuts.cliques.set(cliques) # 3
    if disjunctive is not None:
        model.solver.problem.parameters.mip.cuts.disjunctive.set(disjunctive) # 3
    if liftproj is not None:
        model.solver.problem.parameters.mip.cuts.liftproj.set(liftproj) # 3
    if localimplied is not None:
        model.solver.problem.parameters.mip.cuts.localimplied.set(localimplied) # 3
    if tolmipgap is not None:
        model.solver.problem.parameters.mip.tolerances.mipgap.set(tolmipgap) # 0.05
    if polishafter_mipgap is not None:
        model.solver.problem.parameters.mip.polishafter.mipgap.set(polishafter_mipgap) # 0.05
    if symmetry is not None:
        model.solver.problem.parameters.preprocessing.symmetry.set(symmetry) # 5
    if repeatpresolve is not None:
        model.solver.problem.parameters.preprocessing.repeatpresolve.set(repeatpresolve) # 3
    if zerohalfcut is not None:
        model.solver.problem.parameters.mip.cuts.zerohalfcut.set(zerohalfcut) # 2
    if gomory is not None:
        model.solver.problem.parameters.mip.cuts.gomory.set(gomory) # 2

    return model

def print_cplex_solver_status(model):
    print('optlang solver status: %s.' % (str(model.solver.status)))
    print('cplex solver status (%d): %s.'%(model.solver.problem.solution.get_status(), model.solver.problem.solution.get_status_string()))

# find relative abundance of AGORA models that are mapped from taxonomy
def find_relative_abundance_of_AGORA_models(taxonomy2agora, taxonomy_abundance, cutoff=0.0):
    # add relative abundance
    agora_relative_abundance = {}
    for tax, f in taxonomy_abundance.items():
        if tax in taxonomy2agora and str(taxonomy2agora[tax]) not in ['nan','']:
            agora_id = int(taxonomy2agora[tax]) # it should be an ncbi id
            if agora_id not in agora_relative_abundance:
                agora_relative_abundance[agora_id] = 0.0
            agora_relative_abundance[agora_id] += f

    # apply cutoff
    agora_relative_abundance = {k:v for k,v in agora_relative_abundance.items() if v>=cutoff}

    # renormalize
    agora_relative_abundance = {k:v/sum(agora_relative_abundance.values()) for k,v in agora_relative_abundance.items()}

    return agora_relative_abundance

# Returns metabolic fluxes left unconsumed in the lumen
def get_lumen_flux(fva_sol, medium):
    lumen_flux = deepcopy(fva_sol)
    # select only medium exchange reactions
    reactions_ex_m = [idx for idx in lumen_flux.index if idx.startswith('EX_') and idx.endswith('_m')]
    lumen_flux = lumen_flux.loc[reactions_ex_m]
    # calculate flux to lumen
    for idx in lumen_flux.index:
        met_id = idx[:-2].replace('EX_M_','EX_').split('EX_')[1]
        if met_id in medium.index:
            lumen_flux.loc[idx] = lumen_flux.loc[idx] - medium.loc[met_id].values[0]
    return lumen_flux

# Flux variability analysis
def flux_variability_analysis(model, reaction_list=None, loopless=False):
    if reaction_list is None:
        reaction_list = model.reactions
    else:
        reaction_list = model.reactions.get_by_any(reaction_list)

    prob = model.problem
    fva_results = pd.DataFrame({
        "minimum": zeros(len(reaction_list), dtype=float),
        "maximum": zeros(len(reaction_list), dtype=float)
    }, index=[r.id for r in reaction_list])
    with model:
        print(colored('performing flux variability analysis ...','red'))
        model.objective = Zero  # This will trigger the reset as well
        for what in ("minimum", "maximum"):
            sense = "min" if what == "minimum" else "max"
            model.solver.objective.direction = sense
            for rxn in reaction_list:
                # The previous objective assignment already triggers a reset
                # so directly update coefs here to not trigger redundant resets
                # in the history manager which can take longer than the actual
                model.solver.objective.set_linear_coefficients({rxn.forward_variable: 1, rxn.reverse_variable: -1})

                # set cplex parameters
                model = set_cplex_mip_parameters(model,
                                          emphasis=1,
                                          probe=3,
                                          rinsheur=20,
                                          nodeselect=2,
                                          heuristicfreq=20,
                                          tolmipgap=0.05
                                         )

                # optimization
                try:
                    solution = model.optimize()
                    if model.solver.problem.solution.get_status() not in [1, 101, 102]: # integer optimal
                        if model.solver.problem.solution.get_status() == 3: # infeasible
                            # model maybe numerically unstable (should be feasible)
                            print('solution is numerically unstable.')
                        elif model.solver.problem.solution.get_status() == 107: # time limit
                            # fail to achieve tolmipgap by time limit
                            current_mip_relative_gap = model.solver.problem.solution.MIP.get_mip_relative_gap()
                            print('in function flux_variability_analysis')
                            print('time limit exceeded: mip relative gap is %2.2f.'%(current_mip_relative_gap))
                        else:
                            print_cplex_solver_status(model)
                            print("unexpected error:", sys.exc_info()[0])
                            raise
                except:
                    print_cplex_solver_status(model)
                    print("unexpected error:", sys.exc_info()[0])
                    raise

                # resolve flux loop
                if loopless:
                    value = loopless_fva_iter(model, rxn)
                else:
                    value = model.solver.objective.value
                fva_results.at[rxn.id, what] = value
                print(colored(rxn.id + '(' + what + ') : ' + '%2.6f'%(value), 'red'))

                model.solver.objective.set_linear_coefficients({rxn.forward_variable: 0, rxn.reverse_variable: 0}) # reset
    return fva_results[["minimum", "maximum"]]

# Calculate diet flux from food item consumption
def dietflux_calculator(diet, nutritiondata, molmass):
    dietflux = {}

    for ndb_number in diet.index:
        print(ndb_number)
        total_weight_per_day = diet.loc[ndb_number,'Weight (g)']*diet.loc[ndb_number,'Frequency (times/day)']
        if len(str(ndb_number))==4:
            ndb_number_modified = 'USDA0' + str(ndb_number)
        else:
            ndb_number_modified = 'USDA' + str(ndb_number)
        nutritiondata_ndb = nutritiondata[nutritiondata['food']==ndb_number_modified]
        for idx in nutritiondata_ndb.index:
            print(idx)
            json_acceptable_string = nutritiondata_ndb.loc[idx,'nutrient'].replace("'", "\"").replace("None", "\"None\"")
            dict_nutr = json.loads(json_acceptable_string)
            if dict_nutr['mets'] != list():
                met = dict_nutr['mets'][0]
                nutr_value = nutritiondata_ndb.loc[idx,'nutr_value']
                if met in molmass.index:
                    met_mm = molmass.loc[met,'molecularmass']
                else:
                    continue
                unit = dict_nutr['unit']
                unit_conv = None
                if unit=='g':
                    unit_conv = 1
                if unit=='mg':
                    unit_conv = 1e-3
                if unit=='microg':
                    unit_conv = 1e-6

                if unit_conv is None:
                    continue
                else:
                    if met not in dietflux.keys():
                        dietflux[met] = 0.0
                    dietflux[met] += total_weight_per_day * (nutr_value*unit_conv/100) / met_mm * 1e3 / 24 / 200 # mmol/g(microbiota)/hr, assume human microbiota=200g

    df_dietflux = pd.DataFrame.from_dict(dietflux, orient='index', columns=['Influx (mmol/gDW/h)'])
    return df_dietflux

# modify diet flux converted from food items to make it realistic
def diet_gapfill(diet, essential_metabolites):

    new_diet = deepcopy(diet)

    # multiply by 100
    new_diet = new_diet * 10

    # Add essential metabolites not yet included in the entered diet
    missing_uptakes = list(set(essential_metabolites).difference(set(new_diet.index)))

    # add the missing exchange reactions to the adapted diet
    for met in missing_uptakes:
        new_diet.loc[met] = -1

    # Increase the uptake rate of micronutrients with too low defined uptake
    # rates to sustain microbiota model growth (below 1e-6 mol/day/person).
    # Lower bounds will be relaxed by factor 100 if allowed uptake is below 0.1 mmol*gDW-1*hr-1.
    micronutrients =['adocbl','vitd2','vitd3','psyl','gum','bglc','phyQ','fol','5mthf','q10','retinol_9_cis','pydxn','pydam','pydx','pheme','ribflv',
                     'thm','avite1','pnto_R','na1','cl','k','pi','zn2','cu2']
    for idx in new_diet.index:
        if idx in micronutrients and abs(new_diet.loc[idx].values)<=0.1:
            new_diet.loc[idx] *= 100
        # folate uptake needs to be at least 1
        if idx == 'fol' and abs(new_diet.loc['fol'].values)<1:
            new_diet.loc['fol'] = -1

    return new_diet

# apply given medium to Cobra model
def use_diet(model, medium, model_type='community'):
    if medium is not None:
        # get exchange reactions
        ex_rxns = None
        if model_type=='community':
            ex_rxns = [ex for ex in model.reactions if ex.community_id == "medium"]
        if model_type=='individual':
            ex_rxns = model.exchanges
        if ex_rxns is None:
            raise ValueError('model_type %s unrecognized.' % (model_type))

        # apply diet flux
        number_of_medium_components_found = 0
        for rxn in ex_rxns:
            # get Recon2 id for the metabolite
            if model_type=='community':
                met_id = rxn.id[:-2].replace('EX_M_','EX_').split('EX_')[1]
                #print(rxn.id, met_id)
            if model_type=='individual':
                if '(e)' in rxn.id:
                    met_id = rxn.id.replace('EX_M_','EX_').split('EX_')[1].split('(e)')[0]
                elif '[e]' in rxn.id:
                    met_id = rxn.id.replace('EX_M_','EX_').split('EX_')[1].split('[e]')[0]
                elif '__40__e__41__' in rxn.id:
                    met_id = rxn.id.replace('EX_M_','EX_').split('EX_')[1].split('__40__e__41__')[0]
                elif '__91__e__93__' in rxn.id:
                    met_id = rxn.id.replace('EX_M_','EX_').split('EX_')[1].split('__91__e__93__')[0]
                else:
                    raise RuntimeError('no external compartment tag found in %s.'%(rxn.id))
            if met_id in medium.index:
                rxn.lower_bound = medium.loc[met_id].values[0]
                number_of_medium_components_found += 1
            else:
                rxn.lower_bound = 0

        if number_of_medium_components_found == 0:
            warnings.warn("No medium component was found in exchange reactions.")

    return model

def add_enzymatic_constraints(model, tblenzymecost, protein_pool=0.32, ec_cutoff=9000, replacement_method='use_cutoff', reaction_suffix=None, suffix_weight=1.0):
    # protein_pool: maximum protein fraction used to synthesize metabolic enzymes
    # ec_cutoff: enzyme cost larger than this value will be considered unrealistic and replaced with
    # reaction_suffix: only consider enzyme cost for those reactions with suffix reaction_suffix
    # suffix_weight: fraction of the total metabolic protein pool allocated to reactions with reaction_suffix

    # get averaged enzyme cost based on known values (do it first and then separate the tables)
    ec_ave = [np.median(tblenzymecost.Forward),np.median(tblenzymecost.Reverse)]

    # get forward and reverse tables separately
    if replacement_method == 'use_average':
        tblenzymecost_forward = tblenzymecost['Forward'].to_frame()
        tblenzymecost_forward = tblenzymecost_forward[tblenzymecost_forward.Forward <= ec_cutoff]

        tblenzymecost_reverse = tblenzymecost['Reverse'].to_frame()
        tblenzymecost_reverse = tblenzymecost_reverse[tblenzymecost_reverse.Reverse <= ec_cutoff]
    elif replacement_method == 'use_cutoff':
        tblenzymecost_forward = tblenzymecost['Forward'].to_frame()
        tblenzymecost_forward[tblenzymecost_forward.Forward > ec_cutoff] = ec_cutoff

        tblenzymecost_reverse = tblenzymecost['Reverse'].to_frame()
        tblenzymecost_reverse[tblenzymecost_reverse.Reverse > ec_cutoff] = ec_cutoff
    else:
        raise ValueError('unknown replacement method')

    # find enzyme cost of individual reactions
    individual_enzyme_cost = []
    # total_enzyme_cost = Zero
    for rxn in model.reactions:

        # skip if reaction does not end with reaction_suffix
        if reaction_suffix is not None:
            if not rxn.id.endswith(reaction_suffix):
                continue

        # extract enzyme cost values for the reaction
        if 'ec-code' in rxn.annotation:
            # use the smallest enzyme cost if multiple EC number exists
            eclist = rxn.annotation['ec-code'].split(',')
            curr_ec_forward = []
            curr_ec_reverse = []
            for ec in eclist:

                # forward
                if ec in tblenzymecost_forward.index:
                    curr_ec_forward.append(tblenzymecost_forward.loc[ec,'Forward'])
                elif ec.endswith('TC'):
                    curr_ec_forward.append(300 * 110 / 65)
                else:
                    curr_ec_forward.append(ec_ave[0])

                # reverse
                if ec in tblenzymecost_reverse.index:
                    curr_ec_reverse.append(tblenzymecost_reverse.loc[ec,'Reverse'])
                elif ec.endswith('TC'):
                    curr_ec_reverse.append(300 * 110 / 65)
                else:
                    curr_ec_reverse.append(ec_ave[1])

            curr_ec_forward = min(curr_ec_forward)
            curr_ec_reverse = min(curr_ec_reverse)
            #print(rxn.forward_variable, type(rxn.forward_variable))
            # total_enzyme_cost += curr_ec_forward*rxn.forward_variable + curr_ec_reverse*rxn.reverse_variable
            individual_enzyme_cost.append(curr_ec_forward*rxn.forward_variable)
            individual_enzyme_cost.append(curr_ec_reverse*rxn.reverse_variable)
        else:
            if rxn.gene_reaction_rule != '':
                # total_enzyme_cost += ec_ave[0]*rxn.forward_variable + ec_ave[1]*rxn.reverse_variable
                # print(rxn.forward_variable)
                individual_enzyme_cost.append(ec_ave[0]*rxn.forward_variable)
                individual_enzyme_cost.append(ec_ave[1]*rxn.reverse_variable)

    # add enzyme cost constraint
    total_enzyme_cost = add(*individual_enzyme_cost)/1000/3600 # in unit of g(protein)/gDW
    if reaction_suffix is None:
        metabolic_enzyme_constraint = model.problem.Constraint(total_enzyme_cost, lb=0, ub=protein_pool, name='enzyme_constraint')
    else:
        # sum(f*mW/kcat) <= total protein, where f unit is mmol/h
        # divide both sides of microbiome weight, then f has unit mmol/h/gDW
        metabolic_enzyme_constraint = model.problem.Constraint(total_enzyme_cost, lb=0, ub=protein_pool*suffix_weight, name=reaction_suffix.lstrip('_')+'_enzyme_constraint')
    model.add_cons_vars([metabolic_enzyme_constraint])
    model.solver.update

    return model
