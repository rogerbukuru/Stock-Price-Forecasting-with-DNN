import os
import numpy as np
import pyAgrum as gum

# Define CPT learning algorithms outside the main function

def learn_cpt_mle(data, variables):
    """Maximum Likelihood Estimation (MLE) for learning CPTs."""
    # Implement MLE based on the data and variables
    # This is a placeholder for actual learning
    # Return learned CPTs
    learned_cpt = {}
    for var in variables:
        learned_cpt[var] = np.random.dirichlet(np.ones(len(variables[var])))
    return learned_cpt

def learn_cpt_me(data, variables):
    """Maximum Entropy (ME) for learning CPTs."""
    # Implement ME algorithm for learning CPTs
    # This is a placeholder for actual learning
    # Return learned CPTs
    learned_cpt = {}
    for var in variables:
        learned_cpt[var] = np.random.dirichlet(np.ones(len(variables[var])))
    return learned_cpt

def learn_cpt_bpe(data, variables):
    """Bayesian Parameter Estimation (BPE) for learning CPTs."""
    # Implement BPE based on the data and variables
    # This is a placeholder for actual learning
    # Return learned CPTs
    learned_cpt = {}
    for var in variables:
        learned_cpt[var] = np.random.dirichlet(np.ones(len(variables[var])))
    return learned_cpt


def quality_network(roe_vs_coe_state, relative_debt_equity_state, cagr_vs_inflation_state, systematic_risk_state=None,
                    extension=False, data=None, algorithm='hardcoded'):
    """
    Returns the final Quality Evaluation decision

    Parameters
    ----------
    roe_vs_coe_state : str
       Discrete state for Return on Equity vs Cost of Equity
    relative_debt_equity_state : str
       Discrete state for Relative Debt to Equity
    cagr_vs_inflation_state: str
        Discrete state for Compound Annual Growth Rate vs Inflation
    systematic_risk_state: Union[None, str]
        Discrete state for Share Beta, default is None
    extension: bool
        Boolean to indicate whether the extended network must be run
    data : Union[None, DataFrame]
        Optional data for learning CPTs
    algorithm: str
        Algorithm to use for learning CPTs ('MLE', 'ME', 'BPE', 'hardcoded')
    
    Returns
    -------
    str
    """
    qe_model = gum.InfluenceDiagram()

    # Decision node
    quality_decision = gum.LabelizedVariable('Quality', '', 3)
    quality_decision.changeLabel(0, 'High')
    quality_decision.changeLabel(1, 'Medium')
    quality_decision.changeLabel(2, 'Low')
    qe_model.addDecisionNode(quality_decision)

    # FutureSharePerformance node
    future_share_performance = gum.LabelizedVariable('FutureSharePerformance', '', 3)
    future_share_performance.changeLabel(0, 'Positive')
    future_share_performance.changeLabel(1, 'Stagnant')
    future_share_performance.changeLabel(2, 'Negative')
    qe_model.addChanceNode(future_share_performance)

    # CAGR vs Inflation node
    cagr_vs_inflation = gum.LabelizedVariable('CAGRvsInflation', '', 3)
    cagr_vs_inflation.changeLabel(0, 'InflationPlus')
    cagr_vs_inflation.changeLabel(1, 'Inflation')
    cagr_vs_inflation.changeLabel(2, 'InflationMinus')
    qe_model.addChanceNode(cagr_vs_inflation)

    # ROE vs COE node
    roe_vs_coe = gum.LabelizedVariable('ROEvsCOE', '', 3)
    roe_vs_coe.changeLabel(0, 'Above')
    roe_vs_coe.changeLabel(1, 'EqualTo')
    roe_vs_coe.changeLabel(2, 'Below')
    qe_model.addChanceNode(roe_vs_coe)

    # Relative debt to equity node
    relative_debt_equity = gum.LabelizedVariable('RelDE', '', 3)
    relative_debt_equity.changeLabel(0, 'Above')
    relative_debt_equity.changeLabel(1, 'EqualTo')
    relative_debt_equity.changeLabel(2, 'Below')
    qe_model.addChanceNode(relative_debt_equity)

    quality_utility = gum.LabelizedVariable('Q_Utility', '', 1)
    qe_model.addUtilityNode(quality_utility)

    qe_model.addArc(qe_model.idFromName('FutureSharePerformance'), qe_model.idFromName('CAGRvsInflation'))
    qe_model.addArc(qe_model.idFromName('FutureSharePerformance'), qe_model.idFromName('ROEvsCOE'))
    qe_model.addArc(qe_model.idFromName('FutureSharePerformance'), qe_model.idFromName('RelDE'))
    qe_model.addArc(qe_model.idFromName('FutureSharePerformance'), qe_model.idFromName('Q_Utility'))

    qe_model.addArc(qe_model.idFromName('CAGRvsInflation'), qe_model.idFromName('Quality'))

    qe_model.addArc(qe_model.idFromName('ROEvsCOE'), qe_model.idFromName('Quality'))

    qe_model.addArc(qe_model.idFromName('RelDE'), qe_model.idFromName('Quality'))

    qe_model.addArc(qe_model.idFromName('Quality'), qe_model.idFromName('Q_Utility'))

    # Utilities
    qe_model.utility(qe_model.idFromName('Q_Utility'))[{'Quality': 'High'}] = [[100], [0], [-100]]
    qe_model.utility(qe_model.idFromName('Q_Utility'))[{'Quality': 'Medium'}] = [[50], [100], [-50]]
    qe_model.utility(qe_model.idFromName('Q_Utility'))[{'Quality': 'Low'}] = [[0], [50], [100]]

    # CPT learning logic
    variables = {
        'FutureSharePerformance': ['Positive', 'Stagnant', 'Negative'],
        'RelDE': ['Above', 'EqualTo', 'Below'],
        'ROEvsCOE': ['Above', 'EqualTo', 'Below'],
        'CAGRvsInflation': ['InflationPlus', 'Inflation', 'InflationMinus']
    }

    if data is not None:
        if algorithm == 'MLE':
            learned_cpt = learn_cpt_mle(data, variables)
        elif algorithm == 'ME':
            learned_cpt = learn_cpt_me(data, variables)
        elif algorithm == 'BPE':
            learned_cpt = learn_cpt_bpe(data, variables)
        else:
            raise ValueError("Invalid algorithm specified")
    else:
        learned_cpt = None  # Use hardcoded CPTs

    # CPTs
    # FutureSharePerformance
    if learned_cpt is None:
        qe_model.cpt(qe_model.idFromName('FutureSharePerformance'))[0] = 1 / 3  # Positive
        qe_model.cpt(qe_model.idFromName('FutureSharePerformance'))[1] = 1 / 3  # Stagnant
        qe_model.cpt(qe_model.idFromName('FutureSharePerformance'))[2] = 1 / 3  # Negative
    else:
        qe_model.cpt(qe_model.idFromName('FutureSharePerformance'))[:] = learned_cpt['FutureSharePerformance']

    # RelDE
    if learned_cpt is None:
        qe_model.cpt(qe_model.idFromName('RelDE'))[{'FutureSharePerformance': 'Positive'}] = [0.05, 0.15, 0.80]
        qe_model.cpt(qe_model.idFromName('RelDE'))[{'FutureSharePerformance': 'Stagnant'}] = [0.15, 0.70, 0.15]
        qe_model.cpt(qe_model.idFromName('RelDE'))[{'FutureSharePerformance': 'Negative'}] = [0.80, 0.15, 0.05]
    else:
        qe_model.cpt(qe_model.idFromName('RelDE'))[:] = learned_cpt['RelDE']

    # ROE vs COE
    if learned_cpt is None:
        qe_model.cpt(qe_model.idFromName('ROEvsCOE'))[{'FutureSharePerformance': 'Positive'}] = [0.80, 0.15, 0.05]
        qe_model.cpt(qe_model.idFromName('ROEvsCOE'))[{'FutureSharePerformance': 'Stagnant'}] = [0.20, 0.60, 0.20]
        qe_model.cpt(qe_model.idFromName('ROEvsCOE'))[{'FutureSharePerformance': 'Negative'}] = [0.05, 0.15, 0.80]
    else:
        qe_model.cpt(qe_model.idFromName('ROEvsCOE'))[:] = learned_cpt['ROEvsCOE']

    # CAGR vs Inflation
    if learned_cpt is None:
        qe_model.cpt(qe_model.idFromName('CAGRvsInflation'))[{'FutureSharePerformance': 'Positive'}] = [0.80, 0.15, 0.05]
        qe_model.cpt(qe_model.idFromName('CAGRvsInflation'))[{'FutureSharePerformance': 'Stagnant'}] = [0.15, 0.70, 0.15]
        qe_model.cpt(qe_model.idFromName('CAGRvsInflation'))[{'FutureSharePerformance': 'Negative'}] = [0.05, 0.15, 0.8]
    else:
        qe_model.cpt(qe_model.idFromName('CAGRvsInflation'))[:] = learned_cpt['CAGRvsInflation']

    # Extension
    if extension:
        systematic_risk = gum.LabelizedVariable('SystematicRisk', '', 3)
        systematic_risk.changeLabel(0, 'greater')  # Greater than Market
        systematic_risk.changeLabel(1, 'EqualTo')
        systematic_risk.changeLabel(2, 'lower')
        qe_model.addChanceNode(systematic_risk)

        qe_model.addArc(qe_model.idFromName('FutureSharePerformance'), qe_model.idFromName('SystematicRisk'))
        qe_model.addArc(qe_model.idFromName('SystematicRisk'), qe_model.idFromName('Quality'))

        # Add CPTs for Systematic Risk
        if learned_cpt is None:
            qe_model.cpt(qe_model.idFromName('SystematicRisk'))[{'FutureSharePerformance': 'Positive'}] = [0.80, 0.15, 0.05]
            qe_model.cpt(qe_model.idFromName('SystematicRisk'))[{'FutureSharePerformance': 'Stagnant'}] = [0.15, 0.70, 0.15]
            qe_model.cpt(qe_model.idFromName('SystematicRisk'))[{'FutureSharePerformance': 'Negative'}] = [0.05, 0.15, 0.8]
        else:
            qe_model.cpt(qe_model.idFromName('SystematicRisk'))[:] = learned_cpt['SystematicRisk']

    output_file = os.path.join('res', 'q_e')
    if not os.path.exists(output_file):
        os.makedirs(output_file)
    # gum.saveBN(qe_model, os.path.join(output_file, 'q_e.bifxml'))

    ie = gum.ShaferShenoyLIMIDInference(qe_model)

    if relative_debt_equity_state == "above":
        ie.addEvidence('RelDE', [1, 0, 0])
    elif relative_debt_equity_state == "EqualTo":
        ie.addEvidence('RelDE', [0, 1, 0])
    else:
        ie.addEvidence('RelDE', [0, 0, 1])

    if roe_vs_coe_state == "above":
        ie.addEvidence('ROEvsCOE', [1, 0, 0])
    elif roe_vs_coe_state == "EqualTo":
        ie.addEvidence('ROEvsCOE', [0, 1, 0])
    else:
        ie.addEvidence('ROEvsCOE', [0, 0, 1])

    if cagr_vs_inflation_state == "above":
        ie.addEvidence('CAGRvsInflation', [1, 0, 0])
    elif cagr_vs_inflation_state == "EqualTo":
        ie.addEvidence('CAGRvsInflation', [0, 1, 0])
    else:
        ie.addEvidence('CAGRvsInflation', [0, 0, 1])

    if extension:
        if systematic_risk_state == "greater":
            ie.addEvidence('SystematicRisk', [1, 0, 0])
        elif systematic_risk_state == "EqualTo":
            ie.addEvidence('SystematicRisk', [0, 1, 0])
        else:
            ie.addEvidence('SystematicRisk', [0, 0, 1])

    ie.makeInference()
    var = ie.posteriorUtility('Quality').variable('Quality')

    decision_index = np.argmax(ie.posteriorUtility('Quality').toarray())
    decision = var.label(int(decision_index))
    return format(decision)