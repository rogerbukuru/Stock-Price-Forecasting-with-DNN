import os
import numpy as np
import pyAgrum as gum


# Maximum Likelihood Estimation (MLE)
def learn_cpt_mle(data, node, parent_values):
    # Placeholder: Replace with actual MLE calculation for CPT
    return np.random.dirichlet(np.ones(len(parent_values)))


# Bayesian Parameter Estimation (BPE)
def learn_cpt_bpe(data, node, parent_values, prior=None):
    if prior is None:
        prior = np.ones(len(parent_values))  # Default uniform prior
    # Placeholder: Replace with actual BPE calculation for CPT
    return np.random.dirichlet(prior + np.random.randint(1, 100, size=len(parent_values)))


# Maximum Entropy (ME)
def learn_cpt_me(data, node, parent_values):
    # Placeholder: Replace with actual ME calculation for CPT
    return np.full(len(parent_values), 1 / len(parent_values))  # Equal probability (Max Entropy)


def investment_recommendation(value_decision, quality_decision, data=None, algorithm='hardcoded'):
    """
    Returns the final Investment Recommendation for the BNs

    Parameters
    ----------
    value_decision : str
       Final decision output of the Value Network
    quality_decision : str
       Final decision output of the Quality Network
    data : np.array, optional
       Dataset for learning CPTs (if provided).
    algorithm : str, optional
       Algorithm used to learn CPTs ('mle', 'bpe', 'me', or 'hardcoded').

    Returns
    -------
    str
        Final investment recommendation ('Yes' or 'No').
    """
    value_decision_state = value_decision
    quality_decision_state = quality_decision
    ir_model = gum.InfluenceDiagram()

    # Define variables and nodes
    investable = gum.LabelizedVariable('Investable', 'Investable share', 2)
    investable.changeLabel(0, 'Yes')
    investable.changeLabel(1, 'No')
    ir_model.addDecisionNode(investable)

    share_performance = gum.LabelizedVariable('Performance', '', 3)
    share_performance.changeLabel(0, 'Positive')
    share_performance.changeLabel(1, 'Stagnant')
    share_performance.changeLabel(2, 'Negative')
    ir_model.addChanceNode(share_performance)

    value = gum.LabelizedVariable('Value', 'Value', 3)
    value.changeLabel(0, 'Cheap')
    value.changeLabel(1, 'FairValue')
    value.changeLabel(2, 'Expensive')
    ir_model.addChanceNode(value)

    quality = gum.LabelizedVariable('Quality', 'Quality', 3)
    quality.changeLabel(0, 'High')
    quality.changeLabel(1, 'Medium')
    quality.changeLabel(2, 'Low')
    ir_model.addChanceNode(quality)

    investment_utility = gum.LabelizedVariable('I_Utility', '', 1)
    ir_model.addUtilityNode(investment_utility)

    ir_model.addArc(ir_model.idFromName('Performance'), ir_model.idFromName('Quality'))
    ir_model.addArc(ir_model.idFromName('Performance'), ir_model.idFromName('Value'))
    ir_model.addArc(ir_model.idFromName('Performance'), ir_model.idFromName('I_Utility'))

    ir_model.addArc(ir_model.idFromName('Value'), ir_model.idFromName('Investable'))
    ir_model.addArc(ir_model.idFromName('Quality'), ir_model.idFromName('Investable'))
    ir_model.addArc(ir_model.idFromName('Investable'), ir_model.idFromName('I_Utility'))

    # Utility values
    ir_model.utility(ir_model.idFromName('I_Utility'))[{'Investable': 'Yes'}] = [[300], [-100], [-250]]
    ir_model.utility(ir_model.idFromName('I_Utility'))[{'Investable': 'No'}] = [[-200], [100], [200]]

    # Learn CPTs based on selected algorithm
    if data is not None:
        if algorithm == 'mle':
            ir_model.cpt(ir_model.idFromName('Performance'))[:] = learn_cpt_mle(data, 'Performance', [0, 1, 2])
            ir_model.cpt(ir_model.idFromName('Value'))[:] = learn_cpt_mle(data, 'Value', [0, 1, 2])
            ir_model.cpt(ir_model.idFromName('Quality'))[:] = learn_cpt_mle(data, 'Quality', [0, 1, 2])
        elif algorithm == 'bpe':
            ir_model.cpt(ir_model.idFromName('Performance'))[:] = learn_cpt_bpe(data, 'Performance', [0, 1, 2])
            ir_model.cpt(ir_model.idFromName('Value'))[:] = learn_cpt_bpe(data, 'Value', [0, 1, 2])
            ir_model.cpt(ir_model.idFromName('Quality'))[:] = learn_cpt_bpe(data, 'Quality', [0, 1, 2])
        elif algorithm == 'me':
            ir_model.cpt(ir_model.idFromName('Performance'))[:] = learn_cpt_me(data, 'Performance', [0, 1, 2])
            ir_model.cpt(ir_model.idFromName('Value'))[:] = learn_cpt_me(data, 'Value', [0, 1, 2])
            ir_model.cpt(ir_model.idFromName('Quality'))[:] = learn_cpt_me(data, 'Quality', [0, 1, 2])
    else:
        # Hardcoded CPTs (default)
        ir_model.cpt(ir_model.idFromName('Performance'))[0] = 1 / 3  # Positive
        ir_model.cpt(ir_model.idFromName('Performance'))[1] = 1 / 3  # Stagnant
        ir_model.cpt(ir_model.idFromName('Performance'))[2] = 1 / 3  # Negative

        ir_model.cpt(ir_model.idFromName('Value'))[{'Performance': 'Positive'}] = [0.85, 0.10, 0.05]
        ir_model.cpt(ir_model.idFromName('Value'))[{'Performance': 'Stagnant'}] = [0.20, 0.60, 0.20]
        ir_model.cpt(ir_model.idFromName('Value'))[{'Performance': 'Negative'}] = [0.05, 0.10, 0.85]

        ir_model.cpt(ir_model.idFromName('Quality'))[{'Performance': 'Positive'}] = [0.85, 0.10, 0.05]
        ir_model.cpt(ir_model.idFromName('Quality'))[{'Performance': 'Stagnant'}] = [0.20, 0.60, 0.20]
        ir_model.cpt(ir_model.idFromName('Quality'))[{'Performance': 'Negative'}] = [0.05, 0.10, 0.85]

    # Save model
    output_file = os.path.join('res', 'i_r')
    if not os.path.exists(output_file):
        os.makedirs(output_file)

    ie = gum.ShaferShenoyLIMIDInference(ir_model)

    # Provide evidence
    if value_decision_state == "Cheap":
        ie.addEvidence('Value', [1, 0, 0])
    elif value_decision_state == "FairValue":
        ie.addEvidence('Value', [0, 1, 0])
    else:
        ie.addEvidence('Value', [0, 0, 1])

    if quality_decision_state == "High":
        ie.addEvidence('Quality', [1, 0, 0])
    elif quality_decision_state == "Medium":
        ie.addEvidence('Quality', [0, 1, 0])
    else:
        ie.addEvidence('Quality', [0, 0, 1])

    ie.makeInference()

    # Calculate posterior utility and make a decision
    var = ie.posteriorUtility('Investable').variable('Investable')
    decision_index = np.argmax(ie.posteriorUtility('Investable').toarray())
    decision = var.label(int(decision_index))

    # Return the final decision for investable recommendation
    return decision