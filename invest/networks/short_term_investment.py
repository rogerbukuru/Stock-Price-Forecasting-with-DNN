import pyAgrum as gum
import numpy as np
from invest.utilities import save_bdn_diagram

def short_term_investment(price_momentum_state, volatility_state, valuation_state, 
                          market_condition_state, value_decision_state, quality_decision_state):
    """
    Returns the final decision of the Short-Term Investment Network based on short-term and long-term signals.

    Parameters
    ----------
    price_momentum_state : str
       State of price momentum (e.g., "Uptrend", "Stable", "Downtrend")
    volatility_state : str
       State of volatility (e.g., "High", "Medium", "Low")
    valuation_state : str
       State of stock valuation (e.g., "Undervalued", "Fairly Valued", "Overvalued")
    market_condition_state : str
       Overall market condition (e.g., "Positive", "Neutral", "Negative")
    value_decision_state : str
       Output from the Value Evaluation Network (e.g., "Cheap", "FairValue", "Expensive")
    quality_decision_state : str
       Output from the Quality Evaluation Network (e.g., "High", "Medium", "Low")

    Returns
    -------
    str
    """

    st_model = gum.InfluenceDiagram()

    # Define nodes
    price_momentum = gum.LabelizedVariable('PriceMomentum', '', 3)
    price_momentum.changeLabel(0, 'Uptrend')
    price_momentum.changeLabel(1, 'Stable')
    price_momentum.changeLabel(2, 'Downtrend')
    st_model.addChanceNode(price_momentum)

    volatility = gum.LabelizedVariable('Volatility', '', 3)
    volatility.changeLabel(0, 'Low')
    volatility.changeLabel(1, 'Medium')
    volatility.changeLabel(2, 'High')
    st_model.addChanceNode(volatility)

    valuation = gum.LabelizedVariable('Valuation', '', 3)
    valuation.changeLabel(0, 'Undervalued')
    valuation.changeLabel(1, 'FairlyValued')
    valuation.changeLabel(2, 'Overvalued')
    st_model.addChanceNode(valuation)

    market_condition = gum.LabelizedVariable('MarketCondition', '', 3)
    market_condition.changeLabel(0, 'Positive')
    market_condition.changeLabel(1, 'Neutral')
    market_condition.changeLabel(2, 'Negative')
    st_model.addChanceNode(market_condition)

    value_decision = gum.LabelizedVariable('Value', '', 3)
    value_decision.changeLabel(0, 'Cheap')
    value_decision.changeLabel(1, 'FairValue')
    value_decision.changeLabel(2, 'Expensive')
    st_model.addChanceNode(value_decision)

    quality_decision = gum.LabelizedVariable('Quality', '', 3)
    quality_decision.changeLabel(0, 'High')
    quality_decision.changeLabel(1, 'Medium')
    quality_decision.changeLabel(2, 'Low')
    st_model.addChanceNode(quality_decision)

    # Define the Performance node
    performance = gum.LabelizedVariable('Performance', '', 3)
    performance.changeLabel(0, 'Positive')
    performance.changeLabel(1, 'Neutral')
    performance.changeLabel(2, 'Negative')
    st_model.addChanceNode(performance)

    # Buy/Sell Decision Node
    buy_signal = gum.LabelizedVariable('BuySignal', 'Final Decision', 2)
    buy_signal.changeLabel(0, 'Buy')
    buy_signal.changeLabel(1, 'Sell')
    st_model.addDecisionNode(buy_signal)

    # Utility Node
    st_utility = gum.LabelizedVariable('ST_Utility', 'Short-Term Investment Utility', 1)
    st_model.addUtilityNode(st_utility)

    # Define arcs
    st_model.addArc(st_model.idFromName('PriceMomentum'), st_model.idFromName('BuySignal'))
    st_model.addArc(st_model.idFromName('Volatility'), st_model.idFromName('BuySignal'))
    st_model.addArc(st_model.idFromName('Valuation'), st_model.idFromName('BuySignal'))
    st_model.addArc(st_model.idFromName('MarketCondition'), st_model.idFromName('BuySignal'))
    st_model.addArc(st_model.idFromName('Value'), st_model.idFromName('BuySignal'))
    st_model.addArc(st_model.idFromName('Quality'), st_model.idFromName('BuySignal'))

    st_model.addArc(st_model.idFromName('Performance'), st_model.idFromName('BuySignal'))
    st_model.addArc(st_model.idFromName('Performance'), st_model.idFromName('ST_Utility'))
    st_model.addArc(st_model.idFromName('BuySignal'), st_model.idFromName('ST_Utility'))
    st_model.addArc(st_model.idFromName('Performance'), st_model.idFromName('Value'))
    st_model.addArc(st_model.idFromName('Performance'), st_model.idFromName('Quality'))
    st_model.addArc(st_model.idFromName('Performance'), st_model.idFromName('Valuation'))

    st_model.addArc(st_model.idFromName('MarketCondition'), st_model.idFromName('Volatility'))
    st_model.addArc(st_model.idFromName('MarketCondition'), st_model.idFromName('PriceMomentum'))


    # Performance node (independent probabilities)
    st_model.cpt(st_model.idFromName('Performance'))[0] = 1 / 3  # Positive
    st_model.cpt(st_model.idFromName('Performance'))[1] = 1 / 3  # Neutral
    st_model.cpt(st_model.idFromName('Performance'))[2] = 1 / 3  # Negative

    # Value node, influenced by Performance
    st_model.cpt(st_model.idFromName('Value'))[{'Performance': 'Positive'}] = [0.85, 0.10, 0.05]
    st_model.cpt(st_model.idFromName('Value'))[{'Performance': 'Neutral'}] = [0.20, 0.60, 0.20]
    st_model.cpt(st_model.idFromName('Value'))[{'Performance': 'Negative'}] = [0.05, 0.10, 0.85]

    # Quality node, influenced by Performance
    st_model.cpt(st_model.idFromName('Quality'))[{'Performance': 'Positive'}] = [0.85, 0.10, 0.05]
    st_model.cpt(st_model.idFromName('Quality'))[{'Performance': 'Neutral'}] = [0.20, 0.60, 0.20]
    st_model.cpt(st_model.idFromName('Quality'))[{'Performance': 'Negative'}] = [0.05, 0.10, 0.85]

    # Valuation node, influenced by Performance
    st_model.cpt(st_model.idFromName('Valuation'))[{'Performance': 'Positive'}] = [0.85, 0.10, 0.05]
    st_model.cpt(st_model.idFromName('Valuation'))[{'Performance': 'Neutral'}] = [0.20, 0.60, 0.20]
    st_model.cpt(st_model.idFromName('Valuation'))[{'Performance': 'Negative'}] = [0.05, 0.10, 0.85]

    # Volatility node, influenced by MarketCondition
    st_model.cpt(st_model.idFromName('Volatility'))[{'MarketCondition': 'Positive'}] = [0.60, 0.30, 0.10]
    st_model.cpt(st_model.idFromName('Volatility'))[{'MarketCondition': 'Neutral'}] = [0.30, 0.50, 0.20]
    st_model.cpt(st_model.idFromName('Volatility'))[{'MarketCondition': 'Negative'}] = [0.10, 0.30, 0.60]

    # PriceMomentum node, influenced by MarketCondition
    st_model.cpt(st_model.idFromName('PriceMomentum'))[{'MarketCondition': 'Positive'}] = [0.60, 0.30, 0.10]
    st_model.cpt(st_model.idFromName('PriceMomentum'))[{'MarketCondition': 'Neutral'}] = [0.30, 0.50, 0.20]
    st_model.cpt(st_model.idFromName('PriceMomentum'))[{'MarketCondition': 'Negative'}] = [0.10, 0.30, 0.60]

    # MarketCondition node (independent probabilities)
    st_model.cpt(st_model.idFromName('MarketCondition'))[0] = 1 / 3  # Positive
    st_model.cpt(st_model.idFromName('MarketCondition'))[1] = 1 / 3  # Neutral
    st_model.cpt(st_model.idFromName('MarketCondition'))[2] = 1 / 3  # Negative

    # Assign utilities based on the BuySignal and Performance states
    st_model.utility(st_model.idFromName('ST_Utility'))[{'BuySignal': 'Buy', 'Performance': 'Positive'}] = [300]
    st_model.utility(st_model.idFromName('ST_Utility'))[{'BuySignal': 'Buy', 'Performance': 'Neutral'}] = [100]
    st_model.utility(st_model.idFromName('ST_Utility'))[{'BuySignal': 'Buy', 'Performance': 'Negative'}] = [-200]
    st_model.utility(st_model.idFromName('ST_Utility'))[{'BuySignal': 'Sell', 'Performance': 'Positive'}] = [-100]
    st_model.utility(st_model.idFromName('ST_Utility'))[{'BuySignal': 'Sell', 'Performance': 'Neutral'}] = [50]
    st_model.utility(st_model.idFromName('ST_Utility'))[{'BuySignal': 'Sell', 'Performance': 'Negative'}] = [200]

    # Perform inference with evidence
    ie = gum.ShaferShenoyLIMIDInference(st_model)
    ie.addEvidence('PriceMomentum', {'Uptrend': [1, 0, 0], 'Stable': [0, 1, 0], 'Downtrend': [0, 0, 1]}.get(price_momentum_state, [0, 1, 0]))
    ie.addEvidence('Volatility', {'Low': [1, 0, 0], 'Medium': [0, 1, 0], 'High': [0, 0, 1]}.get(volatility_state, [0, 1, 0]))
    ie.addEvidence('Valuation', {'Undervalued': [1, 0, 0], 'FairlyValued': [0, 1, 0], 'Overvalued': [0, 0, 1]}.get(valuation_state, [0, 1, 0]))
    ie.addEvidence('MarketCondition', {'Positive': [1, 0, 0], 'Neutral': [0, 1, 0], 'Negative': [0, 0, 1]}.get(market_condition_state, [0, 1, 0]))
    ie.addEvidence('Value', {'Cheap': [1, 0, 0], 'FairValue': [0, 1, 0], 'Expensive': [0, 0, 1]}.get(value_decision_state, [0, 1, 0]))
    ie.addEvidence('Quality', {'High': [1, 0, 0], 'Medium': [0, 1, 0], 'Low': [0, 0, 1]}.get(quality_decision_state, [0, 1, 0]))

    ie.makeInference()
    var = ie.posteriorUtility('BuySignal').variable('BuySignal')
    decision_index = np.argmax(ie.posteriorUtility('BuySignal').toarray())
    decision = var.label(int(decision_index))
    save_bdn_diagram(st_model, filename="short_term_investment_bdn")

    return decision
