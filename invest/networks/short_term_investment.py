import pyAgrum as gum
import numpy as np

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

    # Define existing nodes
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

    # New Value and Quality Decision Nodes
    value_decision = gum.LabelizedVariable('ValueDecision', '', 3)
    value_decision.changeLabel(0, 'Cheap')
    value_decision.changeLabel(1, 'FairValue')
    value_decision.changeLabel(2, 'Expensive')
    st_model.addChanceNode(value_decision)

    quality_decision = gum.LabelizedVariable('QualityDecision', '', 3)
    quality_decision.changeLabel(0, 'High')
    quality_decision.changeLabel(1, 'Medium')
    quality_decision.changeLabel(2, 'Low')
    st_model.addChanceNode(quality_decision)

    # Buy/Sell Decision Node
    buy_signal = gum.LabelizedVariable('BuySignal', 'Final Decision', 2)
    buy_signal.changeLabel(0, 'Buy')
    buy_signal.changeLabel(1, 'Sell')
    st_model.addDecisionNode(buy_signal)

    # Utility Node
    st_utility = gum.LabelizedVariable('ST_Utility', 'Short-Term Investment Utility', 1)
    st_model.addUtilityNode(st_utility)

    # Arcs connecting nodes to Buy Signal
    st_model.addArc(st_model.idFromName('PriceMomentum'), st_model.idFromName('BuySignal'))
    st_model.addArc(st_model.idFromName('Volatility'), st_model.idFromName('BuySignal'))
    st_model.addArc(st_model.idFromName('Valuation'), st_model.idFromName('BuySignal'))
    st_model.addArc(st_model.idFromName('MarketCondition'), st_model.idFromName('BuySignal'))
    st_model.addArc(st_model.idFromName('ValueDecision'), st_model.idFromName('BuySignal'))
    st_model.addArc(st_model.idFromName('QualityDecision'), st_model.idFromName('BuySignal'))

    # Utility arc
    st_model.addArc(st_model.idFromName('BuySignal'), st_model.idFromName('ST_Utility'))

    # Define CPTs for Buy Signal (example values based on integrated decisions)
    st_model.cpt(st_model.idFromName('BuySignal'))[{
        'PriceMomentum': 'Uptrend', 'Volatility': 'Low', 'Valuation': 'Undervalued',
        'MarketCondition': 'Positive', 'ValueDecision': 'Cheap', 'QualityDecision': 'High'
    }] = [0.9, 0.1]  # Buy, Sell

    # Utility values (example values)
    st_model.utility(st_model.idFromName('ST_Utility'))[{'BuySignal': 'Buy'}] = [150]
    st_model.utility(st_model.idFromName('ST_Utility'))[{'BuySignal': 'Sell'}] = [-100]

    # Inference with example evidence
    ie = gum.ShaferShenoyLIMIDInference(st_model)
    if price_momentum_state:
        ie.addEvidence('PriceMomentum', {'Uptrend': [1, 0, 0], 'Stable': [0, 1, 0], 'Downtrend': [0, 0, 1]}.get(price_momentum_state, [0, 1, 0]))
    if volatility_state:
        ie.addEvidence('Volatility', {'Low': [1, 0, 0], 'Medium': [0, 1, 0], 'High': [0, 0, 1]}.get(volatility_state, [0, 1, 0]))
    if valuation_state:
        ie.addEvidence('Valuation', {'Undervalued': [1, 0, 0], 'FairlyValued': [0, 1, 0], 'Overvalued': [0, 0, 1]}.get(valuation_state, [0, 1, 0]))
    if market_condition_state:
        ie.addEvidence('MarketCondition', {'Positive': [1, 0, 0], 'Neutral': [0, 1, 0], 'Negative': [0, 0, 1]}.get(market_condition_state, [0, 1, 0]))
    if value_decision_state:
        ie.addEvidence('ValueDecision', {'Cheap': [1, 0, 0], 'FairValue': [0, 1, 0], 'Expensive': [0, 0, 1]}.get(value_decision_state, [0, 1, 0]))
    if quality_decision_state:
        ie.addEvidence('QualityDecision', {'High': [1, 0, 0], 'Medium': [0, 1, 0], 'Low': [0, 0, 1]}.get(quality_decision_state, [0, 1, 0]))

    ie.makeInference()
    var = ie.posteriorUtility('BuySignal').variable('BuySignal')
    decision_index = np.argmax(ie.posteriorUtility('BuySignal').toarray())
    decision = var.label(int(decision_index))

    return decision
