"""
Tools Layer — Safe Financial Calculator
=========================================
Computes financial ratios (Altman Z-score, debt/equity, current ratio, etc.)
using simpleeval — no exec/eval, no arbitrary code execution.

This is the "sandbox" tool the agent calls when a query needs computation
rather than retrieval — e.g. "What is Enron's Altman Z-score for 2000?"
"""

from simpleeval import simple_eval
from typing import Optional


# ── Known financial figures (manually extracted from filings for demo) ──────
# In a full production system this would come from XBRL structured data.
# For this portfolio project we hardcode verified figures from public 10-Ks
# so the calculator tool has real numbers to work with.

FINANCIAL_DATA = {
    ("enron", "2000"): {
        "total_assets":        65_503_000_000,
        "total_liabilities":   54_033_000_000,
        "current_assets":      30_381_000_000,
        "current_liabilities": 28_406_000_000,
        "retained_earnings":    3_226_000_000,
        "ebit":                 1_953_000_000,
        "market_cap":          63_400_000_000,
        "sales":               100_789_000_000,
        "net_income":            979_000_000,
    },
    ("svb", "2022"): {
        "total_assets":       211_793_000_000,
        "total_liabilities":  197_308_000_000,
        "current_assets":      27_221_000_000,   # cash + short-term investments
        "current_liabilities": 173_109_000_000,  # deposits
        "retained_earnings":    8_323_000_000,
        "ebit":                 1_269_000_000,
        "market_cap":           6_800_000_000,    # post-collapse low
        "sales":                7_516_000_000,
        "net_income":           1_509_000_000,
    },
    ("apple", "2023"): {
        "total_assets":       352_583_000_000,
        "total_liabilities":  290_437_000_000,
        "current_assets":     143_692_000_000,
        "current_liabilities": 145_308_000_000,
        "retained_earnings":   -214_000_000,      # accumulated deficit (buybacks)
        "ebit":               114_301_000_000,
        "market_cap":       2_900_000_000_000,
        "sales":              383_285_000_000,
        "net_income":          96_995_000_000,
    },
}


def get_financials(company_key: str, year: str) -> Optional[dict]:
    """Look up known financial figures for a company/year."""
    return FINANCIAL_DATA.get((company_key, year))


def altman_z_score(company_key: str, year: str) -> dict:
    """
    Altman Z-Score — predicts bankruptcy risk.
    Z = 1.2*A + 1.4*B + 3.3*C + 0.6*D + 1.0*E

    A = Working Capital / Total Assets
    B = Retained Earnings / Total Assets
    C = EBIT / Total Assets
    D = Market Value of Equity / Total Liabilities
    E = Sales / Total Assets

    Interpretation:
      Z > 2.99  → Safe zone
      1.81–2.99 → Grey zone (caution)
      Z < 1.81  → Distress zone (high bankruptcy risk)
    """
    data = get_financials(company_key, year)
    if not data:
        return {"error": f"No financial data available for {company_key} {year}"}

    working_capital = data["current_assets"] - data["current_liabilities"]
    total_assets = data["total_assets"]

    A = working_capital / total_assets
    B = data["retained_earnings"] / total_assets
    C = data["ebit"] / total_assets
    D = data["market_cap"] / data["total_liabilities"]
    E = data["sales"] / total_assets

    formula = "1.2*A + 1.4*B + 3.3*C + 0.6*D + 1.0*E"
    z = simple_eval(formula, names={"A": A, "B": B, "C": C, "D": D, "E": E})

    if z > 2.99:
        zone, risk = "Safe Zone", "low"
    elif z > 1.81:
        zone, risk = "Grey Zone", "moderate"
    else:
        zone, risk = "Distress Zone", "high"

    return {
        "company":       company_key,
        "year":          year,
        "z_score":       round(z, 2),
        "zone":          zone,
        "risk_level":    risk,
        "components": {
            "A_working_capital_ratio":  round(A, 3),
            "B_retained_earnings_ratio": round(B, 3),
            "C_ebit_ratio":              round(C, 3),
            "D_market_to_liabilities":   round(D, 3),
            "E_asset_turnover":          round(E, 3),
        },
        "interpretation": (
            f"Z-score of {round(z,2)} places {company_key.upper()} in the {zone}. "
            f"{'This indicates significant bankruptcy risk.' if risk=='high' else ''}"
            f"{'This warrants monitoring but is not immediately alarming.' if risk=='moderate' else ''}"
            f"{'Financial structure appears stable.' if risk=='low' else ''}"
        ),
    }


def current_ratio(company_key: str, year: str) -> dict:
    """Current Ratio = Current Assets / Current Liabilities. >1.5 is healthy."""
    data = get_financials(company_key, year)
    if not data:
        return {"error": f"No financial data available for {company_key} {year}"}

    ratio = simple_eval(
        "current_assets / current_liabilities",
        names={"current_assets": data["current_assets"], "current_liabilities": data["current_liabilities"]},
    )
    return {
        "company": company_key,
        "year": year,
        "current_ratio": round(ratio, 2),
        "healthy": ratio > 1.5,
        "interpretation": (
            f"Current ratio of {round(ratio,2)} means {company_key.upper()} has "
            f"${round(ratio,2)} in current assets for every $1 of current liabilities. "
            f"{'This is healthy.' if ratio > 1.5 else 'This signals potential liquidity stress.'}"
        ),
    }


def debt_to_equity(company_key: str, year: str) -> dict:
    """Debt-to-Equity = Total Liabilities / (Total Assets - Total Liabilities)."""
    data = get_financials(company_key, year)
    if not data:
        return {"error": f"No financial data available for {company_key} {year}"}

    equity = data["total_assets"] - data["total_liabilities"]
    if equity <= 0:
        return {"error": "Negative or zero equity — debt-to-equity not meaningful"}

    ratio = simple_eval(
        "total_liabilities / equity",
        names={"total_liabilities": data["total_liabilities"], "equity": equity},
    )
    return {
        "company": company_key,
        "year": year,
        "debt_to_equity": round(ratio, 2),
        "interpretation": (
            f"D/E ratio of {round(ratio,2)} means {company_key.upper()} carries "
            f"${round(ratio,2)} of debt for every $1 of equity."
        ),
    }


# ── Tool registry — exposed to the LangGraph agent ──────────────────────────

AVAILABLE_CALCULATIONS = {
    "altman_z_score": altman_z_score,
    "current_ratio":  current_ratio,
    "debt_to_equity": debt_to_equity,
}


def run_calculation(calc_name: str, company_key: str, year: str) -> dict:
    """Dispatch to the requested calculation function."""
    fn = AVAILABLE_CALCULATIONS.get(calc_name)
    if not fn:
        return {"error": f"Unknown calculation: {calc_name}. Available: {list(AVAILABLE_CALCULATIONS.keys())}"}
    return fn(company_key, year)
