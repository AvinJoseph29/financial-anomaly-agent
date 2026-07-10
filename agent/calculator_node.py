"""
Calculator Node
===============
Handles the "calculate" route. Calls the sandboxed financial calculator tool
with the company/year/calc_type extracted by the planner.
"""

from tools.calculator import run_calculation
from agent.state import AgentState


def calculator_node(state: AgentState) -> dict:
    company_key = state.get("company_key")
    year        = state.get("year")
    calc_type   = state.get("calc_type") or "altman_z_score"

    if not company_key or not year:
        return {
            "calc_result": {
                "error": (
                    "Calculation requires both a company and a year. "
                    f"Got company={company_key}, year={year}. "
                    "Please specify both, e.g. 'Altman Z-score for Enron in 2000'."
                )
            }
        }

    result = run_calculation(calc_type, company_key, year)
    return {"calc_result": result}
