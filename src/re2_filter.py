"""RE2 exclusions and removal of prompt instructions and worked examples."""

import re

from re2_inspection import input_text


def remove_prompts(rows):
    """Return id/code/input/output records, retaining task and helper code."""
    cleaned = []
    for row in rows:
        prompt = row["messages"][0]["content"].rsplit("[/EXAMPLE]", 1)[-1]
        main, _, helpers = prompt.partition("Functions called during the execution:")
        blocks = re.findall(r"\[PYTHON\](.*?)\[/PYTHON\]", main, re.S)
        if not blocks:
            raise ValueError(f"Missing task code: {row['id']}")
        code = blocks[-1].strip()
        helper_blocks = re.findall(r"\[PYTHON\](.*?)\[/PYTHON\]", helpers, re.S)
        if helper_blocks:
            code += "\n\n" + "\n\n".join(b.strip() for b in helper_blocks)
        cleaned.append({"id": row["id"], "code": code,
                        "input": input_text(row), "output": row["target_output"]})
    return cleaned

PROBLEMATIC_CASES = {
    "matplotlib@@matplotlib__api_deprecation.py@@deprecated_L97": "Unpredictable function address",
    "matplotlib@@matplotlib__api_deprecation.py@@delete_parameter_L314": "Unpredictable function address",
    "sympy@@sympy_multipledispatch_dispatcher.py@@register_L120": "Unpredictable function address",
    "matplotlib@@matplotlib___init__.py@@_preprocess_data_L1432": "Unpredictable function address",
    "matplotlib@@matplotlib_colors.py@@_repr_png__L2115": "Missing pixels in abbreviated input array",
    "matplotlib@@matplotlib_colors.py@@_repr_html__L2140": "Missing pixels in abbreviated input array",
    "scikit-learn@@sklearn_ensemble__bagging.py@@predict_log_proba_L1055": "Missing fitted estimator state",
    "ClassEval_49@JobMarketplace.withdraw_resume": "Empty resume list raises an exception; reference is null",
    "ClassEval_95@Warehouse.change_order_status": "Integer ID and string dictionary key disagree with reference",
    "ClassEval_95@Warehouse.create_order": "Integer ID and string dictionary key disagree with reference",
}


def filter_re2(rows):
    """Return retained rows and exclusions, using source tags and explicit IDs."""
    kept, excluded = [], []
    for row in rows:
        overlap = row["source_dataset"] == "cruxeval"
        problem = PROBLEMATIC_CASES.get(row["id"])
        if overlap or problem:
            excluded.append({"id": row["id"],
                             "reason": "cruxeval_overlap" if overlap else "problematic_sample",
                             "evidence": "CRUXEval source" if overlap else problem})
        else:
            kept.append(row)
    return kept, excluded
