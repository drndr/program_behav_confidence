# Step-by-step rebuild

Started: 10 September 2026.

This checklist records the sequence for rebuilding the project. Archiving does not launch experiments or freeze the remaining research choices.

| Step | Deliverable | Status |
|---|---|---|
| 0. Preserve previous work | Dated local snapshot, file hashes, Git-state record, post-move verification | Complete |
| 1. Agree on apparatus | Compact table of datasets, exact model variants, generator/judge settings, properties, and experiments | Broad scope accepted; details clarified when needed |
| 2. Define data and scoring | Minimal record schema; dataset provenance; explicit label policy; RE2 inclusion rules | In progress: modular layout, local raw copies, loaders, and inspection notebook; scoring and eligibility pending |
| 3. Build the smallest working pipeline | One dataset loader, one provider integration, answer extraction, correctness scoring, and a fixed-answer confidence call | Pending |
| 4. Check feasibility | Small named development sample; errors, confidence availability, truncation, reasoning controls, cost, and precision assessment | Pending |
| 5. Freeze the main protocol | Property definitions, model versions, repetitions, split groups, analysis contrasts, cost caps, and validation scope | Pending |
| 6. Collect and analyse | Main-study records, reproducible analyses, figures, and manuscript | Pending |

Reuse archived components selectively after inspecting their assumptions. Historical pilot results remain historical; do not silently mix them with new-model main-study results. Build a new dependency environment when the implementation requires one.

Do not reinstate every old document or script merely to reproduce the previous directory structure. Keep each step small enough to inspect before expanding the project.
