---
trigger: always_on
---

🛑 ANTI-RUSHING DIRECTIVE (THE METHODOLOGY RULE)

1. Comprehension Over Completion:

Never rush to the next phase. The primary goal is deep, empirical understanding of the data and the models, not just generating outputs. If a script produces a result, you must stop, analyze the result, and explain why it happened before writing the next script.

2. The Gate System (Build → Measure → Explain): At every phase, you must:

Build the component meticulously, ensuring the code is structurally perfect, handles edge cases, and scales properly.
Measure its performance using a rigorous diagnostic script (e.g., recall, precision, memory usage).
Explain the metric empirically.
Do not advance to the next phase until the current phase is fully validated, understood, and clears its diagnostic gate.

3. No "Throwing Code Over the Wall":

Do not hand the user multiple heavy scripts to run simultaneously without first verifying they work optimally on a local subset. Respect hardware limits, memory constraints, and the user's time.