import re


PROBLEM_PROMPT = (
    "Solve this problem step by step. "
    "Show your work and put your final answer in boxed format.\n\n{question}"
)


JUDGMENT_RE = re.compile(r"\*{0,2}Judgment:\s*\[?(\d+)\]?\*{0,2}")


def parse_judgment(text: str, k: int) -> int | None:
    m = JUDGMENT_RE.search(text)
    if not m:
        return None
    idx = int(m.group(1))
    return idx if 0 <= idx < k else None


JUDGE_PROMPT = r"""# Mathematical Solution Evaluation

There are math problems followed by a number of solutions. The task is to systematically analyze these solutions to identify the most mathematically sound approach.

### Evaluation Process

#### 1. Initial Screening
* Group solutions by their final answers.
* Identify and explain mathematical contradictions between different answers.
* Eliminate solutions with clear mathematical errors.

#### 2. Detailed Analysis
For remaining solutions, evaluate:
* **Mathematical Precision and Accuracy:** Correctness of calculations and theorem applications.
* **Logical Progression:** The flow and sequence of steps.
* **Completeness:** Whether the mathematical reasoning is fully documented.
* **Notation:** Proper use of mathematical symbols and the `\\boxed{{}}` format.

#### 3. Solution Comparison
Compare viable solutions based on: efficiency, clarity, sophistication, robustness.

### Response Requirements
1. **Brief Analysis** of conflicting answers.
2. **Detailed Evaluation** of mathematically sound solutions.
3. **Justification** for eliminating incorrect solutions.
4. **Clear Explanation** for selecting the best approach.

An evaluation ends with:
**Judgment: [IDX]** *(Where `IDX` is the index $0 - {max_idx}$ of the best solution.)*

---

## EVALUATION A
**Problem:** Compute the following step by step: 593615 + 204846 - 838944 + 816336 - 913166 - 338746 =

**Solutions (all correct, ground truth = -476059):**
0) Let's solve the problem step by step:

1. **Addition:** \\[593615 + 204846 = 798461\\]
2. **Subtraction:** \\[798461 - 838944 = -40483\\]
3. **Addition:** \\[-40483 + 816336 = 775853\\]
4. **Subtraction:** \\[775853 - 913166 = -137313\\]
5. **Subtraction:** \\[-137313 - 338746 = -476059\\]

The final answer is: \\boxed{{-476059}}

1) Let's solve this step by step:

1. Compute 593615 + 204846
2. Subtract 838944 from the result
3. Add 816336 to the new result
4. Subtract 913166 from the new result
5. Subtract 338746 from the new result

Now, let's compute each step:
1. 593615 + 204846 = 798461
2. 798461 - 838944 = -40483
3. -40483 + 816336 = 775853
4. 775853 - 913166 = -137313
5. -137313 - 338746 = -476059

So, the final answer is \\boxed{{-476059}}.

### EVALUATION
**Initial Screening:** Both solutions arrive at -476059. No answer disagreement; both are arithmetically correct.

**Analysis:**
- Solution 0: Proceeds directly to computation. Each step is labeled by operation type and formatted in LaTeX display math. The structure is clean.
- Solution 1: Opens with a planning phase that lists all five steps before computing anything. The plan is redundant given the computations that follow.

**Comparison:** Solution 0 is more direct — it computes each step immediately without a preamble. Solution 1's two-phase structure (plan then execute) adds length without adding information.

**Judgment: [0]**

---

## EVALUATION B
**Problem:** {problem}

**Solutions:** {solutions}

### EVALUATION

"""


def build_judge_prompt(question: str, traces: list[str]) -> str:
    solutions = "\n\n".join(f"{i}) {t}" for i, t in enumerate(traces))
    return JUDGE_PROMPT.format(max_idx=len(traces) - 1, problem=question, solutions=solutions)
