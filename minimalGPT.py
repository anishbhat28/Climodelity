import ast
from pathlib import Path

from openai import OpenAI

TRAIN_FILE = Path("train.py")
OUTPUT_FILE = Path("program.md")
MODEL_NAME = "gpt-5.4"

client = OpenAI()


def extract_train_context(train_path: Path) -> str:
    source = train_path.read_text(encoding="utf-8")
    lines = source.splitlines()
    tree = ast.parse(source)

    chunks = []
    constants = []
    keyword_lines = []

    relevant_name_keywords = [
        "train",
        "model",
        "optimizer",
        "forward",
        "loss",
        "eval",
        "validate",
        "attention",
        "block",
        "dataset",
        "loader",
        "dataloader",
        "batch",
        "config",
        "hyper",
        "token",
        "embed",
        "head",
        "mlp",
        "lr",
        "schedule",
    ]

    relevant_line_keywords = [
        "class ",
        "def ",
        "train(",
        "optimizer",
        "adam",
        "adamw",
        "muon",
        "learning_rate",
        "lr",
        "weight_decay",
        "dropout",
        "batch_size",
        "block_size",
        "sequence_length",
        "n_layer",
        "n_head",
        "n_embd",
        "dim",
        "embed",
        "vocab",
        "loss",
        "val",
        "eval",
        "device",
        "compile",
        "torch.compile",
        "mps",
        "cuda",
        "bpb",
        "bits per byte",
        "time budget",
    ]

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            name = node.name.lower()
            if any(keyword in name for keyword in relevant_name_keywords):
                start = getattr(node, "lineno", None)
                end = getattr(node, "end_lineno", None)
                if start and end:
                    snippet = "\n".join(lines[start - 1:end])
                    chunks.append(
                        f"\n### {type(node).__name__} {node.name} (lines {start}-{end}) ###\n{snippet}"
                    )

        elif isinstance(node, ast.Assign):
            start = getattr(node, "lineno", None)
            end = getattr(node, "end_lineno", None)
            if start and end:
                snippet = "\n".join(lines[start - 1:end])
                lowered = snippet.lower()
                if any(keyword in lowered for keyword in relevant_line_keywords):
                    constants.append(f"{start}: {snippet}")

    for i, line in enumerate(lines, start=1):
        lower = line.lower()
        if any(keyword in lower for keyword in relevant_line_keywords):
            keyword_lines.append(f"{i}: {line}")

    return (
        "TRAIN.PY KEYWORD-RELEVANT LINES\n"
        + "\n".join(keyword_lines[:200])
        + "\n\nTRAIN.PY RELEVANT ASSIGNMENTS / CONSTANTS\n"
        + ("\n".join(constants[:80]) if constants else "[No obvious relevant assignments found]")
        + "\n\nTRAIN.PY RELEVANT FUNCTIONS / CLASSES\n"
        + ("\n\n".join(chunks[:20]) if chunks else "[No obvious relevant functions found]")
    )


def build_prompt(train_context: str) -> str:
    return f"""
You are generating a file named program.md for an autonomous coding agent.

Goal:
Write a high-quality program.md that tells an AI research agent how to improve train.py through short experimental iterations.

Context:
The agent will edit train.py, run experiments, evaluate results, and keep or discard changes.
The human will edit program.md, not train.py.
Your job is to infer as much as possible from the provided train.py excerpts and produce instructions specialized to this codebase.

Requirements for your output:
- Output ONLY the raw Markdown contents of program.md
- Do NOT wrap the answer in markdown fences
- Do NOT include explanations before or after the markdown
- Be specific and actionable
- Assume the agent can read and edit train.py
- Assume the agent should preserve functionality and only make justified changes
- Prefer small, high-signal experiments over large rewrites
- Include clear experiment workflow instructions
- Include rules for measuring whether a change is better or worse
- Include rules for reverting bad changes
- Include guidance for logging experiments
- Mention the most likely optimization targets inferred from train.py
- Mention constraints inferred from train.py (device/platform/training loop/metric/time budget if visible)
- If something is ambiguous, state a conservative policy instead of inventing specifics

The program.md should be useful immediately and should ideally include sections such as:
- Mission
- What file to edit
- What not to edit
- How to run experiments
- What metric to optimize
- How to compare runs fairly
- Safe modification priorities
- Experiment loop
- Logging format
- Guardrails / anti-patterns
- First experiment ideas tailored to this train.py

Important:
- Do not tell the agent to ask the human for help
- Do not tell the agent to make huge refactors first
- Do not be generic if the train.py context provides clues
- If a validation metric is visible, prioritize it explicitly
- If the code suggests a fixed time budget, preserve fairness around that budget
- If optimizer/model architecture/hyperparameters are visible, suggest those as tunable levers

Here is the extracted train.py context:

{train_context}
"""


def main():
    train_context = extract_train_context(TRAIN_FILE)
    prompt = build_prompt(train_context)

    response = client.responses.create(
        model=MODEL_NAME,
        input=prompt,
    )

    program_md = response.output_text.strip()

    if program_md.startswith("```"):
        lines = program_md.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        program_md = "\n".join(lines).strip()

    OUTPUT_FILE.write_text(program_md, encoding="utf-8")
    print(f"Saved generated program.md to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()