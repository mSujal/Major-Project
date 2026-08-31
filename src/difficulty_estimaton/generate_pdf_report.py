import json
import argparse
import os
import subprocess
import shutil

def escape_latex(text):
    text = text.replace('\\', r'\BACKSLASHPLACEHOLDER')
    replacements = {
        '&': r'\&',
        '%': r'\%',
        '$': r'\$',
        '#': r'\#',
        '_': r'\_',
        '{': r'\{',
        '}': r'\}',
        '~': r'\textasciitilde{}',
        '^': r'\^{}',
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    text = text.replace(r'\BACKSLASHPLACEHOLDER', r'\textbackslash{}')
    return text

def generate_latex(detailed):
    header = r"""
\documentclass[9pt]{article}
\usepackage[margin=0.6in]{geometry}
\usepackage{enumitem}
\usepackage{array}
\usepackage{xcolor}
\usepackage{hyperref}
\usepackage{booktabs}
\usepackage{fancyhdr}
\usepackage{titlesec}

\hypersetup{colorlinks=true, linkcolor=blue}
\pagestyle{fancy}
\fancyhf{}
\fancyhead[L]{\textbf{Question Difficulty Validation Report}}
\fancyhead[R]{\thepage}
\fancyfoot[C]{}

\titleformat{\section}{\small\bfseries}{}{0em}{}
\setlist[enumerate]{label=\alph.), leftmargin=*, itemsep=0pt, topsep=0pt, parsep=0pt}
\setlength{\parindent}{0pt}
\setlength{\parskip}{3pt}

\begin{document}
\begin{center}
{\Large \textbf{Question Difficulty Validation Report}} \\[4pt]
{\small \# of Questions: """ + str(len(detailed)) + r"""} \\[4pt]
\noindent\rule{\linewidth}{0.4pt}
\end{center}
\vspace{0.3cm}
"""

    footer = r"\end{document}"

    body = []
    for i, q in enumerate(detailed):
        q_text = escape_latex(q.get("question", "MISSING"))
        correct = q.get("correct_answer", "?")
        bloom = q.get("bloom_level", "Unknown")
        emp = q.get("empirical_difficulty", "Unknown")
        comp = q.get("completion_time", 0.0)
        tokens = q.get("tokens", 0)

        opts = q.get("options", {})
        options_lines = ""
        for label in sorted(opts.keys()):
            opt = escape_latex(opts[label])
            if label == correct:
                options_lines += f"    \\item \\textbf{{{label}) {opt}}} % CORRECT\n"
            else:
                options_lines += f"    \\item {label}) {opt}\n"

        body.append(f"""
\\section*{{Question {i+1}}}

\\textbf{{Q:}} {q_text}

\\begin{{enumerate}}[label=\\textbf{{\\alph{{*}})}}, nosep]
{options_lines}
\\end{{enumerate}}

\\begin{{tabular}}{{@{{}}l p{{10.5cm}}@{{}}}}
\\textbf{{Answer:}} & \\textbf{{{correct}}} \\hspace{{1em}} Bloom: {bloom} \\hspace{{1em}} Comp: {comp:.3f}s \\hspace{{1em}} Tokens: {tokens} \\\\
\\textbf{{Empirical:}} & \\textbf{{{emp}}} \\hspace{{1em}} \\textbf{{Actual (Human):}} \\underline{{\\hspace{{5cm}}}}
\\end{{tabular}}

\\vspace{{0.2cm}}
\\rule{{\\textwidth}}{{0.4pt}}
\\vspace{{0.2cm}}
""")

    return header + "\n".join(body) + footer

def compile_pdf(tex_path, pdf_path):
    tex_dir = os.path.dirname(tex_path)
    for _ in range(2):
        proc = subprocess.run(
            ["pdflatex", "-interaction=nonstopmode", "-output-directory=" + tex_dir, tex_path],
            capture_output=True,
            text=False
        )
        if not os.path.exists(os.path.splitext(tex_path)[0] + ".pdf"):
            return False
    generated = os.path.splitext(tex_path)[0] + ".pdf"
    if os.path.exists(generated):
        if pdf_path != generated:
            os.rename(generated, pdf_path)
        return True
    return False

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output")
    args = parser.parse_args()

    with open(args.input, 'r', encoding='utf-8') as f:
        data = json.load(f)

    detailed = data.get("detailed_results", [])
    if not detailed:
        print("No 'detailed_results' found.")
        return

    print(f"Loaded {len(detailed)} questions.")

    out_path = args.output or os.path.splitext(args.input)[0] + ".pdf"
    if out_path.endswith('.tex'):
        out_path = out_path[:-4] + '.pdf'

    tex_path = os.path.splitext(out_path)[0] + ".tex"

    print("Generating LaTeX source...")
    tex_content = generate_latex(detailed)
    with open(tex_path, 'w', encoding='utf-8') as f:
        f.write(tex_content)

    if not shutil.which("pdflatex"):
        print("pdflatex not found. LaTeX source saved to:", tex_path)
        return

    print("Compiling PDF...")
    if compile_pdf(tex_path, out_path):
        print("PDF saved to:", out_path)
        base = os.path.splitext(out_path)[0]
        for ext in [".aux", ".log", ".out", ".tex"]:
            f = base + ext
            if os.path.exists(f):
                os.remove(f)
    else:
        print("PDF compilation failed. LaTeX source kept at:", tex_path)

if __name__ == "__main__":
    main()