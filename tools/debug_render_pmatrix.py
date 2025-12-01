# tools/debug_render_pmatrix.py
import tempfile, subprocess
from pathlib import Path

expr = r"\begin{pmatrix} a & b \\ c & d \end{pmatrix}"   # example that is failing
tex = r"""\documentclass[varwidth=true, border=2pt]{standalone}
\usepackage{amsmath,amssymb,amsfonts,bm}
\begin{document}
\[
%s
\]
\end{document}
""" % expr

td = Path(tempfile.mkdtemp())
tex_file = td / "eq.tex"
tex_file.write_text(tex, encoding="utf-8")
cmd = ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "-output-directory", str(td), str(tex_file)]
print("Running:", " ".join(cmd))
proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
print("Return code:", proc.returncode)
logfile = td / "eq.log"
if logfile.exists():
    print("pdflatex log:\n", logfile.read_text(errors="ignore"))
else:
    print("No eq.log found. stdout:\n", proc.stdout)
