import subprocess
import os
import re

MD_PATH = "docs/project-report/TabSynth_AI_Project_Report.md"
HTML_PATH = "docs/project-report/TabSynth_AI_Project_Report.html"
DOCX_PATH = "docs/project-report/TabSynth_AI_Project_Report.docx"
PDF_PATH = "docs/project-report/TabSynth_AI_Project_Report.pdf"

with open(MD_PATH) as f:
    md_content = f.read()

# Build a clean HTML rendering
html_lines = []
html_lines.append("""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>TabSynth AI: Research Project Report</title>
<style>
body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; line-height: 1.6; color: #1e293b; max-width: 900px; margin: 0 auto; padding: 40px 20px; }
h1 { color: #0f172a; font-size: 26pt; border-bottom: 2px solid #8b5cf6; padding-bottom: 12px; margin-top: 30px; }
h2 { color: #1e1b4b; font-size: 18pt; border-bottom: 1px solid #cbd5e1; padding-bottom: 6px; margin-top: 26px; }
h3 { color: #312e81; font-size: 14pt; margin-top: 20px; }
h4 { color: #4338ca; font-size: 12pt; margin-top: 16px; }
p { margin: 12px 0; font-size: 11pt; }
table { border-collapse: collapse; width: 100%; margin: 18px 0; font-size: 10pt; }
th, td { border: 1px solid #cbd5e1; padding: 8px 12px; text-align: left; }
th { background-color: #f1f5f9; font-weight: bold; color: #0f172a; }
tr:nth-child(even) { background-color: #f8fafc; }
pre, code { font-family: ui-monospace, Menlo, Monaco, "Courier New", monospace; font-size: 9.5pt; background-color: #f1f5f9; }
pre { padding: 14px; border-radius: 6px; overflow-x: auto; border: 1px solid #e2e8f0; }
blockquote { border-left: 4px solid #8b5cf6; padding-left: 14px; color: #475569; margin: 16px 0; background: #faf5ff; padding-top: 6px; padding-bottom: 6px; }
hr { border: none; border-top: 1px solid #e2e8f0; margin: 30px 0; }
img { max-width: 100%; height: auto; border-radius: 4px; border: 1px solid #e2e8f0; margin: 12px 0; }
</style>
</head>
<body>
""")

in_code_block = False
code_buffer = []

in_table = False
table_buffer = []

lines = md_content.split("\n")
for line in lines:
    if line.startswith("```"):
        if in_code_block:
            in_code_block = False
            html_lines.append("<pre><code>" + "\n".join(code_buffer) + "</code></pre>")
            code_buffer = []
        else:
            in_code_block = True
            code_buffer = []
        continue
    
    if in_code_block:
        escaped = line.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        code_buffer.append(escaped)
        continue

    # Tables
    if line.strip().startswith("|") and line.strip().endswith("|"):
        cells = [c.strip() for c in line.strip()[1:-1].split("|")]
        if not in_table:
            in_table = True
            table_buffer = ["<table><thead><tr>" + "".join(f"<th>{c}</th>" for c in cells) + "</tr></thead><tbody>"]
        elif re.match(r"^[:\-\s\|]+$", line.strip()):
            pass # separator line
        else:
            table_buffer.append("<tr>" + "".join(f"<td>{c}</td>" for c in cells) + "</tr>")
        continue
    else:
        if in_table:
            in_table = False
            table_buffer.append("</tbody></table>")
            html_lines.append("\n".join(table_buffer))
            table_buffer = []

    # Headers
    if line.startswith("# "):
        html_lines.append(f"<h1>{line[2:].strip()}</h1>")
    elif line.startswith("## "):
        html_lines.append(f"<h2>{line[3:].strip()}</h2>")
    elif line.startswith("### "):
        html_lines.append(f"<h3>{line[4:].strip()}</h3>")
    elif line.startswith("#### "):
        html_lines.append(f"<h4>{line[5:].strip()}</h4>")
    elif line.startswith("> "):
        html_lines.append(f"<blockquote>{line[2:].strip()}</blockquote>")
    elif line.startswith("---"):
        html_lines.append("<hr/>")
    elif line.strip() == "":
        continue
    else:
        # Process bold, italic, code
        processed = line
        processed = re.sub(r"\*\*(.*?)\*\*", r"<strong>\1</strong>", processed)
        processed = re.sub(r"\*(.*?)\*", r"<em>\1</em>", processed)
        processed = re.sub(r"`(.*?)`", r"<code>\1</code>", processed)
        html_lines.append(f"<p>{processed}</p>")

if in_table:
    table_buffer.append("</tbody></table>")
    html_lines.append("\n".join(table_buffer))

html_lines.append("</body></html>")

with open(HTML_PATH, "w") as f:
    f.write("\n".join(html_lines))
print(f"Generated HTML: {HTML_PATH}")

# Convert to DOCX via textutil
cmd_docx = ["textutil", "-convert", "docx", HTML_PATH, "-output", DOCX_PATH]
res_docx = subprocess.run(cmd_docx, capture_output=True, text=True)
if res_docx.returncode == 0:
    print(f"Successfully generated DOCX: {DOCX_PATH}")
else:
    print(f"DOCX generation failed: {res_docx.stderr}")

# Generate PDF via cupsfilter
with open(PDF_PATH, "wb") as pdf_out:
    res_pdf = subprocess.run(["/usr/sbin/cupsfilter", MD_PATH], stdout=pdf_out, stderr=subprocess.PIPE)
    if res_pdf.returncode == 0:
        print(f"Successfully generated PDF: {PDF_PATH}")
    else:
        print(f"PDF generation failed: {res_pdf.stderr.decode('utf-8', errors='ignore')}")

