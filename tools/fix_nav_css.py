"""Fix nav CSS selectors to use actual Streamlit DOM class names."""
import subprocess, sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parent.parent
path = ROOT / "app18.py"

with open(path, encoding="utf-8") as f:
    text = f.read()

# ─── Remove the old (broken aria-label) nav CSS block we appended ─────────────
OLD_CSS = """\
/* ── Category nav bar (top row) ─────────────────────────────────── */
div[data-testid="stRadio"][aria-label="category"] > div,
div[data-testid="stRadio"][data-testid*="_nav_cat"] > div {
    gap: 0 !important;
    border-bottom: 2px solid #e0e0e0;
    padding-bottom: 0;
    margin-bottom: 4px;
}
div[data-testid="stRadio"][aria-label="category"] label {
    background: transparent !important;
    border: none !important;
    border-bottom: 3px solid transparent;
    border-radius: 0 !important;
    padding: 10px 22px 8px;
    font-size: 14px;
    font-weight: 600;
    color: #666;
    cursor: pointer;
    margin-bottom: -2px;
}
div[data-testid="stRadio"][aria-label="category"] label:hover {
    color: #1f77b4;
    background: rgba(31,119,180,0.04) !important;
}
div[data-testid="stRadio"][aria-label="category"] label[data-baseweb="radio"]:has(input:checked),
div[data-testid="stRadio"][aria-label="category"] label:has(input:checked) {
    color: #1f77b4 !important;
    border-bottom-color: #1f77b4 !important;
    background: transparent !important;
}
/* ── Sub-nav row (second row) ───────────────────────────────────── */
div[data-testid="stRadio"][aria-label="page"] > div {
    gap: 4px;
    padding: 4px 0 8px;
}
div[data-testid="stRadio"][aria-label="page"] label {
    background: #f5f7fa !important;
    border: 1px solid #dde1e7 !important;
    border-radius: 6px !important;
    padding: 5px 16px;
    font-size: 13px;
    font-weight: 500;
    color: #444;
    cursor: pointer;
}
div[data-testid="stRadio"][aria-label="page"] label:hover {
    border-color: #1f77b4 !important;
    color: #1f77b4;
    background: rgba(31,119,180,0.05) !important;
}
div[data-testid="stRadio"][aria-label="page"] label:has(input:checked) {
    background: #1f77b4 !important;
    color: #fff !important;
    border-color: #1f77b4 !important;
    font-weight: 600;
}"""

NEW_CSS = """\
/* ══════════════════════════════════════════════════════════════════
   TWO-LEVEL NAVIGATION
   Category bar  →  .st-key-_nav_cat
   Sub-nav row   →  [class*="st-key-_sub_"]
   ══════════════════════════════════════════════════════════════════ */

/* Hide the auto-generated "category" / "page" label text */
.st-key-_nav_cat > label:first-child,
[class*="st-key-_sub_"] > label:first-child { display: none !important; }

/* ── Category bar: underline-tab style ────────────────────────── */
.st-key-_nav_cat [data-testid="stRadio"] > div {
    gap: 0 !important;
    border-bottom: 2px solid #e0e0e0;
    padding-bottom: 0;
    margin-bottom: 2px;
    flex-wrap: nowrap !important;
}
.st-key-_nav_cat [data-testid="stRadio"] label {
    background: transparent !important;
    border: none !important;
    border-bottom: 3px solid transparent !important;
    border-radius: 0 !important;
    padding: 10px 24px 9px !important;
    font-size: 14px !important;
    font-weight: 600 !important;
    color: #555 !important;
    cursor: pointer;
    margin-bottom: -2px;
    transition: color 0.15s, border-color 0.15s;
}
.st-key-_nav_cat [data-testid="stRadio"] label:hover {
    color: #1f77b4 !important;
    background: rgba(31,119,180,0.04) !important;
}
.st-key-_nav_cat [data-testid="stRadio"] label:has(input:checked) {
    color: #1f77b4 !important;
    border-bottom-color: #1f77b4 !important;
    background: transparent !important;
}
/* Hide the radio circle dot in category bar */
.st-key-_nav_cat [data-testid="stRadio"] [data-testid="stWidgetLabel"],
.st-key-_nav_cat [data-testid="stRadio"] div[data-baseweb="radio"] > div:first-child {
    display: none !important;
}

/* ── Sub-nav: filled-pill style ───────────────────────────────── */
[class*="st-key-_sub_"] [data-testid="stRadio"] > div {
    gap: 6px !important;
    padding: 6px 0 10px !important;
    flex-wrap: wrap;
}
[class*="st-key-_sub_"] [data-testid="stRadio"] label {
    background: #f0f2f6 !important;
    border: 1px solid #d4d8e0 !important;
    border-radius: 20px !important;
    padding: 4px 18px !important;
    font-size: 13px !important;
    font-weight: 500 !important;
    color: #444 !important;
    cursor: pointer;
    transition: all 0.12s;
}
[class*="st-key-_sub_"] [data-testid="stRadio"] label:hover {
    border-color: #1f77b4 !important;
    color: #1f77b4 !important;
    background: rgba(31,119,180,0.06) !important;
}
[class*="st-key-_sub_"] [data-testid="stRadio"] label:has(input:checked) {
    background: #1f77b4 !important;
    color: #fff !important;
    border-color: #1f77b4 !important;
    font-weight: 600 !important;
}
/* Hide radio dot in sub-nav pills too */
[class*="st-key-_sub_"] [data-testid="stRadio"] div[data-baseweb="radio"] > div:first-child {
    display: none !important;
}"""

if OLD_CSS in text:
    text = text.replace(OLD_CSS, NEW_CSS)
    print("✓ Replaced old aria-label CSS with correct class-based CSS")
elif NEW_CSS in text:
    print("  Already up to date, no change needed")
else:
    # Append before </style>
    text = text.replace("</style>", NEW_CSS + "\n      </style>", 1)
    print("✓ Appended new CSS before </style>")

with open(path, "w", encoding="utf-8") as f:
    f.write(text)

result = subprocess.run([sys.executable, "-m", "py_compile", str(path)],
                        capture_output=True, text=True)
print("✓ SYNTAX OK" if result.returncode == 0 else "✗ ERROR:\n" + result.stderr)
print(f"Lines: {len(text.splitlines())}")
