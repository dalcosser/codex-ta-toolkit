"""
Replaces the flat 14-item radio nav in app18.py with a clean
two-level grouped layout:
  Category bar  →  Charts | Scans | Market | AI Research
  Sub-nav row   →  items within that group

All existing  if nav == '...'  blocks are untouched.
Run:  python tools/nav_restructure.py
"""
import re, subprocess, sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parent.parent
path = ROOT / "app18.py"

with open(path, encoding="utf-8") as f:
    text = f.read()

orig_lines = len(text.splitlines())

# ─── 1. Replace the nav radio block ──────────────────────────────────────────
OLD_NAV = '''\
nav = st.radio(
    "Navigation",
    [
        "Chart", "Options", "TradingView", "GPT-5 Agent",
        "MS Analysis", "JPM Earnings", "GS Fundamental",
        "Scans", "Premarket", "Calendar",
        "Scanners", "Movers", "Overnight", "Signal Scanner",
    ],
    horizontal=True,
    index=0,
    key="top_nav",
    label_visibility="collapsed",
)
st.markdown("---")'''

NEW_NAV = '''\
# ── Two-level grouped navigation ─────────────────────────────────────────────
_NAV_GROUPS = {
    "📈 Charts":      ["Chart", "TradingView"],
    "🔍 Scans":       ["Scans", "Signal Scanner", "Scanners", "Overnight"],
    "📊 Market":      ["Options", "Premarket", "Movers", "Calendar"],
    "🤖 AI Research": ["GPT-5 Agent", "MS Analysis", "JPM Earnings", "GS Fundamental"],
}

# Restore last active group/page across reruns
_default_cat = st.session_state.get("_nav_cat", "📈 Charts")
if _default_cat not in _NAV_GROUPS:
    _default_cat = "📈 Charts"

_cat = st.radio(
    "category",
    list(_NAV_GROUPS.keys()),
    index=list(_NAV_GROUPS.keys()).index(_default_cat),
    horizontal=True,
    key="_nav_cat",
    label_visibility="collapsed",
)

# Sub-nav for the selected category
_sub_key  = f"_sub_{_cat}"
_sub_opts = _NAV_GROUPS[_cat]
_sub_default = st.session_state.get(_sub_key, _sub_opts[0])
if _sub_default not in _sub_opts:
    _sub_default = _sub_opts[0]

nav = st.radio(
    "page",
    _sub_opts,
    index=_sub_opts.index(_sub_default),
    horizontal=True,
    key=_sub_key,
    label_visibility="collapsed",
)
try:
    st.session_state["nav"] = nav
except Exception:
    pass
st.markdown("---")
# ─────────────────────────────────────────────────────────────────────────────'''

if OLD_NAV in text:
    text = text.replace(OLD_NAV, NEW_NAV)
    print("✓ nav radio replaced with two-level grouped nav")
else:
    print("✗ OLD_NAV not found — check exact text")
    sys.exit(1)

# ─── 2. Upgrade the CSS block in apply_theme() ───────────────────────────────
# Find the existing nav-pill CSS we added and replace with grouped-tab CSS
OLD_CSS = '''\
/* Navigation radio as pill buttons */
div[data-testid="stHorizontalBlock"] .stRadio > div {
    flex-wrap: wrap;
    gap: 4px;
}
div[data-testid="stHorizontalBlock"] .stRadio label {
    background: #f0f2f6;
    border-radius: 20px;
    padding: 4px 14px;
    font-size: 13px;
    cursor: pointer;
    border: 1px solid transparent;
}
div[data-testid="stHorizontalBlock"] .stRadio label:hover {
    border-color: #1f77b4;
    color: #1f77b4;
}'''

NEW_CSS = '''\
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
}'''

if OLD_CSS in text:
    text = text.replace(OLD_CSS, NEW_CSS)
    print("✓ CSS updated for two-level nav styling")
else:
    # Try to append new CSS before closing </style>
    if NEW_CSS not in text and '</style>' in text:
        text = text.replace('</style>', NEW_CSS + '\n      </style>', 1)
        print("✓ CSS appended (old block not found, injected before </style>)")
    else:
        print("⚠ CSS not updated — may already be present or </style> not found")

# ─── 3. Fix session_state nav reference (already handled in NEW_NAV) ─────────
# Remove old  try: st.session_state['nav'] = nav  if it's now a duplicate
old_dup = "try:\n    st.session_state['nav'] = nav\nexcept Exception:\n    pass"
count = text.count(old_dup)
if count > 1:
    # Remove all but the first
    idx = text.find(old_dup)
    rest = text[idx + len(old_dup):]
    text = text[:idx + len(old_dup)] + rest.replace(old_dup, "", count - 1)
    print(f"✓ Removed {count-1} duplicate session_state nav assignments")

# ─── 4. Collapse excessive blank lines ───────────────────────────────────────
text = re.sub(r'\n{4,}', '\n\n\n', text)

# ─── Write & verify ───────────────────────────────────────────────────────────
with open(path, "w", encoding="utf-8") as f:
    f.write(text)

print(f"\nLines: {orig_lines} → {len(text.splitlines())}")

result = subprocess.run(
    [sys.executable, "-m", "py_compile", str(path)],
    capture_output=True, text=True,
)
if result.returncode == 0:
    print("✓ SYNTAX OK")
else:
    print("✗ SYNTAX ERROR:\n" + result.stderr)
