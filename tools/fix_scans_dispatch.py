"""Remove duplicate dispatch1 (dead code) from Scans tab and fix filter guard.

Dispatch1 at lines 2647-2758 (0-indexed 2646-2757) is dead code — dispatch2
at lines 2759-2879 (0-indexed 2758-2878) immediately overwrites ev.
Additionally, dispatch2 ran at 16-space indent (outside the filter-passed
else: block at 16 spaces), so it ran even for tickers that failed filters.

Fix:
  1. Delete dispatch1 (0-indexed 2646-2757, i.e. lines 2647-2758 in file).
  2. Re-indent dispatch2+display (0-indexed 2758-2878) by +4 spaces so it
     becomes the body of the filter-passed else: at line 2646.
"""
import sys

APP = r"C:\Users\David Alcosser\Documents\Visual Code\codex_ta_toolkit\app18.py"

with open(APP, encoding="utf-8") as f:
    lines = f.readlines()

# Boundaries (all 0-indexed)
DISPATCH1_START = 2646   # first line of dispatch1 (old line 2647)
DISPATCH1_END   = 2758   # exclusive end of dispatch1 (old line 2759 = dispatch2 start)
DISPATCH2_END   = 2879   # exclusive end of dispatch2+display (old line 2880 = blank)

n_before = len(lines)

new_lines = (
    lines[:DISPATCH1_START] +                              # through else: at line 2646
    ["    " + l for l in lines[DISPATCH1_END:DISPATCH2_END]] +  # dispatch2+display +4 spaces
    lines[DISPATCH2_END:]                                  # rest of file
)

with open(APP, "w", encoding="utf-8") as f:
    f.writelines(new_lines)

removed = DISPATCH1_END - DISPATCH1_START
print(f"Done. Was {n_before} lines, now {len(new_lines)} lines.")
print(f"Removed {removed} lines of dead dispatch1 (old 2647-2758).")
print(f"Re-indented {DISPATCH2_END - DISPATCH1_END} lines of dispatch2+display (+4 spaces).")
