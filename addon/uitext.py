"""Panel text that fits the sidebar: Blender cuts long labels and button text in the middle with an ellipsis.

Rules (worklog/scripts/textlint.py checks them, textcheck.sh finds cut text in screenshots of every panel):
- A label whose text isn't a fixed short string goes through label() here.
- Two or more buttons side by side go in a buttons() row, which stacks them when they don't fit.
- No "..." in panel text: the screenshot check takes any three dots for Blender's cut mark.
"""
import blf

# Sidebar room a label doesn't get, measured at UI scale 1 in a 5.3 sidebar: panel margins and the tab column, plus
# an icon's column. A box inside the panel takes another BOX_INSET.
MARGIN = 56
ICON = 24
BOX_INSET = 12
BUTTON = 26  # an icon-only button at the end of the row


def _scale(context):
    return context.preferences.system.ui_scale


def room(context, icon=False, boxed=0, buttons=0):
    """Pixels a label's text gets in the current region."""
    scale = _scale(context)
    width = context.region.width if context.region else 300
    return width - (MARGIN + (ICON if icon else 0) + BOX_INSET * boxed + BUTTON * buttons) * scale


def text_width(context, text):
    blf.size(0, context.preferences.ui_styles[0].widget.points * _scale(context))
    return blf.dimensions(0, text)[0]


def fits(context, text, icon=False, boxed=0, buttons=0):
    return text_width(context, text) <= room(context, icon, boxed, buttons)


def wrap(context, text, width):
    lines = []
    for para in text.split('\n'):
        para_lines = []
        for word in para.split():
            if para_lines and text_width(context, para_lines[-1] + ' ' + word) <= width:
                para_lines[-1] += ' ' + word
                continue
            # a path or other word wider than the panel breaks wherever it has to
            while text_width(context, word) > width and len(word) > 1:
                cut = len(word) - 1
                while cut > 1 and text_width(context, word[:cut]) > width:
                    cut -= 1
                para_lines.append(word[:cut])
                word = word[cut:]
            para_lines.append(word)
        lines.extend(para_lines or [''])
    return lines


def buttons(context, layout, *texts, boxed=0, align=False):
    """A row for buttons with these texts (each with an icon) if they fit side by side, else a column."""
    scale = _scale(context)
    # each button: its text, an icon and the button's own padding
    need = sum(text_width(context, t) + (ICON + 14) * scale for t in texts)
    if need <= room(context, boxed=boxed):
        return layout.row(align=align)
    return layout.column(align=align)


def label(context, layout, text, icon='NONE', boxed=0, buttons=0):
    """A label wrapped onto as many lines as it needs. Returns the first line's row, so a caller can put `buttons`
    icon-only buttons after it."""
    has_icon = icon != 'NONE'
    lines = wrap(context, text, room(context, has_icon, boxed, buttons))
    col = layout.column(align=True)
    first = None
    for i, line in enumerate(lines):
        row = col.row(align=True)
        row.label(text=line, icon=icon if i == 0 else 'BLANK1' if has_icon else 'NONE')
        first = first or row
    return first
