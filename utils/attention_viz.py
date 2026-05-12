"""
Utility: render coloured token attention HTML for display in Streamlit.
"""

from __future__ import annotations


def render_token_attention(tokens: list[str],
                           attention: list[float],
                           colormap: str = "green") -> str:
    """
    Render tokens with background opacity proportional to attention weight.

    Args:
        tokens:    List of token strings.
        attention: Normalised attention weights (0–1) aligned with tokens.
        colormap:  'green' (positive sentiment) | 'red' | 'blue'

    Returns:
        HTML string safe to pass to st.markdown(html, unsafe_allow_html=True).
    """
    COLOR_MAP = {
        "green": (76, 175, 80),
        "red":   (229, 57, 53),
        "blue":  (33, 150, 243),
    }
    r, g, b = COLOR_MAP.get(colormap, COLOR_MAP["green"])

    if not attention:
        return "<p>" + " ".join(tokens) + "</p>"

    spans = []
    for tok, att in zip(tokens, attention):
        opacity  = max(0.05, min(1.0, att))
        bg_color = f"rgba({r},{g},{b},{opacity:.2f})"
        border   = f"1px solid rgba({r},{g},{b},0.4)" if att > 0.5 else "none"
        style    = (f"background:{bg_color};border:{border};"
                    f"border-radius:4px;padding:2px 5px;margin:1px;"
                    f"display:inline-block;font-size:13px;"
                    f"cursor:default")
        title    = f"attention: {att:.3f}"
        spans.append(f'<span style="{style}" title="{title}">{tok}</span>')

    return (
        '<div style="line-height:2.2;padding:8px 0">'
        + " ".join(spans)
        + "</div>"
    )
