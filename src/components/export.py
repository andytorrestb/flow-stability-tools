from pathlib import Path
from typing import Dict

from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
import io
import matplotlib.pyplot as plt
import sympy as sp


def render_latex_to_image(latex: str) -> io.BytesIO:
    """Render LaTeX string to an image in memory using matplotlib."""
    fig = plt.figure(figsize=(0.01, 0.01))
    fig.text(0, 0, f'${latex}$', fontsize=16)
    buf = io.BytesIO()
    plt.axis('off')
    plt.gca().set_axis_off()
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0.2, dpi=200, transparent=True)
    plt.close(fig)
    buf.seek(0)
    return buf


def export_sympy_pdf(
    sympy_data: Dict[str, Dict[str, str]],
    output_path: Path,
    title: str = "Symbolic Expressions Summary"
) -> None:
    """
    Export symbolic expressions for each profile to a formatted PDF.
    sympy_data: Dict[profile_name, Dict[str, str]]
    output_path: Path to save the PDF
    """
    c = canvas.Canvas(str(output_path), pagesize=letter)
    width, height = letter
    margin = 40
    y = height - margin

    c.setFont("Helvetica-Bold", 18)
    c.drawString(margin, y, title)
    y -= 30

    for profile_name, exprs in sympy_data.items():
        c.setFont("Helvetica-Bold", 14)
        c.drawString(margin, y, f"Profile: {profile_name}")
        y -= 22

        for label, desc in [
            ("U(z)", "U_latex"),
            ("U'(z)", "U_prime_latex"),
            ("U''(z)", "U_double_prime_latex")
        ]:
            latex = exprs.get(desc, None)
            if latex:
                img_buf = render_latex_to_image(latex)
                img = ImageReader(img_buf)
                img_width, img_height = img.getSize()
                scale = min((width - 2 * margin) / img_width, 1.0)
                img_width = int(img_width * scale)
                img_height = int(img_height * scale)
                if y - img_height < margin:
                    c.showPage()
                    y = height - margin
                c.setFont("Helvetica", 12)
                c.drawString(margin + 10, y, label)
                y -= 18
                c.drawImage(img, margin + 30, y - img_height, width=img_width, height=img_height, mask='auto')
                y -= img_height + 10
        y -= 10
        if y < margin + 100:
            c.showPage()
            y = height - margin
    c.save()
