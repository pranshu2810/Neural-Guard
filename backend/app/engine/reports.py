"""
Report Generator — PDF and JSON forensic report creation.
"""

import json
import datetime
import os
from typing import Dict, Any, List, Optional
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


def generate_json_report(scan_data: Dict[str, Any], tensors: List[Dict], output_path: str) -> str:
    """Generate a JSON forensic report."""
    report = {
        "report_type": "Neural Guard Forensic Report",
        "version": "1.0",
        "generated_at": datetime.datetime.utcnow().isoformat() + "Z",
        "scan": {
            "scan_id": scan_data.get("id"),
            "filename": scan_data.get("filename"),
            "file_hash_sha256": scan_data.get("file_hash_sha256"),
            "file_size_bytes": scan_data.get("file_size_bytes"),
            "model_format": scan_data.get("model_format"),
            "scanned_at": scan_data.get("created_at"),
            "completed_at": scan_data.get("completed_at"),
        },
        "results": {
            "model_anomaly_score": scan_data.get("anomaly_score"),
            "risk_classification": scan_data.get("risk_classification"),
            "total_tensors": scan_data.get("total_tensors"),
            "total_parameters": scan_data.get("total_parameters"),
            "dtype_distribution": scan_data.get("dtype_distribution"),
        },
        "tensor_results": tensors,
        "methodology": {
            "description": (
                "Statistical analysis of neural network weight tensor bit patterns "
                "using a multi-feature anomaly detection framework with "
                "shrinkage-regularized Mahalanobis distance."
            ),
            "features_used": [
                "G-Test / Entropy Deficit",
                "Pairs-of-Values Test",
                "Bit-Plane Bias",
                "Serial Correlation",
                "Runs Test",
                "Bit Mutual Information",
                "Trailing-Zero Count",
                "Zero-Field Fraction",
                "FP16-Lattice Fraction",
                "Excess Kurtosis",
            ],
            "anomaly_detection": "Shrinkage-regularized Mahalanobis distance with Ledoit-Wolf estimator",
        },
        "limitations": [
            "Uniformly distributed random payload bits may preserve LSB statistics and evade detection.",
            "FP16-to-FP32 conversion creates detectable patterns without proving malicious embedding.",
            "Statistical anomalies do not establish malicious intent.",
            "Model architecture, training pipeline, quantization, and serialization affect feature distributions.",
            "AUC of 1.0 in limited experiments does not establish universal detection reliability.",
            "Cryptographic verification and statistical analysis should be used together when possible.",
            "This tool inspects weight statistics only. It does not analyze model behavior or architecture.",
        ],
        "disclaimer": (
            "This report presents statistical analysis of neural network weight tensors. "
            "Flagged anomalies indicate deviations from expected statistical patterns in "
            "the trusted reference population. These findings do NOT constitute definitive "
            "proof of malicious manipulation. Additional investigation, cryptographic "
            "verification, and provenance analysis are recommended."
        ),
    }
    
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    
    return output_path


def generate_pdf_report(scan_data: Dict[str, Any], tensors: List[Dict], output_path: str) -> str:
    """Generate a PDF forensic report using ReportLab."""
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib import colors
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import mm, cm
        from reportlab.platypus import (
            SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
            PageBreak, HRFlowable
        )
        from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
    except ImportError:
        logger.warning("reportlab not installed, falling back to JSON-only report.")
        return generate_json_report(scan_data, tensors, output_path.replace(".pdf", ".json"))
    
    doc = SimpleDocTemplate(
        output_path, pagesize=A4,
        rightMargin=2*cm, leftMargin=2*cm,
        topMargin=2*cm, bottomMargin=2*cm,
    )
    
    styles = getSampleStyleSheet()
    
    # Custom styles
    title_style = ParagraphStyle(
        'NeuralGuardTitle', parent=styles['Title'],
        fontSize=24, textColor=colors.HexColor("#0EA5E9"),
        spaceAfter=12,
    )
    heading_style = ParagraphStyle(
        'NeuralGuardHeading', parent=styles['Heading1'],
        fontSize=16, textColor=colors.HexColor("#0EA5E9"),
        spaceBefore=18, spaceAfter=8,
    )
    subheading_style = ParagraphStyle(
        'NeuralGuardSubheading', parent=styles['Heading2'],
        fontSize=13, textColor=colors.HexColor("#38BDF8"),
        spaceBefore=12, spaceAfter=6,
    )
    body_style = ParagraphStyle(
        'NeuralGuardBody', parent=styles['Normal'],
        fontSize=10, leading=14,
    )
    warning_style = ParagraphStyle(
        'NeuralGuardWarning', parent=styles['Normal'],
        fontSize=9, textColor=colors.HexColor("#F59E0B"),
        leading=12,
    )
    
    elements = []
    
    # Title
    elements.append(Paragraph("🛡️ Neural Guard", title_style))
    elements.append(Paragraph("Forensic Scan Report", heading_style))
    elements.append(HRFlowable(width="100%", color=colors.HexColor("#0EA5E9")))
    elements.append(Spacer(1, 12))
    
    # Scan Metadata
    elements.append(Paragraph("Scan Information", subheading_style))
    scan_info = [
        ["Scan ID", str(scan_data.get("id", "N/A"))],
        ["Filename", str(scan_data.get("filename", "N/A"))],
        ["SHA-256", str(scan_data.get("file_hash_sha256", "N/A"))[:32] + "..."],
        ["File Size", f"{scan_data.get('file_size_bytes', 0) / 1024 / 1024:.2f} MB"],
        ["Format", str(scan_data.get("model_format", "N/A"))],
        ["Scanned At", str(scan_data.get("created_at", "N/A"))],
    ]
    t = Table(scan_info, colWidths=[4*cm, 12*cm])
    t.setStyle(TableStyle([
        ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
    ]))
    elements.append(t)
    elements.append(Spacer(1, 12))
    
    # Results Summary
    elements.append(Paragraph("Detection Results", subheading_style))
    risk = scan_data.get("risk_classification", "unknown")
    risk_color = {
        "normal": "#22C55E",
        "suspicious": "#F59E0B",
        "high_anomaly": "#EF4444",
        "insufficient_evidence": "#6B7280",
    }.get(risk, "#6B7280")
    
    result_info = [
        ["Anomaly Score", f"{scan_data.get('anomaly_score', 0):.4f}"],
        ["Risk Classification", f'<font color="{risk_color}"><b>{risk.upper().replace("_", " ")}</b></font>'],
        ["Total Tensors", str(scan_data.get("total_tensors", 0))],
        ["Total Parameters", f"{scan_data.get('total_parameters', 0):,}"],
    ]
    for label, val in result_info:
        elements.append(Paragraph(f"<b>{label}:</b> {val}", body_style))
    elements.append(Spacer(1, 12))
    
    # Tensor Results
    elements.append(Paragraph("Tensor-Level Results", subheading_style))
    
    if tensors:
        # Sort by anomaly score descending
        sorted_tensors = sorted(tensors, key=lambda t: t.get("anomaly_score", 0), reverse=True)
        
        header = ["Tensor Name", "Score", "Risk", "Elements"]
        rows = [header]
        for t in sorted_tensors[:50]:  # Limit to top 50
            rows.append([
                str(t.get("tensor_name", ""))[:40],
                f"{t.get('anomaly_score', 0):.3f}",
                str(t.get("risk_classification", "")),
                str(t.get("num_elements", "")),
            ])
        
        table = Table(rows, colWidths=[7*cm, 2.5*cm, 3*cm, 3.5*cm])
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#1E293B")),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('FONTSIZE', (0, 0), (-1, -1), 8),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#334155")),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.HexColor("#F8FAFC"), colors.white]),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
            ('TOPPADDING', (0, 0), (-1, -1), 3),
        ]))
        elements.append(table)
    
    # Explanations for suspicious tensors
    suspicious = [t for t in tensors if t.get("risk_classification") in ("suspicious", "high_anomaly")]
    if suspicious:
        elements.append(PageBreak())
        elements.append(Paragraph("Detailed Explanations", subheading_style))
        for t in suspicious[:20]:
            elements.append(Paragraph(
                f"<b>{t.get('tensor_name', 'Unknown')}</b>",
                ParagraphStyle('TName', parent=body_style, fontSize=10, textColor=colors.HexColor("#EF4444"))
            ))
            explanation = t.get("explanation", "No explanation available.")
            elements.append(Paragraph(explanation, body_style))
            elements.append(Spacer(1, 8))
    
    # Limitations
    elements.append(PageBreak())
    elements.append(Paragraph("Methodology & Limitations", subheading_style))
    elements.append(Paragraph(
        "This analysis uses statistical tests on weight tensor bit patterns to detect "
        "anomalies that may indicate steganographic manipulation or unusual modifications. "
        "The detection methodology is based on a multi-feature framework using a "
        "shrinkage-regularized Mahalanobis distance anomaly detector.",
        body_style,
    ))
    elements.append(Spacer(1, 8))
    
    limitations = [
        "Uniformly distributed random payload bits may evade this statistical detector.",
        "FP16→FP32 conversion creates detectable precision-history patterns without proving malicious intent.",
        "Statistical anomalies do not establish malicious intent.",
        "Training pipeline, quantization, and serialization affect feature distributions.",
        "Cryptographic verification and statistical analysis should be used together.",
    ]
    for lim in limitations:
        elements.append(Paragraph(f"⚠ {lim}", warning_style))
    
    elements.append(Spacer(1, 20))
    elements.append(HRFlowable(width="100%", color=colors.HexColor("#0EA5E9")))
    elements.append(Paragraph(
        f"Report generated by Neural Guard v0.1.0 — {datetime.datetime.utcnow().isoformat()}Z",
        ParagraphStyle('Footer', parent=body_style, fontSize=8, textColor=colors.gray),
    ))
    
    doc.build(elements)
    return output_path
