import { useEffect, useState } from "react";
import { useParams, Link } from "react-router-dom";
import {
  Brain,
  ShieldAlert,
  Lightbulb,
  Layers,
  Loader2,
  ShieldCheck,
} from "lucide-react";
import {
  ResponsiveContainer,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  Cell,
  CartesianGrid,
} from "recharts";
import apiClient from "../api";
import type { ExplanationData } from "../api";

export default function Explainability() {
  const { scanId } = useParams<{ scanId: string }>();
  const [explanations, setExplanations] = useState<ExplanationData[]>([]);
  const [selectedIdx, setSelectedIdx] = useState<number>(0);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    if (scanId) {
      loadExplanations(scanId);
    }
  }, [scanId]);

  const loadExplanations = async (id: string) => {
    try {
      setLoading(true);
      const res = await apiClient.getExplanations(id);
      setExplanations(res.data.explanations || []);
    } catch (err) {
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  if (loading) {
    return (
      <div className="flex flex-col items-center justify-center min-h-[60vh] space-y-4">
        <Loader2 className="w-10 h-10 text-cyan-400 animate-spin" />
        <p className="text-sm text-[var(--color-slate-400)]">
          Generating explainability models and feature decompositions...
        </p>
      </div>
    );
  }

  const current = explanations.length > 0 ? explanations[selectedIdx] : null;

  const contribChartData = current?.feature_contributions
    ? Object.entries(current.feature_contributions).map(([feat, score]) => ({
        feature: feat.toUpperCase(),
        score: Number(score.toFixed(3)),
      }))
    : [];

  return (
    <div className="space-y-8 animate-fade-in">
      {/* Header */}
      <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center gap-4">
        <div>
          <div className="flex items-center gap-2 text-xs text-[var(--color-slate-500)] mb-1">
            <Link to="/" className="hover:text-cyan-400">
              Overview
            </Link>
            <span>/</span>
            <Link to={`/scan/${scanId}`} className="hover:text-cyan-400">
              Scan {scanId?.slice(0, 8)}
            </Link>
            <span>/</span>
            <span className="text-cyan-400">Explainability Center</span>
          </div>
          <h1 className="text-2xl font-bold text-white flex items-center gap-2">
            <Brain className="w-6 h-6 text-cyan-400" />
            Explainable AI Anomaly Insights
          </h1>
          <p className="text-xs text-[var(--color-slate-400)] mt-1">
            Mathematical decomposition of Mahalanobis distance into individual feature anomaly vectors.
          </p>
        </div>

        <Link
          to={`/scan/${scanId}/tensors`}
          className="btn-secondary text-xs flex items-center gap-1.5"
        >
          <Layers className="w-3.5 h-3.5" />
          View All Tensors
        </Link>
      </div>

      {explanations.length === 0 ? (
        <div className="glass-card p-12 text-center space-y-3">
          <ShieldCheck className="w-12 h-12 text-emerald-400 mx-auto" />
          <h2 className="text-lg font-bold text-white">No Flagged Anomalies</h2>
          <p className="text-xs text-[var(--color-slate-400)] max-w-md mx-auto">
            All tensors in this model checkpoint conform to natural training distributions. No
            steganographic deviation was detected above calibration thresholds.
          </p>
          <Link to={`/scan/${scanId}`} className="btn-secondary text-xs inline-flex mt-2">
            Back to Scan Overview
          </Link>
        </div>
      ) : (
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
          {/* Left: List of flagged tensors */}
          <div className="lg:col-span-4 glass-card p-4 space-y-2 max-h-[600px] overflow-y-auto">
            <div className="text-xs font-bold uppercase tracking-wider text-[var(--color-slate-400)] px-2 mb-2">
              Anomalous Tensors ({explanations.length})
            </div>
            {explanations.map((exp, idx) => (
              <button
                key={idx}
                onClick={() => setSelectedIdx(idx)}
                className={`w-full text-left p-3 rounded-lg text-xs transition-all ${
                  selectedIdx === idx
                    ? "bg-cyan-500/15 border border-cyan-500/40 text-white"
                    : "bg-[var(--color-navy-900)]/60 hover:bg-white/[0.04] text-[var(--color-slate-300)]"
                }`}
              >
                <div className="flex justify-between items-center mb-1">
                  <span className="font-mono font-bold truncate max-w-[170px]">
                    {exp.tensor_name}
                  </span>
                  <span
                    className={`badge text-[9px] py-0.5 px-1.5 ${
                      exp.risk_classification === "HIGH_ANOMALY"
                        ? "badge-high-anomaly"
                        : "badge-suspicious"
                    }`}
                  >
                    {exp.risk_classification}
                  </span>
                </div>
                <div className="flex justify-between text-[11px] text-[var(--color-slate-500)] font-mono">
                  <span>Score D²: {exp.anomaly_score.toFixed(3)}</span>
                  <span>[{exp.tensor_shape?.join(", ")}]</span>
                </div>
              </button>
            ))}
          </div>

          {/* Right: Detailed Explanation & Decomposition */}
          {current && (
            <div className="lg:col-span-8 space-y-6">
              {/* Verdict banner */}
              <div className="glass-card p-6 border border-cyan-500/30">
                <div className="flex items-center justify-between mb-3">
                  <span className="text-xs font-mono uppercase text-cyan-400 tracking-wider">
                    Diagnostic Analysis
                  </span>
                  <span className="text-xs font-mono font-bold text-white">
                    Mahalanobis Distance D² = {current.anomaly_score.toFixed(4)}
                  </span>
                </div>
                <h2 className="text-lg font-bold text-white font-mono break-all mb-3">
                  {current.tensor_name}
                </h2>
                <div className="p-4 rounded-xl bg-[var(--color-navy-800)] border border-white/5 text-xs text-[var(--color-slate-200)] leading-relaxed flex items-start gap-3">
                  <Lightbulb className="w-5 h-5 text-amber-400 shrink-0 mt-0.5" />
                  <div>
                    <strong className="text-white block mb-1">Statistical Diagnosis:</strong>
                    {current.explanation}
                  </div>
                </div>
              </div>

              {/* Feature Contribution Decomposition Chart */}
              <div className="glass-card p-6">
                <h3 className="text-sm font-bold text-white mb-1">
                  Feature Contribution Decomposition (Shapley / Mahalanobis Projection)
                </h3>
                <p className="text-xs text-[var(--color-slate-400)] mb-4">
                  Shows how much each of the 10 statistical features drives this tensor's anomaly score.
                </p>

                <div className="h-64 w-full">
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart data={contribChartData} layout="vertical">
                      <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
                      <XAxis type="number" stroke="#64748b" tick={{ fontSize: 11 }} />
                      <YAxis
                        type="category"
                        dataKey="feature"
                        stroke="#64748b"
                        tick={{ fontSize: 11 }}
                        width={60}
                      />
                      <Tooltip
                        contentStyle={{
                          backgroundColor: "#0f1629",
                          borderColor: "rgba(34, 211, 238, 0.2)",
                          borderRadius: "8px",
                          color: "#fff",
                          fontSize: "12px",
                        }}
                      />
                      <Bar dataKey="score" radius={[0, 4, 4, 0]}>
                        {contribChartData.map((entry, index) => (
                          <Cell
                            key={`cell-${index}`}
                            fill={
                              entry.score > 2.0
                                ? "#ef4444"
                                : entry.score > 1.0
                                ? "#f59e0b"
                                : "#06b6d4"
                            }
                          />
                        ))}
                      </Bar>
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              </div>

              {/* Steganography Remediation & Mitigation Recommendations */}
              <div className="glass-card p-6 bg-gradient-to-br from-[var(--color-navy-900)] to-cyan-950/20">
                <h3 className="text-sm font-bold text-white mb-2 flex items-center gap-2">
                  <ShieldAlert className="w-4 h-4 text-cyan-400" />
                  Recommended Security Remediations
                </h3>
                <div className="space-y-3 text-xs text-[var(--color-slate-300)]">
                  <div className="p-3 rounded-lg bg-[var(--color-navy-800)] border border-white/5">
                    <strong className="text-white block mb-0.5">1. Weight Re-quantization / Truncation</strong>
                    If steganographic data is stored in the mantissa LSBs (F01/F02), zeroing the last 2-4 bits or re-quantizing to FP16/INT8 will sanitize the embedded payload without degrading model accuracy.
                  </div>
                  <div className="p-3 rounded-lg bg-[var(--color-navy-800)] border border-white/5">
                    <strong className="text-white block mb-0.5">2. Checkpoint Fine-Tuning Purification</strong>
                    Run a brief 1-epoch fine-tuning pass on trusted training data. Natural gradient updates will overwrite artificially modified weight bit patterns.
                  </div>
                </div>
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
