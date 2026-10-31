import { useEffect, useState } from "react";
import { useParams, Link } from "react-router-dom";
import {
  ShieldAlert,
  ShieldCheck,
  AlertTriangle,
  FileText,
  Download,
  Layers,
  BarChart2,
  Brain,
  Clock,
  ArrowRight,
  Loader2,
  AlertCircle,
  FileCode,
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
import type { ScanRecord, TensorResult } from "../api";

export default function ScanResult() {
  const { scanId } = useParams<{ scanId: string }>();
  const [scan, setScan] = useState<ScanRecord | null>(null);
  const [tensors, setTensors] = useState<TensorResult[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [downloading, setDownloading] = useState<"json" | "pdf" | null>(null);

  useEffect(() => {
    if (scanId) {
      loadData(scanId);
    }
  }, [scanId]);

  const loadData = async (id: string) => {
    try {
      setLoading(true);
      setError(null);
      const [scanRes, tensorRes] = await Promise.all([
        apiClient.getScan(id),
        apiClient.getScanTensors(id, "anomaly_score", "desc"),
      ]);
      setScan(scanRes.data);
      setTensors(tensorRes.data);
    } catch (err: any) {
      console.error(err);
      setError(
        err.response?.data?.detail || "Failed to load scan results. Please check scan ID."
      );
    } finally {
      setLoading(false);
    }
  };

  const handleDownload = async (format: "json" | "pdf") => {
    if (!scanId) return;
    try {
      setDownloading(format);
      const res = await apiClient.getReport(scanId, format);
      if (format === "json") {
        const blob = new Blob([JSON.stringify(res.data, null, 2)], {
          type: "application/json",
        });
        const url = URL.createObjectURL(blob);
        const a = document.createElement("a");
        a.href = url;
        a.download = `neural_guard_${scanId.slice(0, 8)}_report.json`;
        a.click();
      } else {
        const url = URL.createObjectURL(new Blob([res.data], { type: "application/pdf" }));
        const a = document.createElement("a");
        a.href = url;
        a.download = `neural_guard_${scanId.slice(0, 8)}_report.pdf`;
        a.click();
      }
    } catch (err) {
      console.error("Report download failed:", err);
      alert("Report download failed. Check backend logs.");
    } finally {
      setDownloading(null);
    }
  };

  if (loading) {
    return (
      <div className="flex flex-col items-center justify-center min-h-[60vh] space-y-4">
        <Loader2 className="w-10 h-10 text-cyan-400 animate-spin" />
        <p className="text-sm text-[var(--color-slate-400)]">
          Retrieving model security scan report...
        </p>
      </div>
    );
  }

  if (error || !scan) {
    return (
      <div className="glass-card p-8 text-center max-w-lg mx-auto space-y-4">
        <AlertCircle className="w-12 h-12 text-rose-400 mx-auto" />
        <h2 className="text-lg font-bold text-white">Scan Not Found</h2>
        <p className="text-xs text-[var(--color-slate-400)]">{error}</p>
        <Link to="/scan" className="btn-primary inline-flex text-xs">
          Scan Another Model
        </Link>
      </div>
    );
  }

  const topTensor = tensors.length > 0 ? tensors[0] : null;

  // Prepare top tensor feature contributions for bar chart
  const contributionData = topTensor?.feature_contributions
    ? Object.entries(topTensor.feature_contributions).map(([feat, score]) => ({
        feature: feat.replace("f", "F"),
        score: Number(score.toFixed(3)),
      }))
    : [];

  let verdictBg = "border-emerald-500/30 bg-emerald-500/5";
  let verdictColor = "text-emerald-400";
  let VerdictIcon = ShieldCheck;
  let verdictText = "Verified Clean Model Weights";

  if (scan.risk_classification === "HIGH_ANOMALY") {
    verdictBg = "border-rose-500/40 bg-rose-500/10";
    verdictColor = "text-rose-400";
    VerdictIcon = ShieldAlert;
    verdictText = "CRITICAL: High Statistical Anomaly Detected";
  } else if (scan.risk_classification === "SUSPICIOUS") {
    verdictBg = "border-amber-500/40 bg-amber-500/10";
    verdictColor = "text-amber-400";
    VerdictIcon = AlertTriangle;
    verdictText = "WARNING: Suspicious Weight Deviations Detected";
  }

  return (
    <div className="space-y-8 animate-fade-in">
      {/* Top Breadcrumb & Actions */}
      <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center gap-4">
        <div>
          <div className="flex items-center gap-2 text-xs text-[var(--color-slate-500)] mb-1">
            <Link to="/" className="hover:text-cyan-400">
              Overview
            </Link>
            <span>/</span>
            <span className="text-[var(--color-slate-400)]">Scan Report</span>
            <span>/</span>
            <span className="text-cyan-400 font-mono">{scan.id.slice(0, 8)}</span>
          </div>
          <h1 className="text-2xl font-bold text-white flex items-center gap-2">
            <FileCode className="w-6 h-6 text-cyan-400" />
            {scan.filename}
          </h1>
        </div>

        <div className="flex flex-wrap items-center gap-2.5">
          <button
            onClick={() => handleDownload("json")}
            disabled={downloading !== null}
            className="btn-secondary text-xs flex items-center gap-1.5"
          >
            {downloading === "json" ? (
              <Loader2 className="w-3.5 h-3.5 animate-spin" />
            ) : (
              <Download className="w-3.5 h-3.5" />
            )}
            JSON Report
          </button>
          <button
            onClick={() => handleDownload("pdf")}
            disabled={downloading !== null}
            className="btn-secondary text-xs flex items-center gap-1.5"
          >
            {downloading === "pdf" ? (
              <Loader2 className="w-3.5 h-3.5 animate-spin" />
            ) : (
              <FileText className="w-3.5 h-3.5" />
            )}
            PDF Report
          </button>
          <Link
            to={`/scan/${scan.id}/explain`}
            className="btn-primary text-xs flex items-center gap-1.5"
          >
            <Brain className="w-3.5 h-3.5" />
            Explainability Center
          </Link>
        </div>
      </div>

      {/* Main Verdict Card */}
      <div className={`glass-card p-6 border ${verdictBg} flex flex-col md:flex-row justify-between items-start md:items-center gap-6`}>
        <div className="flex items-start gap-4">
          <div className={`w-14 h-14 rounded-2xl flex items-center justify-center shrink-0 ${verdictBg}`}>
            <VerdictIcon className={`w-8 h-8 ${verdictColor}`} />
          </div>
          <div className="space-y-1">
            <div className="flex items-center gap-2">
              <span className={`text-base font-extrabold ${verdictColor}`}>
                {verdictText}
              </span>
              <span className="px-2 py-0.5 rounded text-[10px] font-bold uppercase tracking-wider bg-white/10 text-white">
                {scan.risk_classification || "ANALYZED"}
              </span>
            </div>
            <p className="text-xs text-[var(--color-slate-400)] max-w-2xl leading-relaxed">
              {scan.risk_classification === "HIGH_ANOMALY"
                ? "Multiple weight tensors present severe statistical deviations across LSB entropy, Pov, and bit-plane correlation consistent with hidden payload embedding."
                : scan.risk_classification === "SUSPICIOUS"
                ? "Tensors exhibit atypical statistical features exceeding 99th percentile calibration limits. Inspection recommended before deployment."
                : "No abnormal bit-plane corruption or payload insertion detected. Weight distributions conform to standard deep learning training baselines."}
            </p>
          </div>
        </div>

        <div className="flex md:flex-col items-center md:items-end justify-between w-full md:w-auto border-t md:border-t-0 border-white/5 pt-4 md:pt-0">
          <span className="text-[10px] uppercase font-semibold text-[var(--color-slate-500)] tracking-wider">
            Overall Anomaly Score
          </span>
          <span className="text-3xl font-black font-mono text-cyan-400 mt-0.5">
            {scan.anomaly_score !== null ? scan.anomaly_score.toFixed(3) : "—"}
          </span>
          <span className="text-[11px] text-[var(--color-slate-500)]">
            Mahalanobis Distance D²
          </span>
        </div>
      </div>

      {/* Model Metadata Grid */}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
        <div className="glass-card p-4">
          <div className="text-[10px] text-[var(--color-slate-500)] uppercase font-semibold flex items-center gap-1.5">
            <Layers className="w-3.5 h-3.5 text-cyan-400" /> Total Tensors
          </div>
          <div className="text-xl font-bold text-white mt-1 font-mono">
            {scan.total_tensors}
          </div>
        </div>

        <div className="glass-card p-4">
          <div className="text-[10px] text-[var(--color-slate-500)] uppercase font-semibold flex items-center gap-1.5">
            <Brain className="w-3.5 h-3.5 text-cyan-400" /> Parameters
          </div>
          <div className="text-xl font-bold text-white mt-1 font-mono">
            {scan.total_parameters.toLocaleString()}
          </div>
        </div>

        <div className="glass-card p-4">
          <div className="text-[10px] text-[var(--color-slate-500)] uppercase font-semibold flex items-center gap-1.5">
            <FileCode className="w-3.5 h-3.5 text-cyan-400" /> Format
          </div>
          <div className="text-xl font-bold text-white mt-1 uppercase font-mono">
            {scan.model_format}
          </div>
        </div>

        <div className="glass-card p-4">
          <div className="text-[10px] text-[var(--color-slate-500)] uppercase font-semibold flex items-center gap-1.5">
            <Clock className="w-3.5 h-3.5 text-cyan-400" /> Completed
          </div>
          <div className="text-xs font-medium text-[var(--color-slate-300)] mt-2">
            {scan.completed_at ? new Date(scan.completed_at).toLocaleTimeString() : "Just now"}
          </div>
        </div>
      </div>

      {/* Primary Tensor Feature Anomaly Chart */}
      {topTensor && contributionData.length > 0 && (
        <div className="glass-card p-6">
          <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center mb-6 gap-2">
            <div>
              <h2 className="text-base font-bold text-white flex items-center gap-2">
                <BarChart2 className="w-4 h-4 text-cyan-400" />
                Primary Anomaly Feature Contributions
              </h2>
              <p className="text-xs text-[var(--color-slate-400)]">
                Most anomalous tensor: <span className="font-mono text-cyan-300">{topTensor.tensor_name}</span> (D² = {topTensor.anomaly_score?.toFixed(3)})
              </p>
            </div>
            <Link
              to={`/scan/${scan.id}/explain`}
              className="text-xs text-cyan-400 hover:text-cyan-300 font-semibold flex items-center gap-1"
            >
              Detailed Breakdown <ArrowRight className="w-3 h-3" />
            </Link>
          </div>

          <div className="h-64 w-full">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={contributionData}>
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
                <XAxis dataKey="feature" stroke="#64748b" tick={{ fontSize: 11 }} />
                <YAxis stroke="#64748b" tick={{ fontSize: 11 }} />
                <Tooltip
                  contentStyle={{
                    backgroundColor: "#0f1629",
                    borderColor: "rgba(34, 211, 238, 0.2)",
                    borderRadius: "8px",
                    color: "#fff",
                    fontSize: "12px",
                  }}
                />
                <Bar dataKey="score" radius={[4, 4, 0, 0]}>
                  {contributionData.map((entry, index) => (
                    <Cell
                      key={`bar-${index}`}
                      fill={entry.score > 2.0 ? "#ef4444" : entry.score > 1.0 ? "#f59e0b" : "#06b6d4"}
                    />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
      )}

      {/* Scanned Tensors Table */}
      <div className="glass-card p-6">
        <div className="flex justify-between items-center mb-4">
          <div>
            <h2 className="text-base font-bold text-white flex items-center gap-2">
              <Layers className="w-4 h-4 text-cyan-400" />
              Scanned Tensors Breakdown
            </h2>
            <p className="text-xs text-[var(--color-slate-400)]">
              Ranked by individual tensor Mahalanobis anomaly score
            </p>
          </div>
          <Link
            to={`/scan/${scan.id}/tensors`}
            className="text-xs text-cyan-400 hover:text-cyan-300 font-semibold flex items-center gap-1"
          >
            Explore All in Full View <ArrowRight className="w-3 h-3" />
          </Link>
        </div>

        <div className="overflow-x-auto">
          <table className="w-full text-left text-xs">
            <thead className="text-[var(--color-slate-400)] border-b border-white/10 uppercase tracking-wider">
              <tr>
                <th className="pb-3 font-semibold">Tensor Identifier</th>
                <th className="pb-3 font-semibold">Shape</th>
                <th className="pb-3 font-semibold">Dtype</th>
                <th className="pb-3 font-semibold">Parameters</th>
                <th className="pb-3 font-semibold">Anomaly (D²)</th>
                <th className="pb-3 font-semibold">Integrity Status</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-white/5 text-[var(--color-slate-300)]">
              {tensors.slice(0, 15).map((t) => {
                let badgeClass = "badge-normal";
                if (t.risk_classification === "HIGH_ANOMALY") badgeClass = "badge-high-anomaly";
                else if (t.risk_classification === "SUSPICIOUS") badgeClass = "badge-suspicious";
                else if (t.risk_classification === "INSUFFICIENT_DATA") badgeClass = "badge-insufficient";

                return (
                  <tr key={t.id} className="hover:bg-white/[0.02] transition-colors">
                    <td className="py-3 font-mono font-medium text-white max-w-xs truncate">
                      {t.tensor_name}
                    </td>
                    <td className="py-3 font-mono text-[var(--color-slate-400)]">
                      {t.tensor_shape ? `[${t.tensor_shape.join(", ")}]` : "—"}
                    </td>
                    <td className="py-3 font-mono text-[11px] text-cyan-400">
                      {t.tensor_dtype || "float32"}
                    </td>
                    <td className="py-3 font-mono">
                      {t.num_elements?.toLocaleString() || "—"}
                    </td>
                    <td className="py-3 font-mono font-semibold">
                      {t.anomaly_score !== null ? t.anomaly_score.toFixed(3) : "—"}
                    </td>
                    <td className="py-3">
                      <span className={`badge ${badgeClass}`}>
                        {t.risk_classification || "NORMAL"}
                      </span>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
