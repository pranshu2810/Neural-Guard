import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import {
  ShieldAlert,
  ShieldCheck,
  AlertTriangle,
  Layers,
  Activity,
  UploadCloud,
  FileSearch,
  ArrowRight,
} from "lucide-react";
import {
  ResponsiveContainer,
  PieChart,
  Pie,
  Cell,
  Tooltip,
} from "recharts";
import apiClient from "../api";
import type { DashboardStats } from "../api";

const RISK_COLORS: Record<string, string> = {
  NORMAL: "#22c55e",
  SUSPICIOUS: "#f59e0b",
  HIGH_ANOMALY: "#ef4444",
  INSUFFICIENT_DATA: "#64748b",
};

export default function Dashboard() {
  const [stats, setStats] = useState<DashboardStats | null>(null);

  useEffect(() => {
    fetchStats();
  }, []);

  const fetchStats = async () => {
    try {
      const res = await apiClient.getStats();
      setStats(res.data);
    } catch (err: unknown) {
      console.warn("Could not fetch real dashboard stats:", err);
      // Fallback preview data when database is fresh
      setStats({
        total_scans: 0,
        normal_count: 0,
        suspicious_count: 0,
        high_anomaly_count: 0,
        insufficient_count: 0,
        failed_count: 0,
        avg_anomaly_score: 0.0,
        total_parameters_analyzed: 0,
        recent_scans: [],
        risk_distribution: {
          NORMAL: 0,
          SUSPICIOUS: 0,
          HIGH_ANOMALY: 0,
        },
      });
    }
  };

  const chartData = stats
    ? Object.entries(stats.risk_distribution || {}).map(([key, val]) => ({
        name: key.replace("_", " "),
        value: val,
        rawKey: key,
      }))
    : [];

  const formatNumber = (num: number) => {
    if (num >= 1e9) return (num / 1e9).toFixed(2) + "B";
    if (num >= 1e6) return (num / 1e6).toFixed(2) + "M";
    if (num >= 1e3) return (num / 1e3).toFixed(1) + "K";
    return num.toLocaleString();
  };

  return (
    <div className="space-y-8 animate-fade-in">
      {/* Header Banner */}
      <div className="glass-card p-8 relative overflow-hidden bg-gradient-to-r from-[var(--color-navy-900)] via-[var(--color-navy-800)] to-[var(--color-navy-900)] border border-cyan-500/20">
        <div className="absolute top-0 right-0 w-96 h-96 bg-cyan-500/5 rounded-full blur-3xl -z-10 pointer-events-none" />
        <div className="flex flex-col md:flex-row justify-between items-start md:items-center gap-6">
          <div className="space-y-2">
            <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-cyan-500/10 border border-cyan-500/30 text-cyan-400 text-xs font-semibold uppercase tracking-wider">
              <Activity className="w-3.5 h-3.5" />
              Model Security Operational Center
            </div>
            <h1 className="text-3xl font-extrabold text-white tracking-tight">
              Neural Weight Steganography Guard
            </h1>
            <p className="text-[var(--color-slate-400)] text-sm max-w-2xl">
              Explainable statistical detection of steganographic manipulation and
              anomalous bit-plane distribution across safetensors, PyTorch, and ONNX weights.
            </p>
          </div>
          <div className="flex gap-3">
            <Link to="/scan" className="btn-primary">
              <UploadCloud className="w-4 h-4" />
              Scan Checkpoint
            </Link>
          </div>
        </div>
      </div>

      {/* Metrics Row */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-5">
        <div className="glass-card stat-card flex items-center justify-between">
          <div>
            <div className="stat-value text-white">
              {stats ? stats.total_scans : 0}
            </div>
            <div className="stat-label">Total Models Scanned</div>
          </div>
          <div className="w-12 h-12 rounded-xl bg-cyan-500/10 border border-cyan-500/20 flex items-center justify-center text-cyan-400">
            <Layers className="w-6 h-6" />
          </div>
        </div>

        <div className="glass-card stat-card flex items-center justify-between">
          <div>
            <div className="stat-value text-emerald-400">
              {stats ? stats.normal_count : 0}
            </div>
            <div className="stat-label">Verified Normal</div>
          </div>
          <div className="w-12 h-12 rounded-xl bg-emerald-500/10 border border-emerald-500/20 flex items-center justify-center text-emerald-400">
            <ShieldCheck className="w-6 h-6" />
          </div>
        </div>

        <div className="glass-card stat-card flex items-center justify-between">
          <div>
            <div className="stat-value text-amber-400">
              {stats ? stats.suspicious_count : 0}
            </div>
            <div className="stat-label">Suspicious Tensors</div>
          </div>
          <div className="w-12 h-12 rounded-xl bg-amber-500/10 border border-amber-500/20 flex items-center justify-center text-amber-400">
            <AlertTriangle className="w-6 h-6" />
          </div>
        </div>

        <div className="glass-card stat-card flex items-center justify-between">
          <div>
            <div className="stat-value text-rose-400">
              {stats ? stats.high_anomaly_count : 0}
            </div>
            <div className="stat-label">High Anomaly Flagged</div>
          </div>
          <div className="w-12 h-12 rounded-xl bg-rose-500/10 border border-rose-500/20 flex items-center justify-center text-rose-400">
            <ShieldAlert className="w-6 h-6" />
          </div>
        </div>
      </div>

      {/* Second Metrics & Charts Row */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Risk Distribution Chart */}
        <div className="glass-card p-6 flex flex-col justify-between">
          <div>
            <h2 className="text-base font-bold text-white mb-1">
              Risk Classification Distribution
            </h2>
            <p className="text-xs text-[var(--color-slate-400)] mb-4">
              Calibrated Mahalanobis distance categorizations
            </p>
          </div>

          <div className="h-56 w-full flex items-center justify-center">
            {chartData.length > 0 && chartData.some((d) => d.value > 0) ? (
              <ResponsiveContainer width="100%" height="100%">
                <PieChart>
                  <Pie
                    data={chartData}
                    dataKey="value"
                    nameKey="name"
                    cx="50%"
                    cy="50%"
                    innerRadius={50}
                    outerRadius={80}
                    paddingAngle={4}
                  >
                    {chartData.map((entry, index) => (
                      <Cell
                        key={`cell-${index}`}
                        fill={RISK_COLORS[entry.rawKey] || "#64748b"}
                      />
                    ))}
                  </Pie>
                  <Tooltip
                    contentStyle={{
                      backgroundColor: "#0f1629",
                      borderColor: "rgba(34, 211, 238, 0.2)",
                      borderRadius: "8px",
                      color: "#fff",
                    }}
                  />
                </PieChart>
              </ResponsiveContainer>
            ) : (
              <div className="text-center p-6 text-[var(--color-slate-500)] text-sm">
                <FileSearch className="w-10 h-10 mx-auto mb-2 opacity-50" />
                No scan data collected yet. Scan your first model checkpoint to populate.
              </div>
            )}
          </div>

          <div className="flex justify-center gap-4 text-xs mt-2">
            <span className="flex items-center gap-1.5 text-emerald-400">
              <span className="w-2.5 h-2.5 rounded-full bg-emerald-500 inline-block" />
              Normal
            </span>
            <span className="flex items-center gap-1.5 text-amber-400">
              <span className="w-2.5 h-2.5 rounded-full bg-amber-500 inline-block" />
              Suspicious
            </span>
            <span className="flex items-center gap-1.5 text-rose-400">
              <span className="w-2.5 h-2.5 rounded-full bg-rose-500 inline-block" />
              High Anomaly
            </span>
          </div>
        </div>

        {/* Global Parameter Throughput & Health */}
        <div className="glass-card p-6 flex flex-col justify-between">
          <div>
            <h2 className="text-base font-bold text-white mb-1">
              Parameters Inspected
            </h2>
            <p className="text-xs text-[var(--color-slate-400)] mb-4">
              Scale of analyzed float32 / float16 weights
            </p>
          </div>

          <div className="my-auto py-6 text-center">
            <div className="text-4xl lg:text-5xl font-black text-cyan-400 font-mono tracking-tight">
              {stats ? formatNumber(stats.total_parameters_analyzed) : "0"}
            </div>
            <p className="text-xs uppercase tracking-widest text-[var(--color-slate-500)] mt-2">
              Extracted Weight Elements
            </p>
          </div>

          <div className="p-3.5 rounded-lg bg-[var(--color-navy-800)] border border-white/5 space-y-2 text-xs">
            <div className="flex justify-between text-[var(--color-slate-400)]">
              <span>Average Anomaly Score:</span>
              <span className="font-mono text-cyan-400 font-semibold">
                {stats?.avg_anomaly_score ? stats.avg_anomaly_score.toFixed(3) : "0.000"}
              </span>
            </div>
            <div className="flex justify-between text-[var(--color-slate-400)]">
              <span>Multi-Feature Vector:</span>
              <span className="text-slate-300 font-medium">10 Stat Features</span>
            </div>
          </div>
        </div>

        {/* Quick Guide / Feature Overview */}
        <div className="glass-card p-6 flex flex-col justify-between bg-gradient-to-br from-[var(--color-navy-900)] to-cyan-950/20">
          <div>
            <h2 className="text-base font-bold text-white mb-1">
              Detection Framework
            </h2>
            <p className="text-xs text-[var(--color-slate-400)] mb-4">
              Multi-feature statistical anomaly engine
            </p>
            <ul className="space-y-2.5 text-xs text-[var(--color-slate-300)]">
              <li className="flex items-start gap-2">
                <span className="w-1.5 h-1.5 rounded-full bg-cyan-400 mt-1.5 shrink-0" />
                <span>
                  <strong>LSB Bit Entropy:</strong> Detects injected high-entropy ciphertext into mantissa bit-planes.
                </span>
              </li>
              <li className="flex items-start gap-2">
                <span className="w-1.5 h-1.5 rounded-full bg-cyan-400 mt-1.5 shrink-0" />
                <span>
                  <strong>POV & Serial Runs:</strong> Identifies value pair clustering and run length violations.
                </span>
              </li>
              <li className="flex items-start gap-2">
                <span className="w-1.5 h-1.5 rounded-full bg-cyan-400 mt-1.5 shrink-0" />
                <span>
                  <strong>FP16 Lattice & Sparsity:</strong> Flags zero-field corruption and quantization artifacts.
                </span>
              </li>
            </ul>
          </div>

          <div className="mt-4 pt-4 border-t border-white/5 flex justify-between items-center">
            <span className="text-[11px] text-[var(--color-slate-500)]">
              Paper Implementation MVP
            </span>
            <Link
              to="/settings"
              className="text-xs font-semibold text-cyan-400 hover:text-cyan-300 flex items-center gap-1"
            >
              Verify Config <ArrowRight className="w-3 h-3" />
            </Link>
          </div>
        </div>
      </div>

      {/* Recent Scans Table */}
      <div className="glass-card p-6">
        <div className="flex justify-between items-center mb-6">
          <div>
            <h2 className="text-lg font-bold text-white">Recent Model Scans</h2>
            <p className="text-xs text-[var(--color-slate-400)]">
              Latest checkpoint integrity verification results
            </p>
          </div>
          <Link
            to="/history"
            className="text-xs font-semibold text-cyan-400 hover:text-cyan-300 flex items-center gap-1"
          >
            View All Scans <ArrowRight className="w-3 h-3" />
          </Link>
        </div>

        {stats && stats.recent_scans.length > 0 ? (
          <div className="overflow-x-auto">
            <table className="w-full text-left text-xs">
              <thead className="text-[var(--color-slate-400)] border-b border-white/10 uppercase tracking-wider">
                <tr>
                  <th className="pb-3 font-semibold">Model / Filename</th>
                  <th className="pb-3 font-semibold">Format</th>
                  <th className="pb-3 font-semibold">Tensors</th>
                  <th className="pb-3 font-semibold">Max Anomaly Score</th>
                  <th className="pb-3 font-semibold">Integrity Status</th>
                  <th className="pb-3 font-semibold text-right">Action</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-white/5 text-[var(--color-slate-300)]">
                {stats.recent_scans.map((scan) => {
                  let badgeClass = "badge-normal";
                  if (scan.risk_classification === "HIGH_ANOMALY") badgeClass = "badge-high-anomaly";
                  else if (scan.risk_classification === "SUSPICIOUS") badgeClass = "badge-suspicious";
                  else if (scan.risk_classification === "INSUFFICIENT_DATA") badgeClass = "badge-insufficient";

                  return (
                    <tr key={scan.id} className="hover:bg-white/[0.02] transition-colors">
                      <td className="py-3 font-medium text-white flex items-center gap-2">
                        <FileSearch className="w-4 h-4 text-cyan-400" />
                        <span className="truncate max-w-xs">{scan.filename}</span>
                      </td>
                      <td className="py-3 uppercase font-mono text-[11px] text-[var(--color-slate-400)]">
                        {scan.model_format}
                      </td>
                      <td className="py-3 font-mono">{scan.total_tensors}</td>
                      <td className="py-3 font-mono font-semibold">
                        {scan.anomaly_score !== null ? scan.anomaly_score.toFixed(3) : "—"}
                      </td>
                      <td className="py-3">
                        <span className={`badge ${badgeClass}`}>
                          {scan.risk_classification || scan.status}
                        </span>
                      </td>
                      <td className="py-3 text-right">
                        <Link
                          to={`/scan/${scan.id}`}
                          className="text-cyan-400 hover:text-cyan-300 font-semibold"
                        >
                          View Report →
                        </Link>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        ) : (
          <div className="text-center py-12 border border-dashed border-white/10 rounded-xl">
            <FileSearch className="w-12 h-12 mx-auto text-[var(--color-slate-600)] mb-3" />
            <h3 className="text-sm font-semibold text-white">No scans executed yet</h3>
            <p className="text-xs text-[var(--color-slate-500)] max-w-md mx-auto mt-1 mb-4">
              Upload a .safetensors, .pt/.pth, or .onnx model checkpoint file to inspect its internal weight tensors for steganographic anomalies.
            </p>
            <Link to="/scan" className="btn-primary text-xs">
              <UploadCloud className="w-4 h-4" /> Scan a Model Now
            </Link>
          </div>
        )}
      </div>
    </div>
  );
}
