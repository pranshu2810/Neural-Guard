import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import {
  History,
  Search,
  Filter,
  FileCode,
  ArrowRight,
  Loader2,
} from "lucide-react";
import apiClient from "../api";
import type { ScanRecord } from "../api";

export default function ScanHistory() {
  const [scans, setScans] = useState<ScanRecord[]>([]);
  const [loading, setLoading] = useState(true);
  const [search, setSearch] = useState("");
  const [riskFilter, setRiskFilter] = useState("ALL");

  useEffect(() => {
    loadScans();
  }, []);

  const loadScans = async () => {
    try {
      setLoading(true);
      const res = await apiClient.listScans(100, 0);
      setScans(res.data);
    } catch (err) {
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  const filtered = scans.filter((s) => {
    const matchName = s.filename.toLowerCase().includes(search.toLowerCase());
    const matchRisk = riskFilter === "ALL" || s.risk_classification === riskFilter;
    return matchName && matchRisk;
  });

  return (
    <div className="space-y-6 animate-fade-in">
      {/* Header */}
      <div>
        <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-cyan-500/10 border border-cyan-500/30 text-cyan-400 text-xs font-semibold uppercase tracking-wider mb-2">
          <History className="w-3.5 h-3.5" />
          Audit Trail
        </div>
        <h1 className="text-2xl font-bold text-white">Scan History & Audit Logs</h1>
        <p className="text-xs text-[var(--color-slate-400)] mt-1">
          Complete archive of all inspected neural model checkpoint scans and verdicts.
        </p>
      </div>

      {/* Filter and Search Bar */}
      <div className="glass-card p-4 flex flex-col md:flex-row items-center justify-between gap-4">
        <div className="relative w-full md:w-80">
          <Search className="w-4 h-4 text-[var(--color-slate-500)] absolute left-3 top-1/2 -translate-y-1/2" />
          <input
            type="text"
            placeholder="Search by filename or ID..."
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            className="w-full pl-9 pr-4 py-2 bg-[var(--color-navy-800)] border border-white/10 rounded-lg text-xs text-white placeholder-[var(--color-slate-500)] focus:outline-none focus:border-cyan-500/50"
          />
        </div>

        <div className="flex items-center gap-2 w-full md:w-auto overflow-x-auto pb-1 md:pb-0">
          <span className="text-xs text-[var(--color-slate-500)] flex items-center gap-1">
            <Filter className="w-3.5 h-3.5" /> Filter:
          </span>
          {["ALL", "NORMAL", "SUSPICIOUS", "HIGH_ANOMALY"].map((status) => (
            <button
              key={status}
              onClick={() => setRiskFilter(status)}
              className={`px-3 py-1.5 rounded-lg text-xs font-semibold whitespace-nowrap transition-colors ${
                riskFilter === status
                  ? "bg-cyan-500 text-black"
                  : "bg-[var(--color-navy-800)] text-[var(--color-slate-400)] hover:text-white"
              }`}
            >
              {status.replace("_", " ")}
            </button>
          ))}
        </div>
      </div>

      {/* Scans Table */}
      <div className="glass-card p-6">
        {loading ? (
          <div className="flex flex-col items-center justify-center py-16">
            <Loader2 className="w-8 h-8 text-cyan-400 animate-spin" />
            <span className="text-xs text-[var(--color-slate-400)] mt-2">Loading audit trail...</span>
          </div>
        ) : filtered.length > 0 ? (
          <div className="overflow-x-auto">
            <table className="w-full text-left text-xs">
              <thead className="text-[var(--color-slate-400)] border-b border-white/10 uppercase tracking-wider">
                <tr>
                  <th className="pb-3 font-semibold">Model Filename</th>
                  <th className="pb-3 font-semibold">Format</th>
                  <th className="pb-3 font-semibold">Tensors</th>
                  <th className="pb-3 font-semibold">Parameters</th>
                  <th className="pb-3 font-semibold">Score (D²)</th>
                  <th className="pb-3 font-semibold">Verdict</th>
                  <th className="pb-3 font-semibold">Date</th>
                  <th className="pb-3 font-semibold text-right">Actions</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-white/5 text-[var(--color-slate-300)]">
                {filtered.map((s) => {
                  let badgeClass = "badge-normal";
                  if (s.risk_classification === "HIGH_ANOMALY") badgeClass = "badge-high-anomaly";
                  else if (s.risk_classification === "SUSPICIOUS") badgeClass = "badge-suspicious";
                  else if (s.risk_classification === "INSUFFICIENT_DATA") badgeClass = "badge-insufficient";

                  return (
                    <tr key={s.id} className="hover:bg-white/[0.02] transition-colors">
                      <td className="py-3 font-medium text-white flex items-center gap-2 max-w-xs truncate">
                        <FileCode className="w-4 h-4 text-cyan-400 shrink-0" />
                        <span className="truncate">{s.filename}</span>
                      </td>
                      <td className="py-3 uppercase font-mono text-[11px] text-[var(--color-slate-400)]">
                        {s.model_format}
                      </td>
                      <td className="py-3 font-mono">{s.total_tensors}</td>
                      <td className="py-3 font-mono">{s.total_parameters.toLocaleString()}</td>
                      <td className="py-3 font-mono font-semibold">
                        {s.anomaly_score !== null ? s.anomaly_score.toFixed(3) : "—"}
                      </td>
                      <td className="py-3">
                        <span className={`badge ${badgeClass} text-[10px] py-0.5 px-2`}>
                          {s.risk_classification || s.status}
                        </span>
                      </td>
                      <td className="py-3 text-[var(--color-slate-400)] whitespace-nowrap">
                        {s.created_at ? new Date(s.created_at).toLocaleDateString() : "—"}
                      </td>
                      <td className="py-3 text-right">
                        <Link
                          to={`/scan/${s.id}`}
                          className="text-cyan-400 hover:text-cyan-300 font-semibold inline-flex items-center gap-1"
                        >
                          Details <ArrowRight className="w-3 h-3" />
                        </Link>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        ) : (
          <div className="text-center py-12 text-[var(--color-slate-500)] text-xs">
            No matching scan history records found.
          </div>
        )}
      </div>
    </div>
  );
}
