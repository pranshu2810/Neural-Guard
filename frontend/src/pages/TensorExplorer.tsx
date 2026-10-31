import { useEffect, useState } from "react";
import { useParams, Link } from "react-router-dom";
import {
  Layers,
  Search,
  Filter,
  ChevronRight,
  Loader2,
} from "lucide-react";
import apiClient from "../api";
import type { TensorResult } from "../api";

export default function TensorExplorer() {
  const { scanId } = useParams<{ scanId: string }>();
  const [tensors, setTensors] = useState<TensorResult[]>([]);
  const [loading, setLoading] = useState(true);
  const [search, setSearch] = useState("");
  const [riskFilter, setRiskFilter] = useState<string>("ALL");
  const [selectedTensor, setSelectedTensor] = useState<TensorResult | null>(null);

  useEffect(() => {
    if (scanId) {
      loadTensors(scanId);
    }
  }, [scanId]);

  const loadTensors = async (id: string) => {
    try {
      setLoading(true);
      const res = await apiClient.getScanTensors(id, "anomaly_score", "desc");
      setTensors(res.data);
      if (res.data.length > 0) {
        setSelectedTensor(res.data[0]);
      }
    } catch (err) {
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  const filtered = tensors.filter((t) => {
    const matchName = t.tensor_name.toLowerCase().includes(search.toLowerCase());
    const matchRisk = riskFilter === "ALL" || t.risk_classification === riskFilter;
    return matchName && matchRisk;
  });

  if (loading) {
    return (
      <div className="flex flex-col items-center justify-center min-h-[60vh] space-y-4">
        <Loader2 className="w-10 h-10 text-cyan-400 animate-spin" />
        <p className="text-sm text-[var(--color-slate-400)]">
          Indexing tensor feature space...
        </p>
      </div>
    );
  }

  return (
    <div className="space-y-6 animate-fade-in">
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
            <span className="text-cyan-400">Tensor Explorer</span>
          </div>
          <h1 className="text-2xl font-bold text-white flex items-center gap-2">
            <Layers className="w-6 h-6 text-cyan-400" />
            Tensor Level Feature Explorer
          </h1>
        </div>

        <Link
          to={`/scan/${scanId}/explain`}
          className="btn-secondary text-xs flex items-center gap-1.5"
        >
          View Model Explainability →
        </Link>
      </div>

      {/* Filter and Search Bar */}
      <div className="glass-card p-4 flex flex-col md:flex-row items-center justify-between gap-4">
        <div className="relative w-full md:w-80">
          <Search className="w-4 h-4 text-[var(--color-slate-500)] absolute left-3 top-1/2 -translate-y-1/2" />
          <input
            type="text"
            placeholder="Search tensor by name..."
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            className="w-full pl-9 pr-4 py-2 bg-[var(--color-navy-800)] border border-white/10 rounded-lg text-xs text-white placeholder-[var(--color-slate-500)] focus:outline-none focus:border-cyan-500/50"
          />
        </div>

        <div className="flex items-center gap-2 w-full md:w-auto overflow-x-auto pb-1 md:pb-0">
          <span className="text-xs text-[var(--color-slate-500)] flex items-center gap-1">
            <Filter className="w-3.5 h-3.5" /> Filter:
          </span>
          {["ALL", "HIGH_ANOMALY", "SUSPICIOUS", "NORMAL", "INSUFFICIENT_DATA"].map((status) => (
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

      {/* Main Split Layout: Table on left, Inspector drawer on right */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* Tensors Table */}
        <div className="lg:col-span-7 glass-card p-4 overflow-hidden">
          <div className="text-xs text-[var(--color-slate-400)] mb-3 flex justify-between items-center px-2">
            <span>Showing {filtered.length} of {tensors.length} tensors</span>
            <span className="text-[11px] text-[var(--color-slate-500)]">Click row to inspect features</span>
          </div>

          <div className="overflow-x-auto max-h-[650px] overflow-y-auto">
            <table className="w-full text-left text-xs">
              <thead className="text-[var(--color-slate-400)] border-b border-white/10 sticky top-0 bg-[var(--color-navy-900)] uppercase tracking-wider z-10">
                <tr>
                  <th className="pb-3 px-3 font-semibold">Tensor Name</th>
                  <th className="pb-3 px-3 font-semibold">Elements</th>
                  <th className="pb-3 px-3 font-semibold">Anomaly (D²)</th>
                  <th className="pb-3 px-3 font-semibold">Status</th>
                  <th className="pb-3 px-3"></th>
                </tr>
              </thead>
              <tbody className="divide-y divide-white/5 text-[var(--color-slate-300)]">
                {filtered.map((t) => {
                  let badgeClass = "badge-normal";
                  if (t.risk_classification === "HIGH_ANOMALY") badgeClass = "badge-high-anomaly";
                  else if (t.risk_classification === "SUSPICIOUS") badgeClass = "badge-suspicious";
                  else if (t.risk_classification === "INSUFFICIENT_DATA") badgeClass = "badge-insufficient";

                  const isSelected = selectedTensor?.id === t.id;

                  return (
                    <tr
                      key={t.id}
                      onClick={() => setSelectedTensor(t)}
                      className={`cursor-pointer transition-colors ${
                        isSelected
                          ? "bg-cyan-500/10 border-l-2 border-cyan-400"
                          : "hover:bg-white/[0.02]"
                      }`}
                    >
                      <td className="py-3 px-3 font-mono font-medium text-white max-w-[200px] truncate">
                        {t.tensor_name}
                      </td>
                      <td className="py-3 px-3 font-mono text-[var(--color-slate-400)]">
                        {t.num_elements?.toLocaleString()}
                      </td>
                      <td className="py-3 px-3 font-mono font-bold text-cyan-400">
                        {t.anomaly_score?.toFixed(3) || "—"}
                      </td>
                      <td className="py-3 px-3">
                        <span className={`badge ${badgeClass} text-[10px] py-0.5 px-2`}>
                          {t.risk_classification || "NORMAL"}
                        </span>
                      </td>
                      <td className="py-3 px-3 text-right">
                        <ChevronRight className={`w-4 h-4 ${isSelected ? "text-cyan-400" : "text-slate-600"}`} />
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </div>

        {/* Selected Tensor Feature Inspector */}
        <div className="lg:col-span-5 glass-card p-6 space-y-6">
          {selectedTensor ? (
            <>
              <div>
                <div className="flex items-center justify-between">
                  <span className="text-[10px] font-mono uppercase tracking-wider text-cyan-400">
                    Tensor Feature Profile
                  </span>
                  <span
                    className={`badge ${
                      selectedTensor.risk_classification === "HIGH_ANOMALY"
                        ? "badge-high-anomaly"
                        : selectedTensor.risk_classification === "SUSPICIOUS"
                        ? "badge-suspicious"
                        : "badge-normal"
                    }`}
                  >
                    {selectedTensor.risk_classification}
                  </span>
                </div>
                <h2 className="text-base font-bold text-white font-mono break-all mt-1">
                  {selectedTensor.tensor_name}
                </h2>
                <div className="flex gap-4 text-xs text-[var(--color-slate-400)] mt-2 font-mono">
                  <span>Shape: [{selectedTensor.tensor_shape?.join(", ")}]</span>
                  <span>Dtype: {selectedTensor.tensor_dtype}</span>
                </div>
              </div>

              {/* Anomaly & Explanation Box */}
              <div className="p-4 rounded-xl bg-[var(--color-navy-800)] border border-white/5 space-y-2">
                <div className="flex justify-between items-center text-xs">
                  <span className="text-[var(--color-slate-400)]">Mahalanobis Anomaly D²:</span>
                  <span className="text-base font-mono font-bold text-cyan-400">
                    {selectedTensor.anomaly_score?.toFixed(4)}
                  </span>
                </div>
                <p className="text-xs text-[var(--color-slate-300)] leading-relaxed pt-2 border-t border-white/5">
                  {selectedTensor.explanation || "No anomaly explanation triggered. Tensor conforms to clean baseline parameters."}
                </p>
              </div>

              {/* 10 Statistical Features Table */}
              <div>
                <h3 className="text-xs font-bold uppercase text-[var(--color-slate-400)] mb-3 tracking-wider">
                  Extracted Feature Vectors (F01 - F10)
                </h3>
                <div className="space-y-1.5 text-xs">
                  {[
                    { code: "F01", name: "LSB Entropy Deficit", val: selectedTensor.f01_entropy_deficit },
                    { code: "F02", name: "Pairs of Values (POV)", val: selectedTensor.f02_pairs_of_values },
                    { code: "F03", name: "Bit-Plane Bias", val: selectedTensor.f03_bit_plane_bias },
                    { code: "F04", name: "Serial Correlation", val: selectedTensor.f04_serial_correlation },
                    { code: "F05", name: "Runs Test Statistic", val: selectedTensor.f05_runs_test },
                    { code: "F06", name: "Bit Mutual Information", val: selectedTensor.f06_bit_mutual_info },
                    { code: "F07", name: "Trailing Zero Count", val: selectedTensor.f07_trailing_zero_count },
                    { code: "F08", name: "Zero-Field Fraction", val: selectedTensor.f08_zero_field_fraction },
                    { code: "F09", name: "FP16 Lattice Fraction", val: selectedTensor.f09_fp16_lattice_fraction },
                    { code: "F10", name: "Excess Kurtosis", val: selectedTensor.f10_excess_kurtosis },
                  ].map((feat) => (
                    <div
                      key={feat.code}
                      className="flex items-center justify-between p-2 rounded bg-[var(--color-navy-900)]/80 hover:bg-[var(--color-navy-900)] border border-white/5"
                    >
                      <div className="flex items-center gap-2">
                        <span className="font-mono text-cyan-400 font-bold">{feat.code}</span>
                        <span className="text-[var(--color-slate-300)]">{feat.name}</span>
                      </div>
                      <span className="font-mono font-semibold text-white">
                        {feat.val !== null && feat.val !== undefined ? feat.val.toFixed(4) : "—"}
                      </span>
                    </div>
                  ))}
                </div>
              </div>
            </>
          ) : (
            <div className="text-center py-16 text-[var(--color-slate-500)] text-xs">
              Select a tensor on the left to inspect its complete 10-feature statistical profile.
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
