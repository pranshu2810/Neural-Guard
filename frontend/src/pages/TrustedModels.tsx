import { useEffect, useState } from "react";
import {
  ShieldCheck,
  Plus,
  UploadCloud,
  CheckCircle2,
  Loader2,
  AlertCircle,
} from "lucide-react";
import apiClient from "../api";
import type { Baseline } from "../api";

export default function TrustedModels() {
  const [baselines, setBaselines] = useState<Baseline[]>([]);
  const [loading, setLoading] = useState(true);
  const [showUpload, setShowUpload] = useState(false);
  const [name, setName] = useState("");
  const [description, setDescription] = useState("");
  const [file, setFile] = useState<File | null>(null);
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    loadBaselines();
  }, []);

  const loadBaselines = async () => {
    try {
      setLoading(true);
      const res = await apiClient.listBaselines();
      setBaselines(res.data);
    } catch (err) {
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  const handleCreate = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!name || !file) return;

    try {
      setSubmitting(true);
      setError(null);
      await apiClient.createBaseline(name, description, file);
      setShowUpload(false);
      setName("");
      setDescription("");
      setFile(null);
      await loadBaselines();
    } catch (err: any) {
      console.error(err);
      setError(err.response?.data?.detail || "Failed to register baseline checkpoint.");
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <div className="space-y-8 animate-fade-in">
      {/* Header */}
      <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center gap-4">
        <div>
          <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-emerald-500/10 border border-emerald-500/30 text-emerald-400 text-xs font-semibold uppercase tracking-wider mb-2">
            <ShieldCheck className="w-3.5 h-3.5" />
            Golden Baselines Registry
          </div>
          <h1 className="text-2xl font-bold text-white">
            Trusted Reference Models
          </h1>
          <p className="text-xs text-[var(--color-slate-400)] mt-1">
            Maintain verified clean checkpoints to establish empirical statistical means and covariance matrices.
          </p>
        </div>

        <button
          onClick={() => setShowUpload(!showUpload)}
          className="btn-primary text-xs flex items-center gap-1.5"
        >
          <Plus className="w-3.5 h-3.5" />
          {showUpload ? "Cancel" : "Add Trusted Baseline"}
        </button>
      </div>

      {/* Upload Modal / Drawer Form */}
      {showUpload && (
        <div className="glass-card p-6 border-cyan-500/30 space-y-4">
          <h2 className="text-base font-bold text-white">
            Register New Golden Baseline
          </h2>
          <p className="text-xs text-[var(--color-slate-400)]">
            Upload an uncorrupted reference model. Its weight tensors will be parsed to compute standard distribution statistics.
          </p>

          <form onSubmit={handleCreate} className="space-y-4">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div>
                <label className="text-xs font-semibold text-[var(--color-slate-300)] block mb-1">
                  Baseline Name
                </label>
                <input
                  type="text"
                  required
                  placeholder="e.g., Llama-3-8B-Instruct-Official"
                  value={name}
                  onChange={(e) => setName(e.target.value)}
                  className="w-full px-3 py-2 bg-[var(--color-navy-800)] border border-white/10 rounded-lg text-xs text-white placeholder-[var(--color-slate-500)] focus:outline-none focus:border-cyan-500/50"
                />
              </div>

              <div>
                <label className="text-xs font-semibold text-[var(--color-slate-300)] block mb-1">
                  Description
                </label>
                <input
                  type="text"
                  placeholder="Official HuggingFace weights release v1.0"
                  value={description}
                  onChange={(e) => setDescription(e.target.value)}
                  className="w-full px-3 py-2 bg-[var(--color-navy-800)] border border-white/10 rounded-lg text-xs text-white placeholder-[var(--color-slate-500)] focus:outline-none focus:border-cyan-500/50"
                />
              </div>
            </div>

            <div>
              <label className="text-xs font-semibold text-[var(--color-slate-300)] block mb-1">
                Model File (.safetensors, .pt, .onnx)
              </label>
              <input
                type="file"
                required
                onChange={(e) => setFile(e.target.files?.[0] || null)}
                className="w-full px-3 py-2 bg-[var(--color-navy-800)] border border-white/10 rounded-lg text-xs text-white file:mr-4 file:py-1 file:px-3 file:rounded-md file:border-0 file:text-xs file:bg-cyan-500/20 file:text-cyan-400 cursor-pointer"
              />
            </div>

            {error && (
              <div className="p-3 rounded-lg bg-rose-500/10 border border-rose-500/30 text-rose-300 text-xs flex items-center gap-2">
                <AlertCircle className="w-4 h-4 shrink-0" />
                {error}
              </div>
            )}

            <div className="flex justify-end gap-3 pt-2">
              <button
                type="button"
                onClick={() => setShowUpload(false)}
                className="btn-secondary text-xs"
              >
                Cancel
              </button>
              <button
                type="submit"
                disabled={submitting}
                className="btn-primary text-xs"
              >
                {submitting ? (
                  <>
                    <Loader2 className="w-3.5 h-3.5 animate-spin" /> Registering...
                  </>
                ) : (
                  <>
                    <UploadCloud className="w-3.5 h-3.5" /> Save Baseline
                  </>
                )}
              </button>
            </div>
          </form>
        </div>
      )}

      {/* Baselines List */}
      {loading ? (
        <div className="flex flex-col items-center justify-center py-16">
          <Loader2 className="w-8 h-8 text-cyan-400 animate-spin" />
        </div>
      ) : baselines.length > 0 ? (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {baselines.map((base) => (
            <div key={base.id} className="glass-card p-6 flex flex-col justify-between">
              <div>
                <div className="flex items-center justify-between mb-2">
                  <span className="text-[10px] font-mono uppercase tracking-wider text-emerald-400 flex items-center gap-1">
                    <CheckCircle2 className="w-3 h-3" /> Verified Baseline
                  </span>
                  <span className="text-[10px] font-mono text-[var(--color-slate-400)] uppercase">
                    {base.model_format}
                  </span>
                </div>
                <h3 className="text-base font-bold text-white mb-1">{base.name}</h3>
                <p className="text-xs text-[var(--color-slate-400)] mb-4 leading-relaxed">
                  {base.description || "Reference model checkpoint for statistical Mahalanobis distance covariance calibration."}
                </p>
              </div>

              <div className="pt-4 border-t border-white/5 space-y-2 text-xs">
                <div className="flex justify-between text-[var(--color-slate-400)]">
                  <span>Tensors Indexed:</span>
                  <span className="font-mono text-white font-semibold">
                    {base.num_tensors || "All"}
                  </span>
                </div>
                <div className="flex justify-between text-[var(--color-slate-400)]">
                  <span>Registered:</span>
                  <span className="text-[var(--color-slate-300)]">
                    {base.created_at ? new Date(base.created_at).toLocaleDateString() : "Active"}
                  </span>
                </div>
                {base.file_hash_sha256 && (
                  <div className="flex justify-between text-[var(--color-slate-400)]">
                    <span>SHA-256:</span>
                    <span className="font-mono text-[10px] text-cyan-400">
                      {base.file_hash_sha256.slice(0, 10)}...
                    </span>
                  </div>
                )}
              </div>
            </div>
          ))}
        </div>
      ) : (
        <div className="glass-card p-12 text-center max-w-lg mx-auto space-y-4">
          <ShieldCheck className="w-12 h-12 text-cyan-400 mx-auto opacity-70" />
          <h2 className="text-base font-bold text-white">No Custom Baselines Stored</h2>
          <p className="text-xs text-[var(--color-slate-400)]">
            Neural Guard currently uses its built-in default empirical calibration thresholds. You can register golden reference models here to customize threshold calibration for specific architectures.
          </p>
          <button
            onClick={() => setShowUpload(true)}
            className="btn-primary text-xs inline-flex"
          >
            <Plus className="w-3.5 h-3.5" /> Add First Baseline
          </button>
        </div>
      )}
    </div>
  );
}
