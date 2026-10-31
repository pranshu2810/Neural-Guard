/**
 * API Client for Neural Guard backend.
 */

import axios from "axios";

const api = axios.create({
  baseURL: "/api/v1",
  timeout: 120000, // 2 min for large model scans
});

// ─── Types ──────────────────────────────────────────────────────

export interface ScanRecord {
  id: string;
  filename: string;
  file_hash_sha256: string | null;
  file_size_bytes: number | null;
  model_format: string;
  status: string;
  created_at: string | null;
  completed_at: string | null;
  total_tensors: number;
  total_parameters: number;
  anomaly_score: number | null;
  risk_classification: string | null;
  dtype_distribution: Record<string, number> | null;
  warnings: string[] | null;
  error_message: string | null;
}

export interface TensorResult {
  id: string;
  scan_id: string;
  tensor_name: string;
  tensor_shape: number[] | null;
  tensor_dtype: string | null;
  num_elements: number | null;
  f01_entropy_deficit: number | null;
  f02_pairs_of_values: number | null;
  f03_bit_plane_bias: number | null;
  f04_serial_correlation: number | null;
  f05_runs_test: number | null;
  f06_bit_mutual_info: number | null;
  f07_trailing_zero_count: number | null;
  f08_zero_field_fraction: number | null;
  f09_fp16_lattice_fraction: number | null;
  f10_excess_kurtosis: number | null;
  anomaly_score: number | null;
  risk_classification: string | null;
  feature_contributions: Record<string, number> | null;
  explanation: string | null;
  warnings: string[] | null;
}

export interface DashboardStats {
  total_scans: number;
  normal_count: number;
  suspicious_count: number;
  high_anomaly_count: number;
  insufficient_count: number;
  failed_count: number;
  avg_anomaly_score: number;
  total_parameters_analyzed: number;
  recent_scans: ScanRecord[];
  risk_distribution: Record<string, number>;
}

export interface Baseline {
  id: string;
  name: string;
  description: string | null;
  file_hash_sha256: string | null;
  model_format: string | null;
  created_at: string | null;
  num_tensors: number | null;
  metadata: Record<string, unknown> | null;
}

export interface ExplanationData {
  tensor_name: string;
  tensor_shape: number[] | null;
  tensor_dtype: string | null;
  anomaly_score: number;
  risk_classification: string;
  explanation: string;
  feature_contributions: Record<string, number>;
  feature_values: Record<string, number | null>;
  warnings: string[] | null;
}

export interface ExperimentResult {
  experiment_type: string;
  n_clean: number;
  n_embedded: number;
  tensor_shape: number[];
  train_size: number;
  calibration_size: number;
  test_size: number;
  thresholds: { suspicious: number; high_anomaly: number };
  metrics: Record<string, unknown>;
  disclaimer: string;
}

// ─── API Functions ──────────────────────────────────────────────

export const apiClient = {
  // Health
  health: () => api.get("/health"),

  // Dashboard
  getStats: () => api.get<DashboardStats>("/dashboard/stats"),

  // Scanning
  uploadScan: (file: File) => {
    const formData = new FormData();
    formData.append("file", file);
    return api.post<ScanRecord>("/scan/upload", formData, {
      headers: { "Content-Type": "multipart/form-data" },
    });
  },

  getScan: (scanId: string) => api.get<ScanRecord>(`/scan/${scanId}`),

  getScanTensors: (scanId: string, sortBy = "anomaly_score", order = "desc") =>
    api.get<TensorResult[]>(`/scan/${scanId}/tensors`, {
      params: { sort_by: sortBy, order, limit: 500 },
    }),

  getExplanations: (scanId: string) =>
    api.get<{ scan_id: string; explanations: ExplanationData[] }>(
      `/scan/${scanId}/explanations`
    ),

  getReport: (scanId: string, format: "json" | "pdf" = "json") =>
    api.get(`/scan/${scanId}/report`, {
      params: { format },
      responseType: format === "pdf" ? "blob" : "json",
    }),

  listScans: (limit = 50, offset = 0) =>
    api.get<ScanRecord[]>("/scans", { params: { limit, offset } }),

  // Baselines
  createBaseline: (name: string, description: string, file: File) => {
    const formData = new FormData();
    formData.append("file", file);
    return api.post<Baseline>("/baselines", formData, {
      params: { name, description },
      headers: { "Content-Type": "multipart/form-data" },
    });
  },

  listBaselines: () => api.get<Baseline[]>("/baselines"),

  // Compare
  compare: (scanId: string, baselineId: string) =>
    api.post("/compare", { scan_id: scanId, baseline_id: baselineId }),

  // Experiments
  runExperiment: (params: {
    n_clean: number;
    n_embedded: number;
    tensor_shape: number[];
    seed: number;
  }) => api.post<ExperimentResult>("/experiments/run", params),
};

export default apiClient;
