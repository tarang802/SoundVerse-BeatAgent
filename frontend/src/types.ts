// Shapes of the JSON the Python API returns (app/server.py). Keep in sync.

export interface LibraryItem {
  name: string;
  path: string;
  pack: string;
  synthetic: boolean;
  description: string;
}

export interface TempoSegment {
  start_beat: number;
  end_beat: number;
  start_bar: number;
  end_bar: number;
  start_time: number;
  bpm: number;
}

export interface GridSummary {
  duration_s: number;
  n_beats: number;
  last_beat_index: number;
  n_bars: number;
  pickup_beats_before_bar_1: number;
  time_signature: string;
  beats_per_bar: number;
  tempo_bpm: number;
  grid_type: "constant" | "variable";
  tempo_segments: TempoSegment[];
}

export interface DeckState {
  sid: string;
  version: number;
  state_id: number;
  can_undo: boolean;
  can_redo: boolean;
  history: string[];
  file: string;
  sr: number;
  subtype: string;
  channels: number;
  interpreter: string;
  summary: GridSummary;
  confidence: { overall?: number; beat_agreement?: number; downbeat_agreement?: number; note?: string };
  beats: number[];
  positions: number[];
  duration: number;
  peaks: string; // base64 int8 (min,max) pairs
  peak_rate: number;
  meta: Record<string, unknown>;
  analysis_s?: number;
  tempo_octave?: { tracker_bpm: number; reported_bpm: number; factor: number; preferred_range: number[] } | null;
  user_octave?: number | null;
}

export interface ChatReply {
  text: string;
  action: string;
  ok: boolean;
  echo: string[];
  warnings: string[];
  interpretation: string;
  backend: string;
  changed: boolean;
}

export interface SyncPlan {
  a_bar: number;
  b_bar: number;
  a_start: number;
  b_start: number;
  duration: number;
  a_period: number;
  b_period: number;
  time_ratio: number;
  a_bpm: number;
  b_bpm: number;
  b_needed: number;
  notes: string[];
  offset: number;
  stretch_percent: number;
}

export interface Alignment {
  n_beats: number;
  n_measured: number;
  median_ms?: number;
  median_abs_ms?: number;
  p95_abs_ms?: number;
  max_abs_ms?: number;
  worst_at_s?: number;
  offsets_ms?: number[];
  target?: { median_abs_ms: number; max_abs_ms: number };
  pass?: boolean;
  error?: string;
}

export interface SyncResult {
  id: string;
  a: string;
  b: string;
  method: string;
  render_s: number;
  plan: SyncPlan;
  alignment: Alignment;
  beats: number[];
  positions: number[];
  peaks: { a: string; b: string; mix: string };
  peak_rate: number;
}

export interface RecreateResult {
  id: string;
  name: string;
  tempo_bpm: number;
  time_signature: string;
  bars: number;
  loop_s: number;
  analysis: Record<string, unknown>;
  tracks: { name: string; kind: string; notes: number }[];
}
