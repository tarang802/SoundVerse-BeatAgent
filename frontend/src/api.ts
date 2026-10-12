import type { ChatReply, DeckState, LibraryItem, RecreateChatReply, RecreateResult, SyncResult } from "./types";

// Every call surfaces the server's own error message (FastAPI puts it in
// `detail`), so the UI can show "Track A has bars 1-186; there is no bar 500."
// instead of a generic "400 Bad Request".
async function call<T>(url: string, init?: RequestInit): Promise<T> {
  const r = await fetch(url, init);
  let body: unknown = null;
  try {
    body = await r.json();
  } catch {
    /* non-JSON error page */
  }
  if (!r.ok) {
    const detail = (body as { detail?: string } | null)?.detail;
    throw new Error(detail ?? `${r.status} ${r.statusText}`);
  }
  return body as T;
}

const post = <T>(url: string, data?: unknown) =>
  call<T>(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: data === undefined ? undefined : JSON.stringify(data),
  });

export const api = {
  library: () => call<LibraryItem[]>("/api/library"),
  open: (path: string) => post<DeckState>("/api/open", { path }),
  upload: (file: File) => {
    const fd = new FormData();
    fd.append("file", file);
    return call<DeckState>("/api/upload", { method: "POST", body: fd });
  },
  chat: (sid: string, text: string, partner?: string) =>
    post<{ reply: ChatReply; state: DeckState; sync_id?: string }>(`/api/${sid}/chat`, { text, partner }),
  undo: (sid: string) => post<{ reply: ChatReply; state: DeckState }>(`/api/${sid}/undo`),
  redo: (sid: string) => post<{ reply: ChatReply; state: DeckState }>(`/api/${sid}/redo`),
  octave: (sid: string, factor: 2 | 0.5) => post<DeckState>(`/api/${sid}/octave`, { factor }),
  sync: (a: string, b: string, a_bar: number, b_bar: number, duration: number) =>
    post<SyncResult>("/api/sync", { a, b, a_bar, b_bar, duration }),
  audioUrl: (s: DeckState, click: boolean) => `/api/${s.sid}/audio/${s.state_id}.wav?click=${click ? 1 : 0}`,
  gridUrl: (s: DeckState) => `/api/${s.sid}/grid.json`,
  clickUrl: (s: DeckState) => `/api/${s.sid}/click.wav`,
  p3Library: () => call<LibraryItem[]>("/api/p3/library"),
  recreate: (req: { path?: string; sid?: string }) => post<RecreateResult>("/api/recreate", req),
  recreateUrl: (id: string, which: "page.html" | "score.json" | "original.wav") => `/api/recreate/${id}/${which}`,
  recreateChat: (id: string, text: string) => post<RecreateChatReply>(`/api/recreate/${id}/chat`, { text }),
  syncUrl: (id: string, which: "a" | "b" | "mix" | "report") =>
    `/api/sync/${id}/${which === "report" ? "report.json" : `${which}.wav`}`,
};
