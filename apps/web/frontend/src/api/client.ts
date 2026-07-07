const API = "http://127.0.0.1:8000";

import { DetectionCandidate, AutoDetectResponse } from "../types";


export async function uploadPdf(file: File): Promise<{ paper_id: string }> {
  const form = new FormData();
  form.append("file", file);

  const r = await fetch(`${API}/upload`, {
    method: "POST",
    body: form,
  });
  if (!r.ok) throw new Error(await r.text());
  return r.json();
}

export async function getPageCount(paperId: string) {
  const r = await fetch(`${API}/papers/${paperId}/pages`);
  if (!r.ok) throw new Error(await r.text());
  return r.json(); // { pages: number }
}

export function pageImageURL(
  paperId: string,
  idx: number,
  zoom: number = 1.5
) {
  return `${API}/papers/${paperId}/page/${idx}/image?zoom=${zoom}`;
}

export async function pageMeta(
  paperId: string,
  idx: number,
  zoom: number = 1.5
) {
  const r = await fetch(`${API}/papers/${paperId}/page/${idx}/meta?zoom=${zoom}`);
  if (!r.ok) throw new Error(await r.text());
  return r.json();
}

export async function listEquations(paperId: string) {
  const r = await fetch(`${API}/papers/${paperId}/equations`);
  if (!r.ok) throw new Error(await r.text());
  return r.json();
}

export async function saveEquation(paperId: string, payload: any) {
  const r = await fetch(`${API}/papers/${paperId}/equations`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (!r.ok) throw new Error(await r.text());
  return r.json();
}

export async function validateLatex(latex: string) {
  const r = await fetch(`${API}/validate`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ latex }),
  });
  if (!r.ok) throw new Error(await r.text());
  return r.json();
}

// frontend/src/api/client.ts
export async function updateEquation(paperId: string, eqUid: string, payload: any) {
  const r = await fetch(`${API}/papers/${paperId}/equations/${encodeURIComponent(eqUid)}`, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (!r.ok) throw new Error(await r.text());
  return r.json();
}

export async function deleteEquation(paperId: string, eqUid: string) {
  const r = await fetch(`${API}/papers/${paperId}/equations/${encodeURIComponent(eqUid)}`, {
    method: "DELETE",
  });
  if (!r.ok) throw new Error(await r.text());
  return r.json();
}

// frontend/src/api/client.ts
export async function findProfileByPdf(basename: string) {
  const url = `${API}/papers/find_by_pdf?basename=${encodeURIComponent(basename)}`;
  const r = await fetch(url);
  if (r.status === 404) return null;
  if (!r.ok) throw new Error(await r.text());
  return r.json();
}

export async function autodetectPage(paperId: string, pageIndex: number): Promise<DetectionCandidate[]> {
  const url = `${API}/papers/${paperId}/autodetect_page`;
  const res = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ page_index: pageIndex }),
  });

  if (!res.ok) throw new Error(`Auto-detect failed: ${res.statusText}`);
  
  const data: AutoDetectResponse = await res.json();
  return data.candidates;
}

export async function rescanBox(
  paperId: string, 
  pageIndex: number, 
  bbox: [number, number, number, number]
): Promise<{ latex: string }> {
  const url = `${API}/papers/${paperId}/rescan_box`;
  const res = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ page_index: pageIndex, bbox }),
  });

  if (!res.ok) throw new Error(`Rescan failed: ${res.statusText}`);
  return res.json();
}

export async function autodetectAll(paperId: string): Promise<{ equations_found: number }> {
  const url = `${API}/papers/${paperId}/autodetect_all`;
  const res = await fetch(url, { method: "POST" });
  if (!res.ok) throw new Error(`Scan failed: ${res.statusText}`);
  return res.json();
}