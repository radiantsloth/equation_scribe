// frontend/src/types.ts

export type BBox = [number, number, number, number];

export type Box = {
  page: number;
  bbox_pdf: BBox;
  id?: string;
};

// SavedBox includes which equation it belongs to and which index
export type SavedBox = Box & {
  eq_uid: string;
  box_idx: number;
  id: string;
};

export type EquationRecord = {
  eq_uid: string;
  paper_id: string;
  latex: string;
  notes: string;
  boxes: Box[];
};

// --- NEW TYPES FOR SPIRAL 3 ---
export interface DetectionCandidate {
  bbox_pdf: [number, number, number, number];
  latex: string;
  score: number;
}

export interface AutoDetectResponse {
  candidates: DetectionCandidate[];
}