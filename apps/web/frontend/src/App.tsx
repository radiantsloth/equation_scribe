// frontend/src/App.tsx
import React, { useEffect, useMemo, useState, useCallback } from "react";
import PdfImage from "./pdf/PdfImage";
import Boxes from "./canvas/Boxes";
import {
  getPageCount,
  listEquations,
  saveEquation,
  updateEquation,
  deleteEquation,
  validateLatex,
  findProfileByPdf,
  uploadPdf,
  rescanBox,
} from "./api/client";
import type { Box, SavedBox, EquationRecord } from "./types";
import "katex/dist/katex.min.css";
import LaTeXPreview from "./components/LaTeXPreview";
import { AutoDetectButton } from "./components/AutoDetectButton";

export default function App() {
  const [paperId, setPaperId] = useState<string | null>(null);
  const [pages, setPages] = useState(0);
  const [pageIndex, setPageIndex] = useState(0);
  const [zoom, setZoom] = useState(1.5);
  const [img, setImg] = useState<HTMLImageElement | null>(null);
  const [pagePx, setPagePx] = useState({ width: 800, height: 1100 });
  const [status, setStatus] = useState<string>("No PDF loaded");
  const [pdfDims, setPdfDims] = useState<{ widthPts: number; heightPts: number }>({ widthPts: 0, heightPts: 0 });

  const [equations, setEquations] = useState<EquationRecord[]>([]);
  const [savedBoxes, setSavedBoxes] = useState<SavedBox[]>([]);
  const [currentBoxes, setCurrentBoxes] = useState<Box[]>([]);
  const [selectedEqUid, setSelectedEqUid] = useState<string | null>(null);
  const [selectedBoxId, setSelectedBoxId] = useState<string | null>(null);
  const [latex, setLatex] = useState("");
  const [notes, setNotes] = useState("");
  const hasPdf = !!paperId && pages > 0;

  // --- HELPER: Centralized State Loader ---
  const loadPaperData = async (pid: string) => {
    try {
      // 1. Load Equations from Backend
      const saved = await listEquations(pid);
      const eqs: EquationRecord[] = saved.items || [];
      setEquations(eqs);

      // 2. Rebuild Saved Boxes for Canvas
      const sBoxes: SavedBox[] = [];
      for (const eq of eqs) {
        (eq.boxes || []).forEach((b, idx) => {
          sBoxes.push({
            page: b.page,
            bbox_pdf: b.bbox_pdf,
            eq_uid: eq.eq_uid,
            box_idx: idx,
            id: `saved-${eq.eq_uid}-${idx}`,
          });
        });
      }
      setSavedBoxes(sBoxes);
      
      // 3. Clear working state (Red boxes) now that Gray boxes are loaded
      setCurrentBoxes([]);
      return true;
    } catch (err: any) {
      console.error(err);
      setStatus(`Error loading paper data: ${err.message}`);
      return false;
    }
  };

  const handleFileChange = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    try {
      setStatus(`Uploading "${file.name}"...`);
      const { paper_id } = await uploadPdf(file);
      setPaperId(paper_id);
      
      const { pages } = await getPageCount(paper_id);
      setPages(pages);
      setPageIndex(0);
      
      const success = await loadPaperData(paper_id);
      
      if (success) {
        setStatus(`Loaded "${file.name}" (${pages} pages). Restored saved state.`);
      } else {
        setStatus(`Loaded "${file.name}" (${pages} pages).`);
      }
      
      setSelectedEqUid(null);
      setSelectedBoxId(null);
      setLatex("");
      setNotes("");

    } catch (err: any) {
      console.error(err);
      setStatus(`Error uploading/loading PDF: ${err.message}`);
    }
  };

  const handleImageReady = useCallback((image: HTMLImageElement, meta: any) => {
    setImg(image);
    setPagePx({ width: meta.width_px, height: meta.height_px });
    setPdfDims({ widthPts: meta.width_pts, heightPts: meta.height_pts });
  }, []);

  const handleScanComplete = async () => {
    if (!paperId) return;
    await loadPaperData(paperId);
    setStatus("✅ Paper scan complete. Data reloaded.");
  };

  async function onValidate() {
    const r = await validateLatex(latex);
    setStatus(r.ok ? "✅ OK" : `❌ ${r.errors?.join("; ") || ""}`);
  }

  // --- SAVE HANDLER (Unified) ---
  async function onSave() {
    if (!paperId) {
      setStatus("❌ No PDF loaded.");
      return;
    }
    try {
      if (selectedEqUid) {
        // UPDATE EXISTING
        const existing = equations.find((e) => e.eq_uid === selectedEqUid);
        if (existing) {
          const updated: EquationRecord = {
            ...existing,
            paper_id: paperId,
            latex: latex,
            notes: notes,
            boxes: existing.boxes.map((b) => ({ page: b.page, bbox_pdf: b.bbox_pdf })),
          };
          await updateEquation(paperId, existing.eq_uid, updated);
          setStatus("✅ Updated existing equation.");
        }
      } else {
        // CREATE NEW
        if (currentBoxes.length === 0) {
          setStatus("❌ Add at least one box.");
          return;
        }
        const rec: EquationRecord = {
          eq_uid: crypto.randomUUID().slice(0, 16),
          paper_id: paperId,
          latex,
          notes,
          boxes: currentBoxes.map((b) => ({ page: b.page, bbox_pdf: b.bbox_pdf })),
        };
        await saveEquation(paperId, rec);
        setStatus(`✅ Saved ${currentBoxes.length} box(es).`);
      }
      
      await loadPaperData(paperId);
      
    } catch (err: any) {
      console.error(err);
      setStatus(`❌ Error saving: ${err.message}`);
    }
  }

  function handleSelectSaved(eq_uid: string, boxId: string) {
    setSelectedBoxId(boxId);
    setSelectedEqUid(eq_uid);
    const eq = equations.find((e) => e.eq_uid === eq_uid);
    if (eq) {
      setLatex(eq.latex || "");
      setNotes(eq.notes || "");
    }
  }

  function handleSavedBoxChange(boxId: string, newBox: Box) {
    setSavedBoxes((prev) =>
      prev.map((sb) => (sb.id === boxId ? { ...sb, bbox_pdf: newBox.bbox_pdf } : sb))
    );
    setEquations((prev) =>
      prev.map((eq) => {
        const updated = { ...eq };
        updated.boxes = updated.boxes.map((b, idx) => {
          const id = `saved-${eq.eq_uid}-${idx}`;
          return id === boxId ? { ...b, bbox_pdf: newBox.bbox_pdf } : b;
        });
        return updated;
      })
    );
  }

  async function handleDeleteSavedBox() {
    if (!paperId || !selectedBoxId) return;
    const sb = savedBoxes.find((s) => s.id === selectedBoxId);
    if (!sb) return;

    const { eq_uid, box_idx } = sb;
    const eq = equations.find((e) => e.eq_uid === eq_uid);
    if (!eq) return;

    const newBoxes = eq.boxes.filter((b, idx) => idx !== box_idx);
    try {
      if (newBoxes.length === 0) {
        await deleteEquation(paperId, eq_uid);
        setStatus("✅ Deleted equation.");
      } else {
        const updated = { ...eq, boxes: newBoxes };
        await updateEquation(paperId, eq_uid, updated);
        setStatus("✅ Deleted box.");
      }
      
      await loadPaperData(paperId);
      
      setSelectedBoxId(null);
      setSelectedEqUid(null);
      setLatex("");
      setNotes("");
    } catch (err: any) {
      setStatus(`❌ Delete error: ${err.message}`);
    }
  }

  async function handleRescanSelected() {
    if (!paperId || !selectedBoxId) return;
    let box: Box | undefined = savedBoxes.find((s) => s.id === selectedBoxId);
    if (!box) box = currentBoxes.find((c) => c.id === selectedBoxId);
    
    if (!box) {
      setStatus("❌ Select a box to rescan.");
      return;
    }
    setStatus("⏳ Scanning selection...");
    try {
      const result = await rescanBox(paperId, box.page, box.bbox_pdf);
      setLatex(result.latex);
      setStatus("✅ Rescan complete.");
    } catch (err: any) {
      setStatus(`❌ Rescan error: ${err.message}`);
    }
  }

  return (
    <div style={{ 
      display: "flex", 
      height: "100vh", 
      width: "100vw", // Ensure full width
      overflow: "hidden", 
      padding: 12, 
      gap: 12, 
      boxSizing: "border-box" // <--- FIX: This pulls the bottom edge up so it fits on screen
    }}>
      {/* LEFT: PDF viewport */}
      <div style={{ flex: "0 0 70%", maxWidth: "70%", minWidth: 600, borderRight: "1px solid #ddd", display: "flex", background: "#f5f5f5", overflow: "auto" }}>
        <div style={{ margin: "auto", position: "relative" }}>
          {hasPdf && (
            <PdfImage paperId={paperId!} pageIndex={pageIndex} zoom={zoom} onImageReady={handleImageReady} />
          )}
          <Boxes
            pageIndex={pageIndex}
            image={img}
            pagePx={pagePx}
            pdfDims={pdfDims}
            savedBoxes={savedBoxes}
            currentBoxes={currentBoxes}
            setCurrentBoxes={setCurrentBoxes}
            onSelectSaved={handleSelectSaved}
            onSelectBox={(id) => setSelectedBoxId(id)}
            onSavedBoxChange={handleSavedBoxChange}
            onDeleteSaved={(boxId: string) => {
              setSelectedBoxId(boxId);
              handleDeleteSavedBox();
            }}
          />
        </div>
      </div>

      {/* RIGHT: sidebar / controls */}
      <div style={{ flex: "0 0 30%", maxWidth: "30%", minWidth: 360, display: "flex", flexDirection: "column", gap: 8, overflowY: "auto" }}>
        <div style={{ marginBottom: 16 }}>
          <label>
            <strong>Load PDF: </strong>
            <input type="file" accept="application/pdf" onChange={handleFileChange} />
          </label>
          {paperId && <div style={{ marginTop: 8, fontSize: 12, color: "#555" }}>Loaded: <code>{paperId}</code></div>}
        </div>

        <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
          <button disabled={!hasPdf} onClick={() => setPageIndex((idx) => Math.max(0, idx - 1))}>◀</button>
          <span>{hasPdf ? `Page ${pageIndex + 1}/${pages}` : "No PDF loaded"}</span>
          <button disabled={!hasPdf} onClick={() => setPageIndex((idx) => Math.min(pages - 1, idx + 1))}>▶</button>
          <button disabled={!hasPdf} onClick={() => setZoom((z) => Math.max(0.5, z - 0.25))}>−</button>
          <span>Zoom {zoom.toFixed(2)}x</span>
          <button disabled={!hasPdf} onClick={() => setZoom((z) => z + 0.25)}>+</button>
        </div>

        {hasPdf && (
          <div style={{ marginBottom: 8, textAlign: "center" }}>
            <AutoDetectButton paperId={paperId} onScanComplete={handleScanComplete} />
          </div>
        )}

        <div style={{ border: "1px solid #eee", padding: 8 }}>
          <h3 style={{ marginTop: 0 }}>Equation Editor</h3>
          <textarea rows={5} value={latex} onChange={(e) => setLatex(e.target.value)} style={{ width: "100%" }} />
          <textarea rows={3} value={notes} onChange={(e) => setNotes(e.target.value)} style={{ width: "100%" }} placeholder="Notes..." />
          <div style={{ display: "flex", gap: 8, marginTop: 8, flexWrap: "wrap" }}>
            <button disabled={!hasPdf} onClick={onValidate}>Check</button>
            <button disabled={!hasPdf || !selectedBoxId} onClick={handleRescanSelected}>↻ Rescan</button>
            <button disabled={!hasPdf} onClick={onSave}>Approve & Save</button>
            <button disabled={!hasPdf} onClick={() => setCurrentBoxes([])}>Clear Boxes</button>
            <button disabled={!hasPdf || !selectedBoxId || !savedBoxes.some(sb => sb.id === selectedBoxId)} onClick={handleDeleteSavedBox}>Delete</button>  
          </div>
        </div>

        <div style={{ border: "1px solid #eee", padding: 8 }}>
          <div>Saved: {savedBoxes.filter((b) => b.page === pageIndex).length}</div>
          <div>Current: {currentBoxes.filter((b) => b.page === pageIndex).length}</div>
          <div style={{ marginTop: 8, color: "#666" }}>{status}</div>
        </div>

        <div style={{ marginTop: 12 }}>
          <label style={{ fontWeight: 600 }}>Rendered:</label>
          <div style={{ border: "1px solid #eee", padding: 8, minHeight: 48, background: "#fff" }}>
            <LaTeXPreview latex={latex} />
          </div>
        </div>
      </div>
    </div>
  );
}