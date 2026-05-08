// frontend/src/canvas/Boxes.tsx
import React, { useEffect, useMemo, useRef, useState } from "react";
import { Stage, Layer, Rect, Transformer, Image as KonvaImage } from "react-konva";
import type { Box, SavedBox } from "../types";

type Props = {
  pageIndex: number;
  image: HTMLImageElement | null;
  pagePx: { width: number; height: number };
  pdfDims: { widthPts: number; heightPts: number };
  savedBoxes: SavedBox[];
  currentBoxes: Box[];
  setCurrentBoxes: (b: Box[]) => void;
  onSelectSaved?: (eq_uid: string, boxId: string) => void;
  onSelectBox?: (boxId: string | null) => void;
  onDeleteSaved?: (boxId: string) => void;
  onSavedBoxChange?: (boxId: string, newBox: Box) => void;
};

export default function Boxes({
  pageIndex,
  image,
  pagePx,
  pdfDims,
  savedBoxes,
  currentBoxes,
  setCurrentBoxes,
  onSelectSaved,
  onSelectBox,
  onDeleteSaved,
  onSavedBoxChange,
}: Props) {
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const trRef = useRef<any>(null);
  const layerRef = useRef<any>(null);

  const scaleX = pagePx.width / (pdfDims.widthPts || 1);
  const scaleY = pagePx.height / (pdfDims.heightPts || 1);
  const pdfToPx = (x:number,y:number)=>({ x: x*scaleX, y: y*scaleY });
  const pxToPdf = (x:number,y:number)=>({ x: x/scaleX, y: y/scaleY });

  const currentRects = useMemo(()=>{
    return currentBoxes
      .map((b,i)=>({...b, id: b.id ?? `cur-${i}`}))
      .filter(b=>b.page===pageIndex)
      .map(b=>{
        const [x0,y0,x1,y1] = b.bbox_pdf;
        const p0 = pdfToPx(x0,y0), p1 = pdfToPx(x1,y1);
        const x = Math.min(p0.x, p1.x), y = Math.min(p0.y, p1.y);
        const w = Math.abs(p1.x - p0.x), h = Math.abs(p1.y - p0.y);
        return {...b, id:b.id!, x, y, w, h};
      });
  }, [currentBoxes, pageIndex, scaleX, scaleY]);

  const savedRects = useMemo(()=>{
    return savedBoxes
      .filter(b=>b.page===pageIndex)
      .map((b) => {
        const [x0,y0,x1,y1] = b.bbox_pdf;
        const p0 = pdfToPx(x0,y0), p1 = pdfToPx(x1,y1);
        const x = Math.min(p0.x, p1.x), y = Math.min(p0.y, p1.y);
        const w = Math.abs(p1.x - p0.x), h = Math.abs(p1.y - p0.y);
        return { ...b, x, y, w, h };
      });
  }, [savedBoxes, pageIndex, scaleX, scaleY]);

  useEffect(()=>{
    const tr = trRef.current;
    if (!tr) return;
    const stage = tr.getStage();
    const sel = stage?.findOne(`#${selectedId}`);
    if (sel) tr.nodes([sel]); else tr.nodes([]);
    tr.getLayer()?.batchDraw();
  }, [selectedId, currentRects, savedRects]);

  // Rubber-band draw state
  const [dragStart, setDragStart] = useState<{x:number;y:number}|null>(null);
  const [dragRect, setDragRect] = useState<{x:number;y:number;w:number;h:number}|null>(null);

  // --- NEW: Unified Selection Handler with Cycling ---
  function handleStageClick(e: any) {
    // 1. Ignore if clicking on Transformer
    if (e.target.getParent()?.className === 'Transformer') {
      return;
    }
    
    // 2. Ignore if we just finished drawing a box
    if (dragRect) return;

    // 3. Get click position in PDF coordinates
    const stage = e.target.getStage();
    const pos = stage.getPointerPosition();
    if (!pos) return;
    const clickPdf = pxToPdf(pos.x, pos.y);

    // 4. Find ALL intersecting boxes (Saved + Current)
    // A point (cx, cy) is inside [x0, y0, x1, y1] if x0 <= cx <= x1 AND y0 <= cy <= y1
    const cx = clickPdf.x;
    const cy = clickPdf.y;

    const hitSaved = savedRects.filter(r => {
        const [x0,y0,x1,y1] = r.bbox_pdf;
        return cx >= x0 && cx <= x1 && cy >= y0 && cy <= y1;
    }).map(r => ({ id: r.id, type: 'saved', uid: r.eq_uid }));

    const hitCurrent = currentRects.filter(r => {
        const [x0,y0,x1,y1] = r.bbox_pdf;
        return cx >= x0 && cx <= x1 && cy >= y0 && cy <= y1;
    }).map(r => ({ id: r.id, type: 'current', uid: null }));

    const allHits = [...hitSaved, ...hitCurrent];

    if (allHits.length === 0) {
        // Deselect
        setSelectedId(null);
        if (onSelectBox) onSelectBox(null);
        return;
    }

    // 5. Cycling Logic
    let nextId = allHits[0].id;
    
    // If we are currently selected on one of the hits, pick the NEXT one
    if (selectedId) {
        const currentIndex = allHits.findIndex(h => h.id === selectedId);
        if (currentIndex !== -1) {
            const nextIndex = (currentIndex + 1) % allHits.length;
            nextId = allHits[nextIndex].id;
        }
    }

    const hit = allHits.find(h => h.id === nextId)!;
    
    // Perform Selection
    setSelectedId(hit.id);
    if (onSelectBox) onSelectBox(hit.id);
    if (hit.type === 'saved' && onSelectSaved) {
        onSelectSaved(hit.uid!, hit.id);
    }
  }

  function onMouseDown(e:any){
    // Only start drag if clicking on background (not on an existing box/transformer)
    // But we still want to allow selection cycling on existing boxes.
    // So we assume onMouseDown is for DRAWING new boxes unless we clicked a box?
    // Actually, simpler: Start drag always, but only "draw" if we moved.
    if (e.target.name() !== "bg") return;
    
    const pos = e.target.getStage().getPointerPosition();
    if (!pos) return;
    setDragStart(pos);
  }

  function onMouseMove(e:any){
    if (!dragStart) return;
    const pos = e.target.getStage().getPointerPosition();
    if (!pos) return;
    setDragRect({
      x: Math.min(dragStart.x, pos.x),
      y: Math.min(dragStart.y, pos.y),
      w: Math.abs(pos.x - dragStart.x),
      h: Math.abs(pos.y - dragStart.y),
    });
  }

  function onMouseUp(e: any){
    // If we dragged a reasonable amount, create a new box
    if (dragRect && dragRect.w > 4 && dragRect.h > 4){
      const p0 = pxToPdf(dragRect.x, dragRect.y);
      const p1 = pxToPdf(dragRect.x + dragRect.w, dragRect.y + dragRect.h);
      const bbox:[number,number,number,number] = [
        Math.min(p0.x, p1.x), Math.min(p0.y, p1.y),
        Math.max(p0.x, p1.x), Math.max(p0.y, p1.y),
      ];
      const nb:Box = { page: pageIndex, bbox_pdf: bbox, id: `cur-${Date.now()}` };
      setCurrentBoxes([...currentBoxes, nb]);
      setSelectedId(nb.id!);
      if (onSelectBox) onSelectBox(nb.id!); 
    }
    
    setDragStart(null);
    setDragRect(null);
  }

  // ... [Keep nodeToPdfBBox and transform/drag handlers] ...
  function onSavedDragEnd(savedId: string, e:any){
    const node = e.target;
    const {x,y,width,height,scaleX: sx, scaleY: sy} = node.attrs;
    const w = width * sx, h = height * sy;
    const p0 = pxToPdf(x, y);
    const p1 = pxToPdf(x + w, y + h);
    const bbox:[number,number,number,number] = [
        Math.min(p0.x, p1.x), Math.min(p0.y, p1.y),
        Math.max(p0.x, p1.x), Math.max(p0.y, p1.y),
    ];
    if (onSavedBoxChange) onSavedBoxChange(savedId, { page: pageIndex, bbox_pdf: bbox, id: savedId });
  }

  function nodeToPdfBBox(node:any) {
    const scaleX = node.scaleX();
    const scaleY = node.scaleY();
    const x = node.x();
    const y = node.y();
    const width = node.width() * scaleX;
    const height = node.height() * scaleY;
    node.scaleX(1);
    node.scaleY(1);
    const p0 = pxToPdf(x, y);
   const p1 = pxToPdf(x + width, y + height);
   return [Math.min(p0.x, p1.x), Math.min(p0.y, p1.y), Math.max(p0.x, p1.x), Math.max(p0.y, p1.y)] as [number,number,number,number];
  }

  function onSavedTransformEnd(savedId: string, e:any){
    const node = e.target;
    const bbox = nodeToPdfBBox(node);
    if (onSavedBoxChange) onSavedBoxChange(savedId, { page: pageIndex, bbox_pdf: bbox, id: savedId });
  }

  function onRectTransformEnd(id:string, e:any){
    const node = e.target;
    const bbox = nodeToPdfBBox(node);
    setCurrentBoxes(currentBoxes.map(b => b.id===id ? {...b, bbox_pdf: bbox} : b));
  }

  function onRectDragMove(id:string, e:any){
    const node = e.target;
    const {x,y,width,height,scaleX,scaleY} = node.attrs;
    const w = width * scaleX, h = height * scaleY;
    const p0 = pxToPdf(x, y);
    const p1 = pxToPdf(x + w, y + h);
    const bbox:[number,number,number,number] = [
      Math.min(p0.x, p1.x), Math.min(p0.y, p1.y),
      Math.max(p0.x, p1.x), Math.max(p0.y, p1.y),
    ];
    setCurrentBoxes(currentBoxes.map(b => b.id===id ? {...b, bbox_pdf: bbox} : b));
  }

  function deleteSelected(){
    if (!selectedId) return;
    if (selectedId.startsWith("saved-")) {
      if (onDeleteSaved) onDeleteSaved(selectedId);
      setSelectedId(null);
      return;
    }
    setCurrentBoxes(currentBoxes.filter(b => b.id !== selectedId));
    setSelectedId(null);
    if (onSelectBox) onSelectBox(null);
  }

  return (
    <Stage
      width={pagePx.width}
      height={pagePx.height}
      onMouseDown={onMouseDown}
      onMouseMove={onMouseMove}
      onMouseUp={onMouseUp}
      onClick={handleStageClick} // <--- Unified Click Handler
      onTap={handleStageClick}
      tabIndex={0}
      onKeyDown={(e:any)=>{ if (e.key === "Delete") deleteSelected(); }}
      style={{border: "1px solid #ddd"}}
    >
      <Layer ref={layerRef}>
        <KonvaImage image={image || undefined} name="bg" listening={true}/>

        {savedRects.map(r => {
          const isSelected = r.id === selectedId;
          return (
            <Rect
              key={r.id}
              id={r.id}
              x={r.x}
              y={r.y}
              width={r.w}
              height={r.h}
              stroke={isSelected ? "red" : "gray"}
              dash={isSelected ? undefined : [4,4]}
              fill={isSelected ? "rgba(255,0,0,0.06)" : undefined}
              draggable={isSelected}
              // REMOVED onClick/onTap from Rect (handled by Stage)
              onDragEnd={(e)=>onSavedDragEnd(r.id, e)}
              onTransformEnd={(e)=>onSavedTransformEnd(r.id, e)}
            />
          );
        })}

        {currentRects.map(r => (
          <Rect key={r.id} id={r.id} x={r.x} y={r.y} width={r.w} height={r.h}
                stroke="red" fill="rgba(255,0,0,0.06)" draggable
                // REMOVED onClick/onTap from Rect (handled by Stage)
                onDragMove={(e)=>onRectDragMove(r.id!, e)}
                onTransformEnd={(e)=>onRectTransformEnd(r.id!, e)}
          />
        ))}

        <Transformer
          ref={trRef}
          rotateEnabled={false}
          enabledAnchors={["top-left", "top-center", "top-right", "middle-left", "middle-right", "bottom-left", "bottom-center", "bottom-right"]}
        />
        {dragRect && (
          <Rect x={dragRect.x} y={dragRect.y} width={dragRect.w} height={dragRect.h}
                stroke="#2a8" dash={[6,4]} listening={false}/>
        )}
      </Layer>
    </Stage>
  );
}