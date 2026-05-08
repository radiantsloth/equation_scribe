// frontend/src/pdf/PdfImage.tsx
import React, { useEffect, useState } from "react";
import { pageImageURL, pageMeta } from "../api/client";

type Props = {
  paperId: string;
  pageIndex: number;
  zoom: number;
  onImageReady: (img: HTMLImageElement, meta: any) => void;
};

export default function PdfImage({ paperId, pageIndex, zoom, onImageReady }: Props) {
  const [src, setSrc] = useState<string>("");

  useEffect(() => {
    // console.log("PdfImage useEffect run", { paperId, pageIndex, zoom, onImageReady });
    let cancelled = false;

    async function run() {
      // Request page metadata for the given paper/page/zoom
      const meta = await pageMeta(paperId, pageIndex, zoom);

      // Build image URL for the given paper/page/zoom
      const url = pageImageURL(paperId, pageIndex, zoom);

      const img = new Image();
      img.onload = () => {
        if (!cancelled) onImageReady(img, meta);
      };
      img.onerror = (e) => {
        console.error("Failed to load PDF page image:", e);
      };
      img.src = url;
      setSrc(url);
    }

    run().catch(console.error);

    return () => {
      cancelled = true;
    };
  }, [paperId, pageIndex, zoom, onImageReady]);

  // We don't show the <img> directly; the Konva layer uses the HTMLImageElement
  return <img src={src} alt="pdf-page" style={{ display: "none" }} />;
}
