// frontend/src/components/LaTeXPreview.tsx
import React from "react";
import katex from "katex";

type Props = {
  latex: string;
  errorMode?: "warn" | "silent";
};

export default function LaTeXPreview({ latex, errorMode = "warn" }: Props) {
  if (!latex || latex.trim() === "") {
    return <div style={{ color: "#666", fontStyle: "italic" }}>No LaTeX to render.</div>;
  }

  try {
    const html = katex.renderToString(latex, {
      throwOnError: false,
      displayMode: true, // or false depending on preference
    });
    return <div dangerouslySetInnerHTML={{ __html: html }} />;
  } catch (err: any) {
    if (errorMode === "warn") {
      return <div style={{ color: "red" }}>LaTeX render error: {String(err.message || err)}</div>;
    }
    return null;
  }
}
