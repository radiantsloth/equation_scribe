import React, { useState } from "react";
import { autodetectAll } from "../api/client";

interface Props {
  paperId: string;
  onScanComplete: () => void; // Changed from onCandidatesFound
}

export const AutoDetectButton: React.FC<Props> = ({ paperId, onScanComplete }) => {
  const [loading, setLoading] = useState(false);
  const [status, setStatus] = useState<string | null>(null);

  const handleClick = async () => {
    setLoading(true);
    setStatus("Scanning full paper... (this may take a while)");
    try {
      const res = await autodetectAll(paperId);
      setStatus(`Done! Found ${res.equations_found} equations.`);
      onScanComplete(); // Trigger reload in parent
    } catch (err: any) {
      console.error(err);
      setStatus("Scan failed.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={{ display: "inline-block", marginLeft: "1rem" }}>
      <button 
        onClick={handleClick} 
        disabled={loading}
        style={{ 
          backgroundColor: loading ? "#ccc" : "#28a745", // Green for "Go"
          color: "white", 
          padding: "8px 16px", 
          border: "none", 
          borderRadius: "4px", 
          cursor: "pointer" 
        }}
      >
        {loading ? "Scanning..." : "🚀 Scan Entire Paper"}
      </button>
      {status && <div style={{ marginTop: 4, fontSize: "0.8em", color: "#666" }}>{status}</div>}
    </div>
  );
};