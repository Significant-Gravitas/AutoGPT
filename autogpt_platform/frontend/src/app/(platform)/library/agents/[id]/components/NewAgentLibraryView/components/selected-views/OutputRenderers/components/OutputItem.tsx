"use client";

import React from "react";
import { OutputRenderer, OutputMetadata } from "../types";

interface OutputItemProps {
  value: any;
  metadata?: OutputMetadata;
  renderer: OutputRenderer;
  label?: string;
}

export function OutputItem({
  value,
  metadata,
  renderer,
  label,
}: OutputItemProps) {
  return (
    <div className="relative">
      {label && (
        <label className="mb-1.5 block text-sm font-medium">{label}</label>
      )}

      <div className="relative">{renderer.render(value, metadata)}</div>
    </div>
  );
}
