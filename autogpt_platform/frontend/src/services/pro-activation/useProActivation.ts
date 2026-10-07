"use client";
import { useContext } from "react";
import { ProActivationContext } from "./context";

export function useProActivation() {
  const context = useContext(ProActivationContext);
  if (!context)
    throw new Error("useProActivation requires ProActivationProvider");
  return context;
}
