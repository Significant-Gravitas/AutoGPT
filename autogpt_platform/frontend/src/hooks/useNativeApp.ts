import { useEffect, useState } from "react";
import { isNativeAutoGPTApp } from "@/lib/oauth-popup-support";

export function useNativeApp() {
  const [isNativeApp, setIsNativeApp] = useState(false);
  useEffect(() => setIsNativeApp(isNativeAutoGPTApp()), []);
  return isNativeApp;
}
