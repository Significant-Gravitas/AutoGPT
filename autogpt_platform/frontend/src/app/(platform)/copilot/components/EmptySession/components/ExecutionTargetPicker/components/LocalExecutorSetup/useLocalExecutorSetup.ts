"use client";

import { environment } from "@/services/environment";
import { useEffect, useState } from "react";
import { localExecutorDeployment } from "./helpers";

export function useLocalExecutorSetup() {
  const [deployment, setDeployment] = useState<ReturnType<
    typeof localExecutorDeployment
  > | null>(null);

  useEffect(function loadDeploymentURLs() {
    setDeployment(
      localExecutorDeployment(
        environment.getAGPTServerApiUrl(),
        window.location.origin,
      ),
    );
  }, []);

  return { deployment };
}
