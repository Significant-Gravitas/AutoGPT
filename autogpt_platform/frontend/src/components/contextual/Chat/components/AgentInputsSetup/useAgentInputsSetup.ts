import type { CredentialsMetaInput } from "@/lib/autogpt-server-api/types";
import { useState } from "react";

export function useAgentInputsSetup() {
  const [inputValues, setInputValues] = useState<Record<string, any>>({});
  const [credentialsValues, setCredentialsValues] = useState<
    Record<string, CredentialsMetaInput>
  >({});

  function setInputValue(key: string, value: any) {
    setInputValues((prev) => ({
      ...prev,
      [key]: value,
    }));
  }

  function setCredentialsValue(key: string, value?: CredentialsMetaInput) {
    if (value) {
      setCredentialsValues((prev) => ({
        ...prev,
        [key]: value,
      }));
    } else {
      setCredentialsValues((prev) => {
        const next = { ...prev };
        delete next[key];
        return next;
      });
    }
  }

  return {
    inputValues,
    setInputValue,
    credentialsValues,
    setCredentialsValue,
  };
}
