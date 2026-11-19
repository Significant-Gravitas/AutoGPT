import { useMutation } from "@tanstack/react-query";
import { useState } from "react";

import {
  createCredential,
  storeMCPToken,
  type CredentialMeta,
} from "@/lib/api/integrations";
import { sessionGeneration } from "@/lib/session";

import { rememberCredential } from "../../oauth";
import {
  buildCredentialBody,
  canSave,
  initialValues,
  saveErrorText,
  serverSchemes,
  type SheetTarget,
  type SheetValues,
} from "./helpers";

interface Args {
  target: SheetTarget;
  onSaved: (rowKey: string, credential: CredentialMeta) => void;
}

export function useCredentialSheet({ target, onSaved }: Args) {
  const [values, setValues] = useState(() => initialValues(target));

  const save = useMutation({
    onMutate() {
      return sessionGeneration();
    },
    mutationFn() {
      if (target.kind === "mcp_token") {
        return storeMCPToken(target.serverURL ?? "", values.secret.trim());
      }
      const body = buildCredentialBody(target, values);
      if (!body) throw new Error("Nothing to save");
      return createCredential(target.provider, body);
    },
    onSuccess(credential, _variables, startedIn) {
      // Saved for an account that has since signed out: not this one's to show.
      if (startedIn !== sessionGeneration()) return;
      rememberCredential(credential);
      onSaved(target.rowKey, credential);
    },
  });

  // Each field keeps exactly what was typed; trimming happens on save only.
  function change(field: keyof SheetValues) {
    return function handleChange(text: string) {
      setValues((previous) => ({ ...previous, [field]: text }));
    };
  }

  function submit() {
    if (!canSave(target, values) || save.isPending) return;
    save.mutate();
  }

  return {
    values,
    change,
    schemes: serverSchemes(target.provider),
    canSave: canSave(target, values),
    isSaving: save.isPending,
    error: save.isError ? saveErrorText(save.error) : null,
    submit,
  };
}
