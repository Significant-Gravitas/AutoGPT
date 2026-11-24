import { useGraphStore } from "@/app/(platform)/build/stores/graphStore";
import { useNodeStore } from "@/app/(platform)/build/stores/nodeStore";
import {
  getNodeCredentialsKey,
  getSavedNodeCredentials,
  getStepInputs,
} from "@/components/contextual/CredentialsInput/components/CredentialsGroupedView/node-credentials";
import type { CredentialsMetaInput } from "@/lib/autogpt-server-api/types";
import { useCallback, useState } from "react";
import { useShallow } from "zustand/react/shallow";

type CredentialValues = Record<string, CredentialsMetaInput>;

/**
 * The run dialog's credentials, which outlive the dialog. A key it picked
 * itself follows the steps' keys: when those change, it is picked again. A key
 * the user picked stays.
 */
export function useRunCredentials() {
  const subGraphs = useGraphStore((state) => state.loadedGraph?.sub_graphs);
  // The keys picked on the steps, the sub-workflows' steps included
  const nodeCredentials = useNodeStore(
    useShallow((state) =>
      getSavedNodeCredentials([
        ...state.nodes.map((node) => node.data.hardcodedValues),
        ...getStepInputs(subGraphs ?? []),
      ]),
    ),
  );
  const [credentialValues, setCredentialValues] = useState<CredentialValues>(
    {},
  );
  const [automaticFields, setAutomaticFields] = useState<string[]>([]);
  const nodeCredentialsKey = getNodeCredentialsKey(nodeCredentials);
  const [pickedFor, setPickedFor] = useState(nodeCredentialsKey);

  // Dropped while rendering, so the dialog picks them again in the same pass
  if (pickedFor !== nodeCredentialsKey) {
    setPickedFor(nodeCredentialsKey);
    setAutomaticFields([]);
    setCredentialValues((values) => withoutFields(values, automaticFields));
  }

  const handleCredentialFieldChange = useCallback(
    (key: string, value?: CredentialsMetaInput) => {
      setAutomaticFields((fields) => fields.filter((field) => field !== key));
      setCredentialValues((values) =>
        value ? { ...values, [key]: value } : withoutFields(values, [key]),
      );
    },
    [],
  );

  const handleAutoSelectCredential = useCallback(
    (key: string, value: CredentialsMetaInput) => {
      setAutomaticFields((fields) =>
        fields.includes(key) ? fields : [...fields, key],
      );
      setCredentialValues((values) => ({ ...values, [key]: value }));
    },
    [],
  );

  function replaceCredentialValues(values: CredentialValues) {
    setAutomaticFields([]);
    setCredentialValues(values);
  }

  return {
    nodeCredentials,
    credentialValues,
    handleCredentialFieldChange,
    handleAutoSelectCredential,
    replaceCredentialValues,
  };
}

function withoutFields(values: CredentialValues, fields: string[]) {
  return Object.fromEntries(
    Object.entries(values).filter(([key]) => !fields.includes(key)),
  );
}
