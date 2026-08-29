import { useState, useCallback, useEffect, useRef } from "react";
import { parseAsString, parseAsInteger, useQueryStates } from "nuqs";
import {
  draftService,
  getTempFlowId,
  getOrCreateTempFlowId,
  DraftData,
} from "@/services/builder-draft/draft-service";
import { BuilderDraft } from "@/lib/dexie/db";
import {
  cleanNodes,
  cleanEdges,
  calculateDraftDiff,
  DraftDiff,
} from "@/lib/dexie/draft-utils";
import { useNodeStore } from "../../../stores/nodeStore";
import { useEdgeStore } from "../../../stores/edgeStore";
import { useGraphStore } from "../../../stores/graphStore";
import { useHistoryStore } from "../../../stores/historyStore";
import isEqual from "lodash/isEqual";

const AUTO_SAVE_INTERVAL_MS = 15000; // 15 seconds

interface DraftRecoveryState {
  isOpen: boolean;
  draft: BuilderDraft | null;
  diff: DraftDiff | null;
}

/**
 * Consolidated hook for draft persistence and recovery
 * - Auto-saves builder state every 15 seconds
 * - Saves on beforeunload event
 * - Checks for and manages unsaved drafts on load
 */
export function useDraftManager(isInitialLoadComplete: boolean) {
  const [state, setState] = useState<DraftRecoveryState>({
    isOpen: false,
    draft: null,
    diff: null,
  });

  const [{ flowID, flowVersion }] = useQueryStates({
    flowID: parseAsString,
    flowVersion: parseAsInteger,
  });

  const lastSavedStateRef = useRef<DraftData | null>(null);
  const saveTimeoutRef = useRef<NodeJS.Timeout | null>(null);
  const isDirtyRef = useRef(false);
  const hasCheckedForDraft = useRef(false);

  const getEffectiveFlowId = useCallback((): string => {
    return flowID || getOrCreateTempFlowId();
  }, [flowID]);

  const getCurrentState = useCallback((): DraftData => {
    const nodes = useNodeStore.getState().nodes;
    const edges = useEdgeStore.getState().edges;
    const nodeCounter = useNodeStore.getState().nodeCounter;
    const graphStore = useGraphStore.getState();

    return {
      nodes,
      edges,
      graphSchemas: {
        input: graphStore.inputSchema,
        credentials: graphStore.credentialsInputSchema,
        output: graphStore.outputSchema,
      },
      nodeCounter,
      flowVersion: flowVersion ?? undefined,
    };
  }, [flowVersion]);

  const cleanStateForComparison = useCallback((stateData: DraftData) => {
    return {
      nodes: cleanNodes(stateData.nodes),
      edges: cleanEdges(stateData.edges),
    };
  }, []);

  const hasChanges = useCallback((): boolean => {
    const currentState = getCurrentState();

    if (!lastSavedStateRef.current) {
      return currentState.nodes.length > 0;
    }

    const currentClean = cleanStateForComparison(currentState);
    const lastClean = cleanStateForComparison(lastSavedStateRef.current);

    return !isEqual(currentClean, lastClean);
  }, [getCurrentState, cleanStateForComparison]);

  const saveDraft = useCallback(async () => {
    const effectiveFlowId = getEffectiveFlowId();
    const currentState = getCurrentState();

    if (currentState.nodes.length === 0 && currentState.edges.length === 0) {
      return;
    }

    if (!hasChanges()) {
      return;
    }

    try {
      await draftService.saveDraft(effectiveFlowId, currentState);
      lastSavedStateRef.current = currentState;
      isDirtyRef.current = false;
    } catch (error) {
      console.error("[DraftPersistence] Failed to save draft:", error);
    }
  }, [getEffectiveFlowId, getCurrentState, hasChanges]);

  const scheduleSave = useCallback(() => {
    isDirtyRef.current = true;

    if (saveTimeoutRef.current) {
      clearTimeout(saveTimeoutRef.current);
    }

    saveTimeoutRef.current = setTimeout(() => {
      saveDraft();
    }, AUTO_SAVE_INTERVAL_MS);
  }, [saveDraft]);

  useEffect(() => {
    const unsubscribeNodes = useNodeStore.subscribe((storeState, prevState) => {
      if (storeState.nodes !== prevState.nodes) {
        scheduleSave();
      }
    });

    const unsubscribeEdges = useEdgeStore.subscribe((storeState, prevState) => {
      if (storeState.edges !== prevState.edges) {
        scheduleSave();
      }
    });

    return () => {
      unsubscribeNodes();
      unsubscribeEdges();
    };
  }, [scheduleSave]);

  useEffect(() => {
    const handleBeforeUnload = () => {
      if (isDirtyRef.current) {
        const effectiveFlowId = getEffectiveFlowId();
        const currentState = getCurrentState();

        if (
          currentState.nodes.length === 0 &&
          currentState.edges.length === 0
        ) {
          return;
        }

        draftService.saveDraft(effectiveFlowId, currentState).catch(() => {
          // Ignore errors on unload
        });
      }
    };

    window.addEventListener("beforeunload", handleBeforeUnload);
    return () => {
      window.removeEventListener("beforeunload", handleBeforeUnload);
    };
  }, [getEffectiveFlowId, getCurrentState]);

  useEffect(() => {
    return () => {
      if (saveTimeoutRef.current) {
        clearTimeout(saveTimeoutRef.current);
      }
      if (isDirtyRef.current) {
        saveDraft();
      }
    };
  }, [saveDraft]);

  useEffect(() => {
    draftService.cleanupExpired().catch((error) => {
      console.error(
        "[DraftPersistence] Failed to cleanup expired drafts:",
        error,
      );
    });
  }, []);

  const checkForDraft = useCallback(async () => {
    const effectiveFlowId = flowID || getTempFlowId();

    if (!effectiveFlowId) {
      return;
    }

    try {
      const draft = await draftService.loadDraft(effectiveFlowId);

      if (!draft) {
        return;
      }

      const currentNodes = useNodeStore.getState().nodes;
      const currentEdges = useEdgeStore.getState().edges;

      const isDifferent = draftService.isDraftDifferent(
        draft,
        currentNodes,
        currentEdges,
      );

      if (isDifferent && (draft.nodes.length > 0 || draft.edges.length > 0)) {
        const diff = calculateDraftDiff(
          draft.nodes,
          draft.edges,
          currentNodes,
          currentEdges,
        );
        setState({
          isOpen: true,
          draft,
          diff,
        });
      } else {
        await draftService.deleteDraft(effectiveFlowId);
      }
    } catch (error) {
      console.error("[DraftRecovery] Failed to check for draft:", error);
    }
  }, [flowID]);

  useEffect(() => {
    if (isInitialLoadComplete && !hasCheckedForDraft.current) {
      hasCheckedForDraft.current = true;
      checkForDraft();
    }
  }, [isInitialLoadComplete, checkForDraft]);

  useEffect(() => {
    hasCheckedForDraft.current = false;
    setState({
      isOpen: false,
      draft: null,
      diff: null,
    });
  }, [flowID]);

  const loadDraft = useCallback(async () => {
    if (!state.draft) return;

    const { draft } = state;

    try {
      useNodeStore.getState().setNodes(draft.nodes);
      useEdgeStore.getState().setEdges(draft.edges);
      draft.nodes.forEach((node) => {
        useNodeStore.getState().syncHardcodedValuesWithHandleIds(node.id);
      });

      if (draft.nodeCounter !== undefined) {
        useNodeStore.setState({ nodeCounter: draft.nodeCounter });
      }

      if (draft.graphSchemas) {
        useGraphStore
          .getState()
          .setGraphSchemas(
            draft.graphSchemas.input as Record<string, unknown> | null,
            draft.graphSchemas.credentials as Record<string, unknown> | null,
            draft.graphSchemas.output as Record<string, unknown> | null,
          );
      }

      setTimeout(() => {
        useHistoryStore.getState().initializeHistory();
      }, 100);

      await draftService.deleteDraft(draft.id);

      setState({
        isOpen: false,
        draft: null,
        diff: null,
      });
    } catch (error) {
      console.error("[DraftRecovery] Failed to load draft:", error);
    }
  }, [state.draft]);

  const discardDraft = useCallback(async () => {
    if (!state.draft) {
      setState({ isOpen: false, draft: null, diff: null });
      return;
    }

    try {
      await draftService.deleteDraft(state.draft.id);
    } catch (error) {
      console.error("[DraftRecovery] Failed to discard draft:", error);
    }

    setState({ isOpen: false, draft: null, diff: null });
  }, [state.draft]);

  return {
    // Recovery popup props
    isRecoveryOpen: state.isOpen,
    savedAt: state.draft?.savedAt ?? 0,
    nodeCount: state.draft?.nodes.length ?? 0,
    edgeCount: state.draft?.edges.length ?? 0,
    diff: state.diff,
    loadDraft,
    discardDraft,
  };
}
