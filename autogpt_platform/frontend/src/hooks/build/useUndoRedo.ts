import { useCallback, useEffect, useState, useRef } from "react";
import { Edge, Node } from "@xyflow/react";
import isEqual from "lodash/isEqual";

const LOCAL_STORAGE_KEY = "build-state";
const MAX_HISTORY_LENGTH = 50;

interface CanvasState {
  nodes: Node<any>[];
  edges: Edge<any>[];
}

function useUndoRedo(initialState: CanvasState) {
  const [history, setHistory] = useState<CanvasState[]>([initialState]);
  const [pointer, setPointer] = useState(0);
  const isFirstLoad = useRef(true);
  const lastSavedState = useRef<string | null>(null);

  // Load state from localStorage only once on mount
  useEffect(() => {
    const cachedState = localStorage.getItem(LOCAL_STORAGE_KEY);
    if (cachedState && isFirstLoad.current) {
      try {
        const parsedHistory: CanvasState[] = JSON.parse(cachedState);
        if (parsedHistory.every(validateCanvasState)) {
          setHistory(parsedHistory);
          setPointer(parsedHistory.length - 1);
          lastSavedState.current = cachedState;
        }
      } catch (error) {
        console.error("Failed to parse cached state:", error);
      }
    }
    isFirstLoad.current = false;
  }, []);

  // Save state to localStorage
  useEffect(() => {
    if (isFirstLoad.current) return;

    const newState = JSON.stringify(history.slice(0, pointer + 1));
    if (newState !== lastSavedState.current) {
      lastSavedState.current = newState;
      localStorage.setItem(LOCAL_STORAGE_KEY, newState);
    }
  }, [history, pointer]);

  const addState = useCallback(
    (newState: CanvasState) => {
      if (isFirstLoad.current) return;

      setHistory((prevHistory) => {
        // If the new state is the same as the current state, don't add it
        const currentState = prevHistory[pointer];
        if (isEqual(currentState, newState)) return prevHistory;

        // Truncate the history to the current pointer and add the new state
        const newHistory = [...prevHistory.slice(0, pointer + 1), newState];

        // Maintain the maximum history length
        return newHistory.slice(-MAX_HISTORY_LENGTH);
      });

      // Move the pointer to the new end of history
      setPointer((prev) => {
        const newPointer = prev + 1;
        return Math.min(newPointer, MAX_HISTORY_LENGTH - 1);
      });
    },
    [pointer],
  );

  const undo = useCallback(() => {
    if (pointer > 0) {
      setPointer((prev) => prev - 1);
    }
  }, [pointer]);

  const redo = useCallback(() => {
    if (pointer < history.length - 1) {
      setPointer((prev) => prev + 1);
    }
  }, [pointer, history.length]);

  const reset = useCallback(() => {
    setHistory([initialState]);
    setPointer(0);
  }, [initialState]);

  return {
    history,
    current: history[pointer],
    undo,
    redo,
    canUndo: pointer > 0,
    canRedo: pointer < history.length - 1,
    addState,
    reset,
  };
}

const validateCanvasState = (state: CanvasState): boolean => {
  return (
    Array.isArray(state.nodes) &&
    Array.isArray(state.edges) &&
    state.nodes.every((node) => node && node.id && node.position) &&
    state.edges.every((edge) => edge && edge.id && edge.source && edge.target)
  );
};

export default useUndoRedo;
