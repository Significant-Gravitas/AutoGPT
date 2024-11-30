import { Graph } from "@/lib/autogpt-server-api";
import { useCallback } from "react";

// Define your data type
interface StorageData {
  id: number;
  graph: Graph;
  timestamp: number;
  unsaved: boolean;
}

export const useLocalStorage = (agentId: string | undefined) => {
  // Unique key for each agent
  const STORAGE_KEY = `agent_${agentId}`;

  // Save new data after clearing old data
  const saveData = useCallback(
    (data: any) => {
      try {
        const storageItem: StorageData = {
          id: new Date().getTime(),
          graph: data,
          timestamp: new Date().getTime(),
          unsaved: true,
        };

        localStorage.setItem(STORAGE_KEY, JSON.stringify(storageItem));
        return storageItem.id;
      } catch (error) {
        console.error("Error saving data:", error);
        throw error;
      }
    },
    [STORAGE_KEY],
  );

  // Get data from localStorage
  const getData = useCallback(() => {
    try {
      const storageData = localStorage.getItem(STORAGE_KEY);
      if (!storageData) return null;

      return JSON.parse(storageData) as StorageData;
    } catch (error) {
      console.error("Error getting data:", error);
      throw error;
    }
  }, [STORAGE_KEY]);

  // Clear all data for this agent
  const clearStore = useCallback(() => {
    try {
      localStorage.removeItem(STORAGE_KEY);
      return true;
    } catch (error) {
      console.error("Error clearing store:", error);
      throw error;
    }
  }, [STORAGE_KEY]);

  return {
    saveData,
    getData,
    clearStore,
  };
};
