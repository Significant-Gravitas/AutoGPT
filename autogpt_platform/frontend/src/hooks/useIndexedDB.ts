import Dexie, { type EntityTable } from "dexie";
import { Graph } from "@/lib/autogpt-server-api";
import { useCallback } from "react";

interface StorageData {
  id: number;
  agentId: string;
  graph: Graph;
  timestamp: number;
  unsaved: boolean;
}

class AgentDatabase extends Dexie {
  agents!: EntityTable<StorageData, "id">;

  constructor() {
    super("AgentDatabase");
    this.version(1).stores({
      agents: "++id, agentId, timestamp",
    });
  }
}

const db = new AgentDatabase(); // Singleton

export const useIndexedDB = (agentId: string | undefined) => {
  // Save new data
  const saveData = useCallback(
    async (data: any) => {
      try {
        const storageItem: Omit<StorageData, "id"> = {
          agentId: agentId ?? "",
          graph: data,
          timestamp: new Date().getTime(),
          unsaved: true,
        };

        const id = await db.agents.add(storageItem as StorageData);

        await db.agents
          .where("agentId")
          .equals(agentId ?? "")
          .and((item) => item.id !== id)
          .delete();

        return id;
      } catch (error) {
        console.error("Error saving data:", error);
        throw error;
      }
    },
    [agentId],
  );

  // Get data
  const getData = useCallback(async () => {
    try {
      if (!agentId) return null;

      const data = await db.agents.where("agentId").equals(agentId).first();

      return data || null;
    } catch (error) {
      console.error("Error getting data:", error);
      throw error;
    }
  }, [agentId]);

  // Clear all data for this agent
  const clearStore = useCallback(async () => {
    try {
      if (!agentId) return false;

      await db.agents.where("agentId").equals(agentId).delete();

      return true;
    } catch (error) {
      console.error("Error clearing store:", error);
      throw error;
    }
  }, [agentId]);

  return {
    saveData,
    getData,
    clearStore,
  };
};
