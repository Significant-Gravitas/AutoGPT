"use server";

import { revalidatePath } from "next/cache";
import BackendAPI from "@/lib/autogpt-server-api/client";
import { OttoQuery, OttoResponse } from "@/lib/autogpt-server-api/types";

const api = new BackendAPI();

export async function askOtto(
  query: string,
  conversationHistory: { query: string; response: string }[],
  includeGraphData: boolean,
  graphId?: string,
): Promise<OttoResponse> {
  const messageId = `${Date.now()}-web`;

  const ottoQuery: OttoQuery = {
    query,
    conversation_history: conversationHistory,
    message_id: messageId,
    include_graph_data: includeGraphData,
    graph_id: graphId,
  };

  try {
    const response = await api.askOtto(ottoQuery);
    return response;
  } catch (error) {
    console.error("Error in askOtto server action:", error);
    return {
      answer: error instanceof Error ? error.message : "Unknown error occurred",
      documents: [],
      success: false,
      error: true,
    };
  }
}
