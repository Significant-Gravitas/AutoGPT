"use server";

import { revalidatePath } from "next/cache";
import BackendAPI from "../client";
import { OttoQuery, OttoResponse } from "../types";
import getServerSupabase from "@/lib/supabase/getServerSupabase";

const api = new BackendAPI();

export async function askOtto(
  query: string,
  conversationHistory: { query: string; response: string }[],
  includeGraphData: boolean,
  graphId?: string,
): Promise<OttoResponse> {
  const messageId = `${Date.now()}-web`;
  
  // Get the user ID from server-side Supabase client
  const supabase = getServerSupabase();
  const { data: { user } } = await supabase.auth.getUser();
  if (!user) {
    throw new Error("Authentication required");
  }

  const ottoQuery: OttoQuery = {
    query,
    conversation_history: conversationHistory,
    user_id: user.id,
    message_id: messageId,
    include_graph_data: includeGraphData,
    graph_id: graphId,
  };

  try {
    const response = await api.askOtto(ottoQuery);
    revalidatePath("/build");
    return response;
  } catch (error) {
    console.error("Error in askOtto server action:", error);
    throw error;
  }
}
