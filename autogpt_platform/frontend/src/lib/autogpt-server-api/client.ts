import { SupabaseClient } from "@supabase/supabase-js";
import { createClient } from "../supabase/client";
import BaseAutoGPTServerAPI from "./baseClient";

export class AutoGPTServerAPI extends BaseAutoGPTServerAPI {
  constructor(
    baseUrl: string = process.env.NEXT_PUBLIC_AGPT_SERVER_URL ||
      "http://localhost:8006/api",
    wsUrl: string = process.env.NEXT_PUBLIC_AGPT_WS_SERVER_URL ||
      "ws://localhost:8001/ws",
    supabaseClient: SupabaseClient | null = createClient(),
  ) {
    super(baseUrl, wsUrl, supabaseClient);
  }
}
