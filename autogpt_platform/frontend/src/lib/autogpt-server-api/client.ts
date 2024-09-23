import { createClient } from "../supabase/client";
import BaseAutoGPTServerAPI from "./baseClient";

export default class AutoGPTServerAPI extends BaseAutoGPTServerAPI {
  constructor(
    baseUrl: string = process.env.NEXT_PUBLIC_AGPT_SERVER_URL ||
      "http://localhost:8006/api",
    wsUrl: string = process.env.NEXT_PUBLIC_AGPT_WS_SERVER_URL ||
      "ws://localhost:8001/ws",
  ) {
    const supabaseClient = createClient();
    super(baseUrl, wsUrl, supabaseClient);
  }
}
