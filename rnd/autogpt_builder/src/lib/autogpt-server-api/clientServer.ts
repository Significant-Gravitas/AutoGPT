import { createServerClient } from "../supabase/server";
import BaseAutoGPTServerAPI from "./baseClient";

export default class AutoGPTServerAPIServerSide extends BaseAutoGPTServerAPI {
  constructor(
    baseUrl: string = process.env.NEXT_PUBLIC_AGPT_SERVER_URL ||
      "http://localhost:8006/api",
    wsUrl: string = process.env.NEXT_PUBLIC_AGPT_WS_SERVER_URL ||
      "ws://localhost:8001/ws",
  ) {
    const supabaseClient = createServerClient();
    super(baseUrl, wsUrl, supabaseClient);
  }
}
