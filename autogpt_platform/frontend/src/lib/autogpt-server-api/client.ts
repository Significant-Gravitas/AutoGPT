import { createClient } from "../supabase/client";
import { createServerClient } from "../supabase/server";
import BaseAutoGPTServerAPI from "./baseClient";

class AutoGPTServerAPIClientSide extends BaseAutoGPTServerAPI {
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

class AutoGPTServerAPIServerSide extends BaseAutoGPTServerAPI {
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

export const AutoGPTServerAPI =
  typeof window !== "undefined"
    ? AutoGPTServerAPIClientSide
    : AutoGPTServerAPIServerSide;

export type AutoGPTServerAPI =
  | AutoGPTServerAPIClientSide
  | AutoGPTServerAPIServerSide;
