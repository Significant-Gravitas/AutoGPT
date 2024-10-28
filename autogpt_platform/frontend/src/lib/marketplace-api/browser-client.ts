import { createClient } from "../supabase/client";
import BaseMarketplaceAPI from "./base-client";

export default class ClientSideMarketplaceAPI extends BaseMarketplaceAPI {
  constructor(
    baseUrl: string = process.env.NEXT_PUBLIC_AGPT_MARKETPLACE_URL ||
      "http://localhost:8015/api/v1/market",
  ) {
    const supabaseClient = createClient();
    super(baseUrl, supabaseClient);
  }
}
