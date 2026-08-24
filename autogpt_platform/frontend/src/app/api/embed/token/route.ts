import { z } from "zod";

import { verifyPartnerAssertion } from "@/lib/partner-embed/assertion";
import {
  getPartnerEmbedConfig,
  PartnerEmbedConfigurationError,
} from "@/lib/partner-embed/config";
import {
  mintPartnerEmbedToken,
  PARTNER_EMBED_TOKEN_TTL_SECONDS,
} from "@/lib/partner-embed/embed-token";
import { provisionPartnerIdentity } from "@/lib/partner-embed/provision";

const exchangeRequestSchema = z.object({
  assertion: z.string().min(1),
});

export async function POST(request: Request) {
  let body: unknown;
  try {
    body = await request.json();
  } catch {
    return Response.json({ error: "Invalid JSON body" }, { status: 400 });
  }

  const parsed = exchangeRequestSchema.safeParse(body);
  if (!parsed.success) {
    return Response.json(
      { error: "A partner assertion is required" },
      { status: 400 },
    );
  }

  let config;
  try {
    config = getPartnerEmbedConfig(parsed.data.assertion);
  } catch (error) {
    return Response.json(
      error instanceof PartnerEmbedConfigurationError
        ? { error: "Partner embedding is not configured" }
        : { error: "Invalid partner assertion" },
      { status: error instanceof PartnerEmbedConfigurationError ? 503 : 401 },
    );
  }

  let identity;
  try {
    identity = await verifyPartnerAssertion(parsed.data.assertion, config);
  } catch {
    return Response.json(
      { error: "Invalid partner assertion" },
      { status: 401 },
    );
  }

  try {
    const provisioned = await provisionPartnerIdentity(identity);
    const accessToken = await mintPartnerEmbedToken(identity, provisioned);
    return Response.json({
      access_token: accessToken,
      token_type: "Bearer",
      expires_in: PARTNER_EMBED_TOKEN_TTL_SECONDS,
    });
  } catch {
    return Response.json(
      { error: "Partner token exchange failed" },
      { status: 502 },
    );
  }
}
