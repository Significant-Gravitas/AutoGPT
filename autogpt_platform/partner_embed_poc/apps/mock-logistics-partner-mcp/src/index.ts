import { buildApp } from "./app.js";

const port = Number(process.env.PORT ?? "8790");
const host = process.env.HOST ?? "127.0.0.1";
const sharedSecret = process.env.PARTNER_MCP_SHARED_SECRET ?? "";

const app = buildApp({ sharedSecret });
await app.listen({ host, port });
