import { BlockIOCredentialsSubSchema } from "@/lib/autogpt-server-api/types";

export type AgentCredentialsFields = Record<
  string,
  BlockIOCredentialsSubSchema
>;
