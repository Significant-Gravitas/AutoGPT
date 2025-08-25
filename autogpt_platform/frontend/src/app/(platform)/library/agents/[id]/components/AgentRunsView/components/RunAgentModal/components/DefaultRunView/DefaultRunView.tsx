import { LibraryAgent } from "@/app/api/__generated__/models/libraryAgent";
import { RunVariant } from "../../useAgentRunModal";
import { WebhookTriggerBanner } from "../WebhookTriggerBanner/WebhookTriggerBanner";
import { AgentInputFields } from "../AgentInputFields/AgentInputFields";

interface Props {
  agent: LibraryAgent;
  defaultRunType: RunVariant;
  inputValues: Record<string, any>;
  onInputChange: (key: string, value: string) => void;
}

export function DefaultRunView({
  agent,
  defaultRunType,
  inputValues,
  onInputChange,
}: Props) {
  return (
    <div className="space-y-4">
      {defaultRunType === "automatic-trigger" && <WebhookTriggerBanner />}

      <AgentInputFields
        agent={agent}
        inputValues={inputValues}
        onInputChange={onInputChange}
      />
    </div>
  );
}
