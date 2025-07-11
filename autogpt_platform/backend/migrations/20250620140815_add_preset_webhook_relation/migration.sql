-- Add webhookId column
ALTER TABLE "AgentPreset" ADD COLUMN     "webhookId" TEXT;

-- Add AgentPreset<->IntegrationWebhook relation
ALTER TABLE "AgentPreset" ADD CONSTRAINT "AgentPreset_webhookId_fkey" FOREIGN KEY ("webhookId") REFERENCES "IntegrationWebhook"("id") ON DELETE SET NULL ON UPDATE CASCADE;
