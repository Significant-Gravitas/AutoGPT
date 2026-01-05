-- CreateIndex
-- Index for efficient LLM model lookups on AgentNode.constantInput->>'model'
-- This improves performance of model migration queries in the LLM registry
CREATE INDEX "AgentNode_constantInput_model_idx" ON "AgentNode" ((("constantInput"->>'model')));
