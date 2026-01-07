import type { LlmModel } from "@/app/api/__generated__/models/llmModel";
import type { LlmModelCreator } from "@/app/api/__generated__/models/llmModelCreator";
import type { LlmProvider } from "@/app/api/__generated__/models/llmProvider";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/atoms/Table/Table";
import { Button } from "@/components/atoms/Button/Button";
import { toggleLlmModelAction } from "../actions";
import { DeleteModelModal } from "./DeleteModelModal";
import { DisableModelModal } from "./DisableModelModal";
import { EditModelModal } from "./EditModelModal";

export function ModelsTable({
  models,
  providers,
  creators,
}: {
  models: LlmModel[];
  providers: LlmProvider[];
  creators: LlmModelCreator[];
}) {
  if (!models.length) {
    return (
      <div className="rounded-lg border border-dashed border-border p-6 text-center text-sm text-muted-foreground">
        No models registered yet.
      </div>
    );
  }

  const providerLookup = new Map(
    providers.map((provider) => [provider.id, provider]),
  );

  return (
    <div className="rounded-lg border">
      <Table>
        <TableHeader>
          <TableRow>
            <TableHead>Model</TableHead>
            <TableHead>Provider</TableHead>
            <TableHead>Creator</TableHead>
            <TableHead>Context Window</TableHead>
            <TableHead>Max Output</TableHead>
            <TableHead>Cost</TableHead>
            <TableHead>Status</TableHead>
            <TableHead>Actions</TableHead>
          </TableRow>
        </TableHeader>
        <TableBody>
          {models.map((model) => {
            const cost = model.costs?.[0];
            const provider = providerLookup.get(model.provider_id);
            return (
              <TableRow
                key={model.id}
                className={model.is_enabled ? "" : "opacity-60"}
              >
                <TableCell>
                  <div className="font-medium">{model.display_name}</div>
                  <div className="text-xs text-muted-foreground">
                    {model.slug}
                  </div>
                </TableCell>
                <TableCell>
                  {provider ? (
                    <>
                      <div>{provider.display_name}</div>
                      <div className="text-xs text-muted-foreground">
                        {provider.name}
                      </div>
                    </>
                  ) : (
                    model.provider_id
                  )}
                </TableCell>
                <TableCell>
                  {model.creator ? (
                    <>
                      <div>{model.creator.display_name}</div>
                      <div className="text-xs text-muted-foreground">
                        {model.creator.name}
                      </div>
                    </>
                  ) : (
                    <span className="text-muted-foreground">—</span>
                  )}
                </TableCell>
                <TableCell>{model.context_window.toLocaleString()}</TableCell>
                <TableCell>
                  {model.max_output_tokens
                    ? model.max_output_tokens.toLocaleString()
                    : "—"}
                </TableCell>
                <TableCell>
                  {cost ? (
                    <>
                      <div className="font-medium">
                        {cost.credit_cost} credits
                      </div>
                      <div className="text-xs text-muted-foreground">
                        {cost.credential_provider}
                      </div>
                    </>
                  ) : (
                    "—"
                  )}
                </TableCell>
                <TableCell>
                  <span
                    className={`inline-flex rounded-full px-2.5 py-1 text-xs font-semibold ${
                      model.is_enabled
                        ? "bg-primary/10 text-primary"
                        : "bg-muted text-muted-foreground"
                    }`}
                  >
                    {model.is_enabled ? "Enabled" : "Disabled"}
                  </span>
                </TableCell>
                <TableCell>
                  <div className="flex items-center justify-end gap-2">
                    {model.is_enabled ? (
                      <DisableModelModal
                        model={model}
                        availableModels={models}
                      />
                    ) : (
                      <EnableModelButton modelId={model.id} />
                    )}
                    <EditModelModal
                      model={model}
                      providers={providers}
                      creators={creators}
                    />
                    <DeleteModelModal model={model} availableModels={models} />
                  </div>
                </TableCell>
              </TableRow>
            );
          })}
        </TableBody>
      </Table>
    </div>
  );
}

function EnableModelButton({ modelId }: { modelId: string }) {
  return (
    <form action={toggleLlmModelAction} className="inline">
      <input type="hidden" name="model_id" value={modelId} />
      <input type="hidden" name="is_enabled" value="true" />
      <Button type="submit" variant="outline" size="small" className="min-w-0">
        Enable
      </Button>
    </form>
  );
}
