import type { LlmModel, LlmProvider } from "@/lib/autogpt-server-api/types";

import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/atoms/Table/Table";

import { toggleLlmModelAction } from "../actions";
import { DeleteModelModal } from "./DeleteModelModal";
import { EditModelModal } from "./EditModelModal";

export function ModelsTable({
  models,
  providers,
}: {
  models: LlmModel[];
  providers: LlmProvider[];
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
            <TableHead>Context Window</TableHead>
            <TableHead>Max Output</TableHead>
            <TableHead>Cost</TableHead>
            <TableHead>Status</TableHead>
            <TableHead className="text-right">Actions</TableHead>
          </TableRow>
        </TableHeader>
        <TableBody>
          {models.map((model) => {
            const cost = model.costs[0];
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
                    className={`inline-flex rounded-full px-2 py-0.5 text-xs font-semibold ${
                      model.is_enabled
                        ? "bg-green-100 text-green-700"
                        : "bg-muted text-muted-foreground"
                    }`}
                  >
                    {model.is_enabled ? "Enabled" : "Disabled"}
                  </span>
                </TableCell>
                <TableCell className="text-right text-sm">
                  <div className="flex items-center justify-end gap-2">
                    <ToggleModelButton
                      modelId={model.id}
                      isEnabled={model.is_enabled}
                    />
                    <EditModelModal model={model} providers={providers} />
                    <DeleteModelModal
                      model={model}
                      availableModels={models}
                    />
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

function ToggleModelButton({
  modelId,
  isEnabled,
}: {
  modelId: string;
  isEnabled: boolean;
}) {
  return (
    <form action={toggleLlmModelAction}>
      <input type="hidden" name="model_id" value={modelId} />
      <input type="hidden" name="is_enabled" value={(!isEnabled).toString()} />
      <button
        type="submit"
        className="inline-flex items-center rounded border border-input px-3 py-1 text-xs font-semibold hover:bg-muted"
      >
        {isEnabled ? "Disable" : "Enable"}
      </button>
    </form>
  );
}


