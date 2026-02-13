"use client";

import { useState, useEffect, useRef } from "react";
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
import { Star, Spinner } from "@phosphor-icons/react";
import { getV2ListLlmModels } from "@/app/api/__generated__/endpoints/admin/admin";

const PAGE_SIZE = 50;

export function ModelsTable({
  models: initialModels,
  providers,
  creators,
}: {
  models: LlmModel[];
  providers: LlmProvider[];
  creators: LlmModelCreator[];
}) {
  const [models, setModels] = useState<LlmModel[]>(initialModels);
  const [currentPage, setCurrentPage] = useState(1);
  const [hasMore, setHasMore] = useState(initialModels.length === PAGE_SIZE);
  const [isLoading, setIsLoading] = useState(false);
  const loadedPagesRef = useRef(1);

  // Sync with parent when initialModels changes (e.g., after enable/disable)
  // Re-fetch all loaded pages to preserve expanded state
  useEffect(() => {
    async function refetchAllPages() {
      const pagesToLoad = loadedPagesRef.current;

      if (pagesToLoad === 1) {
        // Only first page loaded, just use initialModels
        setModels(initialModels);
        setHasMore(initialModels.length === PAGE_SIZE);
        return;
      }

      // Re-fetch all pages we had loaded
      const allModels: LlmModel[] = [...initialModels];
      let lastPageHadFullResults = initialModels.length === PAGE_SIZE;

      for (let page = 2; page <= pagesToLoad; page++) {
        try {
          const response = await getV2ListLlmModels({
            page,
            page_size: PAGE_SIZE,
          });
          if (response.status === 200) {
            allModels.push(...response.data.models);
            lastPageHadFullResults = response.data.models.length === PAGE_SIZE;
          }
        } catch (err) {
          console.error(`Error refetching page ${page}:`, err);
          break;
        }
      }

      setModels(allModels);
      setHasMore(lastPageHadFullResults);
    }

    refetchAllPages();
  }, [initialModels]);

  async function loadMore() {
    if (isLoading) return;
    setIsLoading(true);

    try {
      const nextPage = currentPage + 1;
      const response = await getV2ListLlmModels({
        page: nextPage,
        page_size: PAGE_SIZE,
      });

      if (response.status === 200) {
        setModels((prev) => [...prev, ...response.data.models]);
        setCurrentPage(nextPage);
        loadedPagesRef.current = nextPage;
        setHasMore(response.data.models.length === PAGE_SIZE);
      }
    } catch (err) {
      console.error("Error loading more models:", err);
    } finally {
      setIsLoading(false);
    }
  }
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
    <div>
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
                    <div className="flex flex-col gap-1">
                      <span
                        className={`inline-flex rounded-full px-2.5 py-1 text-xs font-semibold ${
                          model.is_enabled
                            ? "bg-primary/10 text-primary"
                            : "bg-muted text-muted-foreground"
                        }`}
                      >
                        {model.is_enabled ? "Enabled" : "Disabled"}
                      </span>
                      {model.is_recommended && (
                        <span className="inline-flex items-center gap-1 rounded-full bg-amber-500/10 px-2.5 py-1 text-xs font-semibold text-amber-600 dark:text-amber-400">
                          <Star size={12} weight="fill" />
                          Recommended
                        </span>
                      )}
                    </div>
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

      {hasMore && (
        <div className="mt-4 flex justify-center">
          <Button onClick={loadMore} disabled={isLoading} variant="outline">
            {isLoading ? (
              <>
                <Spinner className="mr-2 h-4 w-4 animate-spin" />
                Loading...
              </>
            ) : (
              "Load More"
            )}
          </Button>
        </div>
      )}
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
