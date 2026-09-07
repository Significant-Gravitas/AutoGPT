"use client";

import {
  Avatar,
  AvatarFallback,
  AvatarImage,
} from "@/components/atoms/Avatar/Avatar";
import { Button } from "@/components/atoms/Button/Button";
import { Input } from "@/components/atoms/Input/Input";
import { Text } from "@/components/atoms/Text/Text";
import { Dialog } from "@/components/molecules/Dialog/Dialog";
import { Icon } from "@/components/atoms/Icon/Icon";
import { WorkflowSquare01Icon } from "@hugeicons/core-free-icons";
import { INSTALL_WORKFLOW_SOURCES, workflowSubtitle } from "./helpers";
import { useInstallWorkflowPicker } from "./useInstallWorkflowPicker";

interface Props {
  mode: "pick-expert" | "pick-workflow";
  storeListingVersionId?: string;
  expertId?: string;
  open: boolean;
  onClose: () => void;
}

export function InstallWorkflowPicker({
  mode,
  storeListingVersionId,
  expertId,
  open,
  onClose,
}: Props) {
  const {
    title,
    hiredExperts,
    source,
    setSource,
    searchQuery,
    setSearchQuery,
    libraryResults,
    hasMoreLibraryResults,
    loadMoreLibraryResults,
    isLoadingMore,
    marketplaceResults,
    isSearching,
    pendingKey,
    installOnExpert,
    installLibraryAgent,
    installFromListing,
  } = useInstallWorkflowPicker({
    mode,
    storeListingVersionId,
    expertId,
    open,
    onClose,
  });

  const isEmpty =
    source === "library"
      ? libraryResults.length === 0
      : marketplaceResults.length === 0;

  return (
    <Dialog
      title={title}
      variant="compact"
      styling={{ width: "480px", maxHeight: "70vh" }}
      controlled={{
        isOpen: open,
        set: (nextOpen) => {
          if (!nextOpen) onClose();
        },
      }}
    >
      <Dialog.Content>
        {mode === "pick-expert" ? (
          hiredExperts.length === 0 ? (
            <Text variant="body" className="text-zinc-500">
              No hired experts yet.
            </Text>
          ) : (
            <div className="divide-y divide-zinc-100 overflow-hidden rounded-lg border border-zinc-200/80">
              {hiredExperts.map((expert) => (
                <div
                  key={expert.id}
                  className="flex items-center gap-3 px-3.5 py-2.5 transition-colors hover:bg-zinc-50"
                >
                  <Avatar className="h-9 w-9">
                    {expert.avatar_url ? (
                      <AvatarImage src={expert.avatar_url} alt={expert.name} />
                    ) : null}
                    <AvatarFallback>{expert.name}</AvatarFallback>
                  </Avatar>
                  <div className="min-w-0 flex-1">
                    <Text variant="body-medium" className="truncate">
                      {expert.name}
                    </Text>
                    <Text variant="small" className="!text-zinc-500">
                      {expert.role}
                    </Text>
                  </div>
                  <Button
                    variant="secondary"
                    size="xs"
                    loading={pendingKey === expert.id}
                    onClick={() => installOnExpert(expert)}
                  >
                    Install
                  </Button>
                </div>
              ))}
            </div>
          )
        ) : (
          <div className="flex flex-col gap-3">
            <div
              className="flex gap-2"
              role="group"
              aria-label="Workflow source"
            >
              {INSTALL_WORKFLOW_SOURCES.map((option) => (
                <Button
                  key={option.id}
                  type="button"
                  variant="toggle"
                  size="xs"
                  aria-pressed={source === option.id}
                  onClick={() => setSource(option.id)}
                >
                  {option.label}
                </Button>
              ))}
            </div>
            <Input
              id="install-workflow-search"
              size="small"
              label="Search workflows"
              hideLabel
              placeholder={
                source === "library"
                  ? "Search your workflows"
                  : "Search the marketplace"
              }
              value={searchQuery}
              onChange={(event) => setSearchQuery(event.target.value)}
              wrapperClassName="!mb-0"
            />
            {isSearching ? (
              <Text variant="small" className="py-2 text-center !text-zinc-500">
                Searching…
              </Text>
            ) : isEmpty ? (
              <Text variant="small" className="py-2 text-center !text-zinc-500">
                {source === "library"
                  ? "No workflows in your library."
                  : "No workflows found."}
              </Text>
            ) : (
              <div className="divide-y divide-zinc-100 overflow-hidden rounded-lg border border-zinc-200/80">
                {source === "library"
                  ? libraryResults.map((agent) => (
                      <div
                        key={agent.id}
                        data-testid="install-workflow-option"
                        className="flex items-center gap-3 px-3.5 py-2.5 transition-colors hover:bg-zinc-50"
                      >
                        <WorkflowTile />
                        <div className="min-w-0 flex-1">
                          <Text variant="body-medium" className="truncate">
                            {agent.name}
                          </Text>
                          <Text
                            variant="small"
                            className="truncate !text-zinc-500"
                          >
                            {workflowSubtitle(agent.description)}
                          </Text>
                        </div>
                        <Button
                          variant="secondary"
                          size="xs"
                          loading={pendingKey === agent.id}
                          onClick={() => installLibraryAgent(agent)}
                        >
                          Install
                        </Button>
                      </div>
                    ))
                  : marketplaceResults.map((agent) => (
                      <div
                        key={agent.agent_graph_id}
                        data-testid="install-workflow-option"
                        className="flex items-center gap-3 px-3.5 py-2.5 transition-colors hover:bg-zinc-50"
                      >
                        <WorkflowTile />
                        <div className="min-w-0 flex-1">
                          <Text variant="body-medium" className="truncate">
                            {agent.agent_name}
                          </Text>
                          <Text variant="small" className="!text-zinc-500">
                            by {agent.creator}
                          </Text>
                        </div>
                        <Button
                          variant="secondary"
                          size="xs"
                          loading={pendingKey === agent.agent_graph_id}
                          onClick={() => installFromListing(agent)}
                        >
                          Install
                        </Button>
                      </div>
                    ))}
              </div>
            )}
            {source === "library" && hasMoreLibraryResults ? (
              <Button
                variant="secondary"
                size="xs"
                loading={isLoadingMore}
                onClick={() => loadMoreLibraryResults()}
              >
                Load more workflows
              </Button>
            ) : null}
          </div>
        )}
      </Dialog.Content>
    </Dialog>
  );
}

function WorkflowTile() {
  return (
    <span
      aria-hidden="true"
      className="flex size-9 shrink-0 items-center justify-center rounded-lg bg-zinc-100 text-zinc-600"
    >
      <Icon icon={WorkflowSquare01Icon} size={18} />
    </span>
  );
}
