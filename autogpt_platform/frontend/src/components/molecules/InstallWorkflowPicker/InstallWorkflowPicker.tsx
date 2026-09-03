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
import { cn } from "@/lib/utils";
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
    isLibraryAgentInstalled,
    source,
    setSource,
    searchQuery,
    setSearchQuery,
    libraryResults,
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
      styling={{ width: "480px" }}
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
            <div className="divide-y divide-zinc-100 overflow-hidden rounded-xl border border-zinc-200/80">
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
                    size="small"
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
                <button
                  key={option.id}
                  type="button"
                  aria-pressed={source === option.id}
                  onClick={() => setSource(option.id)}
                  className={cn(
                    "rounded-full px-3 py-1.5 text-sm transition-colors focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-zinc-900",
                    source === option.id
                      ? "bg-zinc-800 text-white"
                      : "bg-zinc-100 text-zinc-600 hover:bg-zinc-200",
                  )}
                >
                  {option.label}
                </button>
              ))}
            </div>
            <Input
              id="install-workflow-search"
              label="Search workflows"
              hideLabel
              placeholder={
                source === "library"
                  ? "Search your workflows"
                  : "Search the marketplace"
              }
              value={searchQuery}
              onChange={(event) => setSearchQuery(event.target.value)}
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
              <div className="divide-y divide-zinc-100 overflow-hidden rounded-xl border border-zinc-200/80">
                {source === "library"
                  ? libraryResults.map((agent) => (
                      <div
                        key={agent.id}
                        data-testid="install-workflow-option"
                        className="flex items-center gap-3 px-3.5 py-2.5 transition-colors hover:bg-zinc-50"
                      >
                        <Avatar className="h-9 w-9">
                          {agent.image_url ? (
                            <AvatarImage
                              src={agent.image_url}
                              alt={agent.name}
                            />
                          ) : null}
                          <AvatarFallback>{agent.name}</AvatarFallback>
                        </Avatar>
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
                        {isLibraryAgentInstalled(agent) ? (
                          <Text
                            variant="small"
                            className="shrink-0 !text-zinc-400"
                          >
                            Installed
                          </Text>
                        ) : (
                          <Button
                            variant="secondary"
                            size="small"
                            loading={pendingKey === agent.id}
                            onClick={() => installLibraryAgent(agent)}
                          >
                            Install
                          </Button>
                        )}
                      </div>
                    ))
                  : marketplaceResults.map((agent) => (
                      <div
                        key={agent.agent_graph_id}
                        data-testid="install-workflow-option"
                        className="flex items-center gap-3 px-3.5 py-2.5 transition-colors hover:bg-zinc-50"
                      >
                        <Avatar className="h-9 w-9">
                          {agent.agent_image ? (
                            <AvatarImage
                              src={agent.agent_image}
                              alt={agent.agent_name}
                            />
                          ) : null}
                          <AvatarFallback>{agent.agent_name}</AvatarFallback>
                        </Avatar>
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
                          size="small"
                          loading={pendingKey === agent.agent_graph_id}
                          onClick={() => installFromListing(agent)}
                        >
                          Install
                        </Button>
                      </div>
                    ))}
              </div>
            )}
          </div>
        )}
      </Dialog.Content>
    </Dialog>
  );
}
