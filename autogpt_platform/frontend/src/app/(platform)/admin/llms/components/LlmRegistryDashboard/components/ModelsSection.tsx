"use client";

import type { LlmCreatorAdminResponse } from "@/app/api/__generated__/models/llmCreatorAdminResponse";
import type { LlmModelAdminResponse } from "@/app/api/__generated__/models/llmModelAdminResponse";
import type { LlmProviderAdminResponse } from "@/app/api/__generated__/models/llmProviderAdminResponse";
import { Button } from "@/components/atoms/Button/Button";
import { PencilSimpleIcon, PowerIcon, TrashIcon } from "@phosphor-icons/react";
import { useState } from "react";
import { DeleteModelDialog } from "../../DeleteModelDialog/DeleteModelDialog";
import { ModelFormDialog } from "../../ModelFormDialog/ModelFormDialog";
import {
  EnabledBadge,
  SourceBadge,
  VisibilityBadge,
} from "../../RegistryBadges";
import { SimpleTable } from "../../SimpleTable";
import { ToggleModelDialog } from "../../ToggleModelDialog/ToggleModelDialog";
import { useLlmRegistryMutations } from "../../useLlmRegistryMutations";

interface Props {
  models: LlmModelAdminResponse[];
  providers: LlmProviderAdminResponse[];
  creators: LlmCreatorAdminResponse[];
}

type DialogState =
  | { kind: "closed" }
  | { kind: "create" }
  | { kind: "edit"; model: LlmModelAdminResponse }
  | { kind: "disable"; model: LlmModelAdminResponse }
  | { kind: "delete"; model: LlmModelAdminResponse };

export function ModelsSection({ models, providers, creators }: Props) {
  const [dialog, setDialog] = useState<DialogState>({ kind: "closed" });
  const { toggleModel } = useLlmRegistryMutations();

  function close() {
    setDialog({ kind: "closed" });
  }

  function handleToggle(model: LlmModelAdminResponse) {
    if (model.is_enabled) {
      setDialog({ kind: "disable", model });
      return;
    }
    toggleModel.mutate({ slug: model.slug, data: { is_enabled: true } });
  }

  return (
    <div className="flex flex-col gap-3">
      <div className="flex justify-end">
        <Button size="small" onClick={() => setDialog({ kind: "create" })}>
          Add model
        </Button>
      </div>
      <SimpleTable
        columns={[
          "Slug",
          "Name",
          "Creator",
          "Context",
          "Tier",
          "Status",
          "Visibility",
          "Source",
          "Recommended",
          "Actions",
        ]}
        rows={models.map((m) => [
          <code key="slug" className="text-xs">
            {m.slug}
          </code>,
          m.display_name,
          m.creator?.display_name ?? "—",
          m.context_window.toLocaleString(),
          String(m.price_tier),
          <EnabledBadge key="enabled" enabled={m.is_enabled} />,
          <VisibilityBadge key="vis" visibility={m.visibility} />,
          <SourceBadge key="src" source={m.source} />,
          m.is_recommended ? "★" : "",
          <div key="actions" className="flex gap-1">
            <Button
              size="icon"
              variant="icon"
              aria-label={`Edit ${m.slug}`}
              onClick={() => setDialog({ kind: "edit", model: m })}
            >
              <PencilSimpleIcon size={16} />
            </Button>
            <Button
              size="icon"
              variant="icon"
              aria-label={
                m.is_enabled ? `Disable ${m.slug}` : `Enable ${m.slug}`
              }
              onClick={() => handleToggle(m)}
            >
              <PowerIcon size={16} />
            </Button>
            <Button
              size="icon"
              variant="icon"
              aria-label={`Delete ${m.slug}`}
              onClick={() => setDialog({ kind: "delete", model: m })}
            >
              <TrashIcon size={16} />
            </Button>
          </div>,
        ])}
        emptyLabel="No models in the registry yet — the bundled catalog imports on backend startup"
      />
      {(dialog.kind === "create" || dialog.kind === "edit") && (
        <ModelFormDialog
          open
          editing={dialog.kind === "edit" ? dialog.model : null}
          providers={providers}
          creators={creators}
          models={models}
          onClose={close}
        />
      )}
      {dialog.kind === "disable" && (
        <ToggleModelDialog
          open
          model={dialog.model}
          models={models}
          onClose={close}
        />
      )}
      {dialog.kind === "delete" && (
        <DeleteModelDialog
          open
          model={dialog.model}
          models={models}
          onClose={close}
        />
      )}
    </div>
  );
}
