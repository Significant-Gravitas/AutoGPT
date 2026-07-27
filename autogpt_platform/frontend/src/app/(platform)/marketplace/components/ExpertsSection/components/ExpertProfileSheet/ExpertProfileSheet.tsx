"use client";

import {
  Avatar,
  AvatarFallback,
  AvatarImage,
} from "@/components/atoms/Avatar/Avatar";
import { Button } from "@/components/atoms/Button/Button";
import { Text } from "@/components/atoms/Text/Text";
import { Dialog } from "@/components/molecules/Dialog/Dialog";
import { useExpertProfileSheet } from "./useExpertProfileSheet";

interface Props {
  templateId: string | null;
  onClose: () => void;
}

export function ExpertProfileSheet({ templateId, onClose }: Props) {
  const { template, isHired, isHiring, hire } = useExpertProfileSheet(
    templateId,
    onClose,
  );

  return (
    <Dialog
      title={template?.name ?? ""}
      styling={{ width: "480px" }}
      controlled={{
        isOpen: templateId !== null,
        set: (open) => {
          if (!open) onClose();
        },
      }}
    >
      <Dialog.Content>
        {template ? (
          <div className="flex flex-col gap-4">
            <div className="flex items-center gap-3">
              <Avatar className="h-14 w-14">
                {template.avatar_url ? (
                  <AvatarImage src={template.avatar_url} alt={template.name} />
                ) : null}
                <AvatarFallback>{template.name}</AvatarFallback>
              </Avatar>
              <div>
                <Text variant="large-medium">{template.role}</Text>
                {template.tagline ? (
                  <Text variant="small" className="text-zinc-500">
                    {template.tagline}
                  </Text>
                ) : null}
              </div>
            </div>
            <Text variant="body">{template.identity}</Text>
            {template.workflows.length > 0 ? (
              <div className="flex flex-col gap-1">
                <Text variant="body-medium">Preloaded workflows</Text>
                {template.workflows.map((workflow) => (
                  <Text key={workflow.id} variant="small">
                    {workflow.name ?? workflow.id}
                  </Text>
                ))}
              </div>
            ) : null}
            <Dialog.Footer>
              {isHired ? (
                <Button variant="secondary" disabled>
                  Hired
                </Button>
              ) : (
                <Button variant="primary" onClick={hire} loading={isHiring}>
                  Hire
                </Button>
              )}
            </Dialog.Footer>
          </div>
        ) : null}
      </Dialog.Content>
    </Dialog>
  );
}
