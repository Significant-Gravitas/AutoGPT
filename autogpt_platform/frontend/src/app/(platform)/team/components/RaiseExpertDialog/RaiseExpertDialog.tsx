"use client";

import { Button } from "@/components/atoms/Button/Button";
import { Input } from "@/components/atoms/Input/Input";
import { Text } from "@/components/atoms/Text/Text";
import { Dialog } from "@/components/molecules/Dialog/Dialog";
import { isTouchDevice } from "@/lib/platform";
import { AvatarSlot } from "./components/AvatarSlot";
import { MAX_JOB_TITLE_LENGTH, MAX_NAME_LENGTH } from "./helpers";
import { useRaiseExpertDialog } from "./useRaiseExpertDialog";

interface Props {
  open: boolean;
  onClose: () => void;
  initialJobTitle?: string;
}

export function RaiseExpertDialog({
  open,
  onClose,
  initialJobTitle = "",
}: Props) {
  return (
    <Dialog
      variant="compact"
      title="Create an expert"
      styling={{ maxWidth: "26rem" }}
      controlled={{
        isOpen: open,
        set: (next) => {
          if (!next) onClose();
        },
      }}
    >
      <Dialog.Content>
        {open ? (
          <RaiseExpertForm
            initialJobTitle={initialJobTitle}
            onClose={onClose}
          />
        ) : null}
      </Dialog.Content>
    </Dialog>
  );
}

interface FormProps {
  initialJobTitle: string;
  onClose: () => void;
}

function RaiseExpertForm({ initialJobTitle, onClose }: FormProps) {
  const form = useRaiseExpertDialog({ initialJobTitle, onClose });

  return (
    <form
      onSubmit={form.submit}
      aria-label="Create an expert"
      className="flex flex-col gap-4 px-1"
    >
      <div className="flex items-end gap-3">
        <AvatarSlot
          name={form.name}
          imageUrl={form.imageUrl}
          onPick={form.pickImage}
          onRemove={form.removeImage}
        />
        <Input
          id="raise-expert-name"
          size="small"
          label="Name"
          labelVariant="small-medium"
          placeholder="e.g. Deps merger"
          autoComplete="off"
          data-1p-ignore
          autoFocus={!isTouchDevice()}
          maxLength={MAX_NAME_LENGTH}
          value={form.name}
          onChange={(event) => form.setName(event.target.value)}
          wrapperClassName="!mb-0 min-w-0 flex-1"
        />
      </div>
      <Input
        id="raise-expert-role"
        size="small"
        label="Role"
        labelVariant="small-medium"
        placeholder="e.g. Keeps our dependency updates merged"
        autoComplete="off"
        data-1p-ignore
        maxLength={MAX_JOB_TITLE_LENGTH}
        value={form.jobTitle}
        onChange={(event) => form.setJobTitle(event.target.value)}
        wrapperClassName="!mb-0"
      />
      {form.imageError ? (
        <Text variant="small" tone="error" role="alert">
          {form.imageError}
        </Text>
      ) : (
        <Text variant="small" tone="muted">
          Your expert works out the rest from these and asks you two or three
          quick questions. No photo? We&apos;ll make one.
        </Text>
      )}
      <div className="flex justify-end gap-2">
        <Button
          size="small"
          type="button"
          variant="secondary"
          onClick={onClose}
        >
          Cancel
        </Button>
        <Button
          size="small"
          type="submit"
          variant="primary"
          disabled={!form.canSubmit}
        >
          Create
        </Button>
      </div>
    </form>
  );
}
