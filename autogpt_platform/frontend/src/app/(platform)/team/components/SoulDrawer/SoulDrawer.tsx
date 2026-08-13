"use client";

import { Expert } from "@/app/api/__generated__/models/expert";
import { ExpertSoulUpdate } from "@/app/api/__generated__/models/expertSoulUpdate";
import { LearnedNote } from "@/app/api/__generated__/models/learnedNote";
import {
  Avatar,
  AvatarFallback,
  AvatarImage,
} from "@/components/atoms/Avatar/Avatar";
import { Button } from "@/components/atoms/Button/Button";
import { Input } from "@/components/atoms/Input/Input";
import { Text } from "@/components/atoms/Text/Text";
import {
  Sheet,
  SheetContent,
  SheetDescription,
  SheetTitle,
} from "@/components/ui/sheet";
import { Icon } from "@/components/atoms/Icon/Icon";
import {
  Cancel01Icon,
  Delete02Icon,
  LockIcon,
  PencilEdit02Icon,
  Tick02Icon,
} from "@hugeicons/core-free-icons";
import { ReactNode, useState } from "react";
import { useLearnedNotes } from "./useLearnedNotes";
import { useSoulDrawer } from "./useSoulDrawer";

interface Props {
  expert: Expert | null;
  onClose: () => void;
}

export function SoulDrawer({ expert, onClose }: Props) {
  const [displayedExpert] = useState(expert);
  const { soul, updateField, save, isPending, canSave } = useSoulDrawer({
    expert,
    onClose,
  });

  return (
    <Sheet
      open={expert !== null}
      onOpenChange={(open) => {
        if (!open) onClose();
      }}
    >
      <SheetContent
        side="right"
        className="flex w-full flex-col overflow-hidden p-0 sm:w-1/2 sm:max-w-none"
      >
        <div className="border-b border-zinc-200 bg-white px-6 py-5 pr-12 sm:px-8">
          <div className="flex items-center gap-3">
            <Avatar className="h-11 w-11">
              {displayedExpert?.avatar_url ? (
                <AvatarImage
                  src={displayedExpert.avatar_url}
                  alt={displayedExpert.name}
                />
              ) : null}
              <AvatarFallback>{displayedExpert?.name ?? "Soul"}</AvatarFallback>
            </Avatar>
            <div>
              <SheetTitle>{`${displayedExpert?.name ?? "Expert"}'s Soul`}</SheetTitle>
              <SheetDescription>
                A living document that shapes every reply.
              </SheetDescription>
            </div>
          </div>
        </div>

        <form onSubmit={save} className="flex min-h-0 flex-1 flex-col">
          <div className="flex-1 overflow-y-auto bg-zinc-50 px-4 py-6 sm:px-8">
            <div className="mx-auto max-w-2xl rounded-2xl border border-zinc-200 bg-white p-5 shadow-sm sm:p-8">
              <SoulFields soul={soul} updateField={updateField} />
              <LearnedNotes expert={expert} />
              <ProtectedRules
                rules={displayedExpert?.protected_soul_rules ?? []}
              />
            </div>
          </div>
          <div className="flex justify-end gap-2 border-t border-zinc-200 bg-white px-6 py-4 sm:px-8">
            <Button type="button" variant="ghost" onClick={onClose}>
              Cancel
            </Button>
            <Button
              type="submit"
              variant="primary"
              loading={isPending}
              disabled={!canSave}
            >
              Save Soul
            </Button>
          </div>
        </form>
      </SheetContent>
    </Sheet>
  );
}

interface SoulFieldsProps {
  soul: ExpertSoulUpdate;
  updateField: (field: keyof ExpertSoulUpdate, value: string) => void;
}

function SoulFields({ soul, updateField }: SoulFieldsProps) {
  return (
    <div className="space-y-1">
      <Input
        id="soul-name"
        label="Name"
        labelVariant="small-medium"
        labelClassName="uppercase tracking-[0.12em] !text-purple-600"
        value={soul.name}
        maxLength={100}
        required
        onChange={(event) => updateField("name", event.target.value)}
      />
      <Input
        id="soul-identity"
        label="Identity and personality"
        labelVariant="small-medium"
        labelClassName="uppercase tracking-[0.12em] !text-purple-600"
        type="textarea"
        rows={6}
        value={soul.identity}
        maxLength={10000}
        required
        onChange={(event) => updateField("identity", event.target.value)}
      />
      <Input
        id="soul-voice"
        label="Voice"
        labelVariant="small-medium"
        labelClassName="uppercase tracking-[0.12em] !text-purple-600"
        type="textarea"
        rows={3}
        value={soul.voice_preferences}
        maxLength={4000}
        placeholder="How should this expert sound?"
        onChange={(event) =>
          updateField("voice_preferences", event.target.value)
        }
      />
      <Input
        id="soul-boundaries"
        label="Boundaries"
        labelVariant="small-medium"
        labelClassName="uppercase tracking-[0.12em] !text-purple-600"
        type="textarea"
        rows={4}
        value={soul.boundaries}
        maxLength={4000}
        placeholder="What should this expert avoid or ask about first?"
        onChange={(event) => updateField("boundaries", event.target.value)}
      />
    </div>
  );
}

function LearnedNotes({ expert }: { expert: Expert | null }) {
  const {
    notes,
    editingId,
    editText,
    setEditText,
    startEdit,
    cancelEdit,
    saveEdit,
    removeNote,
    isSaving,
    isDeleting,
  } = useLearnedNotes(expert);

  return (
    <section className="mb-8">
      <SoulSectionTitle>What I&apos;ve learned</SoulSectionTitle>
      {notes.length === 0 ? (
        <Text variant="small" className="text-zinc-500">
          Nothing recorded yet. What this expert learns will appear here.
        </Text>
      ) : (
        <ul className="mt-3 space-y-2">
          {notes.map((note) =>
            editingId === note.id ? (
              <LearnedNoteEditor
                key={note.id}
                note={note}
                editText={editText}
                setEditText={setEditText}
                onSave={saveEdit}
                onCancel={cancelEdit}
                isSaving={isSaving}
              />
            ) : (
              <LearnedNoteRow
                key={note.id}
                note={note}
                onEdit={() => startEdit(note.id, note.fact)}
                onRemove={() => removeNote(note.id)}
                isRemoving={isDeleting}
              />
            ),
          )}
        </ul>
      )}
    </section>
  );
}

interface LearnedNoteRowProps {
  note: LearnedNote;
  onEdit: () => void;
  onRemove: () => void;
  isRemoving: boolean;
}

function LearnedNoteRow({
  note,
  onEdit,
  onRemove,
  isRemoving,
}: LearnedNoteRowProps) {
  return (
    <li className="flex items-start gap-2 rounded-xl bg-zinc-50 p-3">
      <span className="flex-1 text-sm leading-5 text-zinc-700">
        {note.fact}
      </span>
      <Button
        type="button"
        variant="ghost"
        size="small"
        className="min-w-0 px-2"
        aria-label={`Edit note: ${note.fact}`}
        onClick={onEdit}
      >
        <Icon icon={PencilEdit02Icon} size={16} />
      </Button>
      <Button
        type="button"
        variant="ghost"
        size="small"
        className="min-w-0 px-2"
        loading={isRemoving}
        aria-label={`Remove note: ${note.fact}`}
        onClick={onRemove}
      >
        <Icon icon={Delete02Icon} size={16} />
      </Button>
    </li>
  );
}

interface LearnedNoteEditorProps {
  note: LearnedNote;
  editText: string;
  setEditText: (value: string) => void;
  onSave: () => void;
  onCancel: () => void;
  isSaving: boolean;
}

function LearnedNoteEditor({
  note,
  editText,
  setEditText,
  onSave,
  onCancel,
  isSaving,
}: LearnedNoteEditorProps) {
  return (
    <li className="flex items-center gap-2">
      <Input
        id={`note-${note.id}`}
        label="Edit note"
        hideLabel
        value={editText}
        maxLength={2000}
        wrapperClassName="flex-1"
        onChange={(event) => setEditText(event.target.value)}
      />
      <Button
        type="button"
        variant="ghost"
        size="small"
        className="min-w-0 px-2"
        loading={isSaving}
        aria-label="Save note"
        onClick={onSave}
      >
        <Icon icon={Tick02Icon} size={16} />
      </Button>
      <Button
        type="button"
        variant="ghost"
        size="small"
        className="min-w-0 px-2"
        aria-label="Cancel edit"
        onClick={onCancel}
      >
        <Icon icon={Cancel01Icon} size={16} />
      </Button>
    </li>
  );
}

function ProtectedRules({ rules }: { rules: string[] }) {
  return (
    <section className="border-t border-zinc-200 pt-6">
      <div className="mb-3 flex items-center gap-2">
        <Icon icon={LockIcon} size={16} className="text-zinc-500" />
        <SoulSectionTitle>Protected rules</SoulSectionTitle>
      </div>
      <div className="space-y-2 rounded-xl bg-zinc-50 p-4">
        {rules.map((rule) => (
          <div
            key={rule}
            className="flex gap-2 text-sm leading-5 text-zinc-600"
          >
            <Icon icon={LockIcon} size={14} className="mt-0.5 shrink-0" />
            <span>{rule}</span>
          </div>
        ))}
      </div>
      <Text variant="small" className="mt-3 text-zinc-400">
        These safeguards are always active and cannot be edited.
      </Text>
    </section>
  );
}

function SoulSectionTitle({ children }: { children: ReactNode }) {
  return (
    <h3 className="text-xs font-semibold uppercase tracking-[0.12em] text-purple-600">
      {children}
    </h3>
  );
}
