"use client";

import { Button } from "@/components/atoms/Button/Button";
import { Input } from "@/components/atoms/Input/Input";
import { Select } from "@/components/atoms/Select/Select";
import { Dialog } from "@/components/molecules/Dialog/Dialog";
import {
  Form,
  FormControl,
  FormField,
  FormItem,
  FormMessage,
} from "@/components/molecules/Form/Form";

import { JOIN_POLICY_OPTIONS } from "./schema";
import { useCreateTeamDialog } from "./useCreateTeamDialog";

interface Props {
  orgId: string;
  open: boolean;
  onOpenChange: (open: boolean) => void;
  onCreated: () => void;
}

export function CreateTeamDialog({
  orgId,
  open,
  onOpenChange,
  onCreated,
}: Props) {
  const { form, isPending, handleSubmit, handleClose } = useCreateTeamDialog({
    orgId,
    onCreated,
    onClose: () => onOpenChange(false),
  });

  return (
    <Dialog
      title="Create team"
      styling={{ maxWidth: "30rem" }}
      controlled={{
        isOpen: open,
        set: (next) => {
          if (next) {
            onOpenChange(true);
            return;
          }
          if (isPending) return;
          handleClose();
        },
      }}
    >
      <Dialog.Content>
        <Form
          form={form}
          onSubmit={handleSubmit}
          className="flex flex-col gap-4 px-1"
        >
          <FormField
            control={form.control}
            name="name"
            render={({ field }) => (
              <FormItem>
                <FormControl>
                  <Input
                    {...field}
                    id={field.name}
                    label="Name"
                    placeholder="Engineering"
                    wrapperClassName="!mb-0"
                  />
                </FormControl>
                <FormMessage />
              </FormItem>
            )}
          />

          <FormField
            control={form.control}
            name="description"
            render={({ field }) => (
              <FormItem>
                <FormControl>
                  <Input
                    {...field}
                    id={field.name}
                    label="Description (optional)"
                    placeholder="What is this team for?"
                    wrapperClassName="!mb-0"
                  />
                </FormControl>
                <FormMessage />
              </FormItem>
            )}
          />

          <FormField
            control={form.control}
            name="join_policy"
            render={({ field }) => (
              <FormItem>
                <FormControl>
                  <Select
                    id={field.name}
                    label="Join policy"
                    value={field.value}
                    onValueChange={field.onChange}
                    options={JOIN_POLICY_OPTIONS}
                    wrapperClassName="!mb-0"
                  />
                </FormControl>
                <FormMessage />
              </FormItem>
            )}
          />

          <div className="flex justify-end gap-2 pt-2">
            <Button
              type="button"
              variant="secondary"
              onClick={handleClose}
              disabled={isPending}
            >
              Cancel
            </Button>
            <Button type="submit" loading={isPending}>
              Create team
            </Button>
          </div>
        </Form>
      </Dialog.Content>
    </Dialog>
  );
}
