"use client";

import { Button } from "@/components/atoms/Button/Button";
import { Input } from "@/components/atoms/Input/Input";
import { Dialog } from "@/components/molecules/Dialog/Dialog";
import {
  Form,
  FormControl,
  FormField,
  FormItem,
  FormMessage,
} from "@/components/molecules/Form/Form";

import { useCreateOrgDialog } from "./useCreateOrgDialog";

interface Props {
  open: boolean;
  onOpenChange: (open: boolean) => void;
}

export function CreateOrgDialog({ open, onOpenChange }: Props) {
  const { form, isPending, handleNameChange, handleSubmit, handleClose } =
    useCreateOrgDialog({ onClose: () => onOpenChange(false) });

  return (
    <Dialog
      title="Create organization"
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
                    placeholder="Acme Inc."
                    wrapperClassName="!mb-0"
                    onChange={(e) => handleNameChange(e.target.value)}
                  />
                </FormControl>
                <FormMessage />
              </FormItem>
            )}
          />

          <FormField
            control={form.control}
            name="slug"
            render={({ field }) => (
              <FormItem>
                <FormControl>
                  <Input
                    {...field}
                    id={field.name}
                    label="URL slug"
                    placeholder="acme-inc"
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
                    placeholder="What is this organization for?"
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
              Create organization
            </Button>
          </div>
        </Form>
      </Dialog.Content>
    </Dialog>
  );
}
