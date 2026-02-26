"use client";
import { Button } from "@/components/atoms/Button/Button";
import { Input } from "@/components/atoms/Input/Input";
import { Select } from "@/components/atoms/Select/Select";
import { Text } from "@/components/atoms/Text/Text";
import { Dialog } from "@/components/molecules/Dialog/Dialog";
import {
  Form,
  FormControl,
  FormField,
  FormItem,
  FormMessage,
} from "@/components/molecules/Form/Form";
import { zodResolver } from "@hookform/resolvers/zod";
import { FolderSimpleIcon } from "@phosphor-icons/react";
import { useState } from "react";
import { useForm } from "react-hook-form";
import { z } from "zod";
import dynamic from "next/dynamic";
import {
  usePostV2CreateFolder,
  getGetV2ListLibraryFoldersQueryKey,
} from "@/app/api/__generated__/endpoints/folders/folders";
import { useToast } from "@/components/molecules/Toast/use-toast";
import { useQueryClient } from "@tanstack/react-query";
import { FOLDER_COLORS } from "../folder-constants";

const LazyEmojiPicker = dynamic(
  () =>
    import("../LazyEmojiPicker").then((mod) => ({
      default: mod.LazyEmojiPicker,
    })),
  { ssr: false },
);

export const libraryFolderCreationFormSchema = z.object({
  folderName: z
    .string()
    .min(1, "Folder name is required")
    .max(100, "Folder name must be 100 characters or less"),
  folderColor: z.string().min(1, "Folder color is required"),
  folderIcon: z.string().min(1, "Folder icon is required"),
});

export default function LibraryFolderCreationDialog() {
  const [isOpen, setIsOpen] = useState(false);
  const queryClient = useQueryClient();
  const { toast } = useToast();

  const { mutate: createFolder, isPending } = usePostV2CreateFolder({
    mutation: {
      onSuccess: () => {
        queryClient.invalidateQueries({
          queryKey: getGetV2ListLibraryFoldersQueryKey(),
        });
        setIsOpen(false);
        form.reset();
        toast({
          title: "Folder created",
          description: "Your folder has been created successfully.",
        });
      },
      onError: () => {
        toast({
          title: "Error",
          description: "Failed to create folder. Please try again.",
          variant: "destructive",
        });
      },
    },
  });

  const form = useForm<z.infer<typeof libraryFolderCreationFormSchema>>({
    resolver: zodResolver(libraryFolderCreationFormSchema),
    defaultValues: {
      folderName: "",
      folderColor: "",
      folderIcon: "",
    },
  });

  function onSubmit(values: z.infer<typeof libraryFolderCreationFormSchema>) {
    createFolder({
      data: {
        name: values.folderName.trim(),
        color: values.folderColor,
        icon: values.folderIcon,
      },
    });
  }

  return (
    <Dialog
      title="Create Folder"
      styling={{ maxWidth: "30rem" }}
      controlled={{
        isOpen,
        set: setIsOpen,
      }}
      onClose={() => {
        setIsOpen(false);
      }}
    >
      <Dialog.Trigger>
        <Button
          data-testid="upload-agent-button"
          variant="secondary"
          className="h-fit w-fit"
          size="small"
        >
          <FolderSimpleIcon width={18} height={18} />
          <span className="create-folder">Create folder</span>
        </Button>
      </Dialog.Trigger>
      <Dialog.Content>
        <Form
          form={form}
          onSubmit={(values) => onSubmit(values)}
          className="flex flex-col justify-center gap-2 px-1"
        >
          <FormField
            control={form.control}
            name="folderName"
            render={({ field }) => (
              <FormItem>
                <FormControl>
                  <Input
                    {...field}
                    id={field.name}
                    label="Folder name"
                    placeholder="Enter folder name"
                    className="!mb-0 w-full"
                    wrapperClassName="!mb-0"
                  />
                </FormControl>
                <FormMessage />
              </FormItem>
            )}
          />

          <FormField
            control={form.control}
            name="folderColor"
            render={({ field }) => (
              <FormItem>
                <FormControl>
                  <Select
                    id="folderColor"
                    label="Folder color"
                    placeholder="Select a color"
                    value={field.value}
                    onValueChange={field.onChange}
                    options={FOLDER_COLORS.map((color) => ({
                      value: color.value,
                      label: color.label,
                      icon: (
                        <div
                          className="h-4 w-4 rounded-full"
                          style={{ backgroundColor: color.value }}
                        />
                      ),
                    }))}
                    wrapperClassName="!mb-0"
                    renderItem={(option) => (
                      <div className="flex items-center gap-2">
                        {option.icon}
                        <span>{option.label}</span>
                      </div>
                    )}
                  />
                </FormControl>
                <FormMessage />
              </FormItem>
            )}
          />

          <FormField
            control={form.control}
            name="folderIcon"
            render={({ field }) => (
              <FormItem>
                <div className="flex flex-col gap-2">
                  <Text variant="large-medium" as="span" className="text-black">
                    Folder icon
                  </Text>
                  <div className="flex flex-col gap-3">
                    <div className="flex items-center gap-3">
                      <Text variant="small" className="text-zinc-500">
                        Selected:
                      </Text>
                      <div className="flex h-10 w-10 items-center justify-center rounded-lg border border-zinc-200 bg-zinc-50 text-2xl">
                        {form.watch("folderIcon") || (
                          <span className="text-sm text-zinc-400">—</span>
                        )}
                      </div>
                    </div>
                    <div className="h-[295px] w-full overflow-hidden">
                      <LazyEmojiPicker
                        onEmojiSelect={(emoji) => {
                          field.onChange(emoji);
                        }}
                      />
                    </div>
                  </div>
                  <FormMessage />
                </div>
              </FormItem>
            )}
          />

          <Button
            type="submit"
            variant="primary"
            className="mt-2 min-w-[18rem]"
            disabled={!form.formState.isValid || isPending}
            loading={isPending}
          >
            Create
          </Button>
        </Form>
      </Dialog.Content>
    </Dialog>
  );
}
