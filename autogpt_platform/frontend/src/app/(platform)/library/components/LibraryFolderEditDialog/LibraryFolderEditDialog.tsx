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
import { useToast } from "@/components/molecules/Toast/use-toast";
import { zodResolver } from "@hookform/resolvers/zod";
import { useEffect } from "react";
import { useForm } from "react-hook-form";
import { z } from "zod";
import dynamic from "next/dynamic";
import {
  usePatchV2UpdateFolder,
  getGetV2ListLibraryFoldersQueryKey,
} from "@/app/api/__generated__/endpoints/folders/folders";
import { useQueryClient } from "@tanstack/react-query";
import type { LibraryFolder } from "@/app/api/__generated__/models/libraryFolder";
import type { getV2ListLibraryFoldersResponseSuccess } from "@/app/api/__generated__/endpoints/folders/folders";
import { ApiError } from "@/lib/autogpt-server-api/helpers";
import { FOLDER_COLORS } from "../folder-constants";

const LazyEmojiPicker = dynamic(
  () =>
    import("../LazyEmojiPicker").then((mod) => ({
      default: mod.LazyEmojiPicker,
    })),
  { ssr: false },
);

const editFolderSchema = z.object({
  folderName: z
    .string()
    .min(1, "Folder name is required")
    .max(100, "Folder name must be 100 characters or less"),
  folderColor: z.string().min(1, "Folder color is required"),
  folderIcon: z.string().min(1, "Folder icon is required"),
});

interface Props {
  folder: LibraryFolder;
  isOpen: boolean;
  setIsOpen: (open: boolean) => void;
}

export function LibraryFolderEditDialog({ folder, isOpen, setIsOpen }: Props) {
  const queryClient = useQueryClient();
  const { toast } = useToast();

  const form = useForm<z.infer<typeof editFolderSchema>>({
    resolver: zodResolver(editFolderSchema),
    defaultValues: {
      folderName: folder.name,
      folderColor: folder.color ?? "",
      folderIcon: folder.icon ?? "",
    },
  });

  useEffect(() => {
    if (isOpen) {
      form.reset({
        folderName: folder.name,
        folderColor: folder.color ?? "",
        folderIcon: folder.icon ?? "",
      });
    }
  }, [isOpen, folder, form]);

  const { mutate: updateFolder, isPending } = usePatchV2UpdateFolder({
    mutation: {
      onMutate: async ({ folderId, data }) => {
        await queryClient.cancelQueries({
          queryKey: getGetV2ListLibraryFoldersQueryKey(),
        });

        const previousData =
          queryClient.getQueriesData<getV2ListLibraryFoldersResponseSuccess>({
            queryKey: getGetV2ListLibraryFoldersQueryKey(),
          });

        queryClient.setQueriesData<getV2ListLibraryFoldersResponseSuccess>(
          { queryKey: getGetV2ListLibraryFoldersQueryKey() },
          (old) => {
            if (!old?.data?.folders) return old;
            return {
              ...old,
              data: {
                ...old.data,
                folders: old.data.folders.map((f) =>
                  f.id === folderId
                    ? {
                        ...f,
                        name: data.name ?? f.name,
                        color: data.color ?? f.color,
                        icon: data.icon ?? f.icon,
                      }
                    : f,
                ),
              },
            };
          },
        );

        return { previousData };
      },
      onError: (error: unknown, _variables, context) => {
        if (context?.previousData) {
          for (const [queryKey, data] of context.previousData) {
            queryClient.setQueryData(queryKey, data);
          }
        }
        if (error instanceof ApiError) {
          const detail = (error.response as any)?.detail ?? "";
          if (
            typeof detail === "string" &&
            detail.toLowerCase().includes("already exists")
          ) {
            form.setError("folderName", {
              message: "A folder with this name already exists",
            });
            return;
          }
        }
        toast({
          title: "Error",
          description: "Failed to update folder. Please try again.",
          variant: "destructive",
        });
      },
      onSuccess: () => {
        setIsOpen(false);
        toast({
          title: "Folder updated",
          description: "Your folder has been updated successfully.",
        });
      },
      onSettled: () => {
        queryClient.invalidateQueries({
          queryKey: getGetV2ListLibraryFoldersQueryKey(),
        });
      },
    },
  });

  function onSubmit(values: z.infer<typeof editFolderSchema>) {
    updateFolder({
      folderId: folder.id,
      data: {
        name: values.folderName.trim(),
        color: values.folderColor,
        icon: values.folderIcon,
      },
    });
  }

  return (
    <Dialog
      title="Edit Folder"
      styling={{ maxWidth: "30rem" }}
      controlled={{
        isOpen,
        set: setIsOpen,
      }}
    >
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
                    className="w-full"
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
                    wrapperClassName="!mb-0"
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
            Save Changes
          </Button>
        </Form>
      </Dialog.Content>
    </Dialog>
  );
}
