"use client";
import { Button } from "@/components/atoms/Button/Button";
import { FileInput } from "@/components/atoms/FileInput/FileInput";
import { Input } from "@/components/atoms/Input/Input";
import { LoadingSpinner } from "@/components/atoms/LoadingSpinner/LoadingSpinner";
import { Dialog } from "@/components/molecules/Dialog/Dialog";
import {
  Form,
  FormControl,
  FormField,
  FormItem,
  FormMessage,
} from "@/components/molecules/Form/Form";
import { UploadSimpleIcon } from "@phosphor-icons/react";
import { z } from "zod";
import { useLibraryUploadAgentDialog } from "./useLibraryUploadAgentDialog";

export const uploadAgentFormSchema = z.object({
  agentFile: z.string().min(1, "Agent file is required"),
  agentName: z.string().min(1, "Agent name is required"),
  agentDescription: z.string(),
});

export default function LibraryUploadAgentDialog() {
  const { onSubmit, isUploading, isOpen, setIsOpen, form, agentObject } =
    useLibraryUploadAgentDialog();

  return (
    <Dialog
      title="Upload Agent"
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
          variant="primary"
          className="h-[2.78rem] w-full md:w-[12rem]"
          size="small"
        >
          <UploadSimpleIcon width={18} height={18} />
          <span className="">Upload agent</span>
        </Button>
      </Dialog.Trigger>
      <Dialog.Content>
        <Form
          form={form}
          onSubmit={onSubmit}
          className="flex flex-col justify-center gap-0 px-1"
        >
          <FormField
            control={form.control}
            name="agentName"
            render={({ field }) => (
              <FormItem>
                <FormControl>
                  <Input
                    {...field}
                    id={field.name}
                    label="Agent name"
                    className="w-full rounded-[10px]"
                  />
                </FormControl>
                <FormMessage />
              </FormItem>
            )}
          />

          <FormField
            control={form.control}
            name="agentDescription"
            render={({ field }) => (
              <FormItem>
                <FormControl>
                  <Input
                    {...field}
                    id={field.name}
                    label="Agent description"
                    type="textarea"
                    className="w-full rounded-[10px]"
                  />
                </FormControl>
                <FormMessage />
              </FormItem>
            )}
          />

          <FormField
            control={form.control}
            name="agentFile"
            render={({ field }) => (
              <FormItem>
                <FormControl>
                  <FileInput
                    mode="base64"
                    value={field.value}
                    onChange={field.onChange}
                    accept=".json,application/json"
                    placeholder="Agent file"
                    maxFileSize={10 * 1024 * 1024}
                    showStorageNote={false}
                    className="mb-8 mt-4"
                  />
                </FormControl>
                <FormMessage />
              </FormItem>
            )}
          />

          <Button
            type="submit"
            variant="primary"
            className="min-w-[18rem]"
            disabled={!agentObject || isUploading}
          >
            {isUploading ? (
              <div className="flex items-center gap-2">
                <LoadingSpinner size="small" className="text-white" />
                <span>Uploading...</span>
              </div>
            ) : (
              "Upload"
            )}
          </Button>
        </Form>
      </Dialog.Content>
    </Dialog>
  );
}
