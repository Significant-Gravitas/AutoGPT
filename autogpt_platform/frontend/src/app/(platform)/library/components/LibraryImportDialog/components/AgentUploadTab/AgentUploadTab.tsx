"use client";
import { Button } from "@/components/atoms/Button/Button";
import { FileInput } from "@/components/atoms/FileInput/FileInput";
import { Input } from "@/components/atoms/Input/Input";
import { LoadingSpinner } from "@/components/atoms/LoadingSpinner/LoadingSpinner";
import {
  Form,
  FormControl,
  FormField,
  FormItem,
  FormMessage,
} from "@/components/molecules/Form/Form";
import { TabsLineContent } from "@/components/molecules/TabsLine/TabsLine";
import { useLibraryUploadAgentDialog } from "../../../LibraryUploadAgentDialog/useLibraryUploadAgentDialog";

type AgentUploadTabProps = {
  upload: ReturnType<typeof useLibraryUploadAgentDialog>;
};

export default function AgentUploadTab({ upload }: AgentUploadTabProps) {
  return (
    <TabsLineContent value="agent">
      <p className="mb-4 text-sm text-neutral-500">
        Upload a previously exported AutoGPT agent file (.json).
      </p>
      <Form
        form={upload.form}
        onSubmit={upload.onSubmit}
        className="flex flex-col justify-center gap-0 px-1"
      >
        <FormField
          control={upload.form.control}
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
          control={upload.form.control}
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
          control={upload.form.control}
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
          className="w-full"
          disabled={!upload.agentObject || upload.isUploading}
        >
          {upload.isUploading ? (
            <div className="flex items-center gap-2">
              <LoadingSpinner size="small" className="text-white" />
              <span>Uploading...</span>
            </div>
          ) : (
            "Upload"
          )}
        </Button>
      </Form>
    </TabsLineContent>
  );
}
