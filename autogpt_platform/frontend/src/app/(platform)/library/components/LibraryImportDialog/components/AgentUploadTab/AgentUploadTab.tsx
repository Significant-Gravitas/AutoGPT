"use client";
import { Text } from "@/components/atoms/Text/Text";
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
      <Text variant="body-tight" tone="muted" className="mb-4">
        Upload a previously exported AutoGPT workflow file (.json).
      </Text>
      <Form
        form={upload.form}
        onSubmit={upload.onSubmit}
        className="mx-1 flex flex-col justify-center"
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
                  label="Workflow name"
                  className="w-full"
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
                  label="Workflow description"
                  type="textarea"
                  className="w-full"
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
                  placeholder="Workflow file"
                  maxFileSize={10 * 1024 * 1024}
                  showStorageNote={false}
                  className="mt-4 mb-8"
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
              <LoadingSpinner size="small" />
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
