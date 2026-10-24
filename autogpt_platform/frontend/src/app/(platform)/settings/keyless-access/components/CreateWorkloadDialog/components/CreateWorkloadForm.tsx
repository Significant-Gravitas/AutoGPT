"use client";

import type { UseFormReturn } from "react-hook-form";

import { Button } from "@/components/atoms/Button/Button";
import { Input } from "@/components/atoms/Input/Input";
import {
  Form,
  FormControl,
  FormField,
  FormItem,
  FormMessage,
} from "@/components/molecules/Form/Form";

import { PermissionsCheckboxGroup } from "@/app/(platform)/settings/api-keys/components/CreateAPIKeyDialog/components/PermissionsCheckboxGroup";

import type { Platform, WorkloadFormValues } from "../../../helpers";
import { PlatformFields } from "./PlatformFields";

interface Props {
  form: UseFormReturn<WorkloadFormValues>;
  platform: Platform;
  canSubmit: boolean;
  isPending: boolean;
  onSubmit: (values: WorkloadFormValues) => Promise<void> | void;
  onBack: () => void;
}

export function CreateWorkloadForm({
  form,
  platform,
  canSubmit,
  isPending,
  onSubmit,
  onBack,
}: Props) {
  return (
    <Form form={form} onSubmit={onSubmit} className="flex flex-col gap-4 px-1">
      <PlatformFields form={form} platform={platform} />

      <FormField
        control={form.control}
        name="name"
        render={({ field }) => (
          <FormItem>
            <FormControl>
              <Input
                {...field}
                id={field.name}
                label="Name (optional)"
                placeholder="Shown in your list of trusted workloads"
                wrapperClassName="!mb-0"
              />
            </FormControl>
            <FormMessage />
          </FormItem>
        )}
      />

      <FormField
        control={form.control}
        name="permissions"
        render={({ field }) => (
          <FormItem>
            <FormControl>
              <PermissionsCheckboxGroup
                value={field.value}
                onChange={field.onChange}
              />
            </FormControl>
            <FormMessage />
          </FormItem>
        )}
      />

      <div className="flex flex-col-reverse gap-2 sm:flex-row sm:justify-end">
        <Button variant="secondary" size="large" onClick={onBack}>
          Back
        </Button>
        <Button
          type="submit"
          variant="primary"
          size="large"
          disabled={!canSubmit || isPending}
          loading={isPending}
        >
          Trust this workload
        </Button>
      </div>
    </Form>
  );
}
