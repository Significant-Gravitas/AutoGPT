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

import type { CreateAPIKeyFormValues } from "../schema";
import { PermissionsCheckboxGroup } from "./PermissionsCheckboxGroup";

interface Props {
  form: UseFormReturn<CreateAPIKeyFormValues>;
  onSubmit: (values: CreateAPIKeyFormValues) => Promise<void> | void;
  isPending: boolean;
}

export function CreateAPIKeyForm({ form, onSubmit, isPending }: Props) {
  return (
    <Form form={form} onSubmit={onSubmit} className="flex flex-col gap-4 px-1">
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
                placeholder="My integration key"
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
                placeholder="Describe what this key is used for"
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

      <Button
        type="submit"
        variant="primary"
        size="large"
        disabled={!form.formState.isValid || isPending}
        loading={isPending}
      >
        Create Key
      </Button>
    </Form>
  );
}
