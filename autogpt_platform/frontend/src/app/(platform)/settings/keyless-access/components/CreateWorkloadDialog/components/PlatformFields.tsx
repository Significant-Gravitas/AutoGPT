"use client";

import type { UseFormReturn } from "react-hook-form";

import { Input } from "@/components/atoms/Input/Input";
import { Select } from "@/components/atoms/Select/Select";
import {
  FormControl,
  FormField,
  FormItem,
  FormMessage,
} from "@/components/molecules/Form/Form";

import type { Platform, WorkloadFormValues } from "../../../helpers";
import {
  type FieldConfig,
  PLATFORM_FIELDS,
  VERCEL_ENVIRONMENTS,
  VERCEL_ISSUER_MODES,
} from "../helpers";

interface Props {
  form: UseFormReturn<WorkloadFormValues>;
  platform: Platform;
}

export function PlatformFields({ form, platform }: Props) {
  return (
    <>
      {PLATFORM_FIELDS[platform].map((config) => (
        <TextField key={config.name} form={form} config={config} />
      ))}
      {platform === "vercel" && (
        <>
          <SelectField
            form={form}
            name="vercelEnvironment"
            label="Environment"
            options={VERCEL_ENVIRONMENTS}
          />
          <SelectField
            form={form}
            name="vercelIssuerMode"
            label="Issuer mode"
            hint="Project Settings, Security, Secure backend access."
            options={VERCEL_ISSUER_MODES}
          />
        </>
      )}
    </>
  );
}

interface TextFieldProps {
  form: UseFormReturn<WorkloadFormValues>;
  config: FieldConfig;
}

function TextField({ form, config }: TextFieldProps) {
  return (
    <FormField
      control={form.control}
      name={config.name}
      render={({ field }) => (
        <FormItem>
          <FormControl>
            <Input
              id={field.name}
              name={field.name}
              label={config.label}
              placeholder={config.placeholder}
              hint={config.hint}
              value={String(field.value ?? "")}
              onChange={field.onChange}
              onBlur={field.onBlur}
              wrapperClassName="!mb-0"
            />
          </FormControl>
          <FormMessage />
        </FormItem>
      )}
    />
  );
}

interface SelectFieldProps {
  form: UseFormReturn<WorkloadFormValues>;
  name: "vercelEnvironment" | "vercelIssuerMode";
  label: string;
  hint?: string;
  options: { value: string; label: string }[];
}

function SelectField({ form, name, label, hint, options }: SelectFieldProps) {
  return (
    <FormField
      control={form.control}
      name={name}
      render={({ field }) => (
        <FormItem>
          <FormControl>
            <Select
              id={name}
              label={label}
              hint={hint}
              value={field.value}
              onValueChange={field.onChange}
              options={options}
              wrapperClassName="!mb-0"
            />
          </FormControl>
          <FormMessage />
        </FormItem>
      )}
    />
  );
}
