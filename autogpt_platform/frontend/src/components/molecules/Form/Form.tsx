"use client";

import {
  FormControl as KobraFormControl,
  FormDescription,
  FormField,
  FormItem,
  FormLabel,
  FormMessage,
} from "@/components/ui/form";
import { cn } from "@/lib/utils";
import * as React from "react";
import {
  FormProvider,
  type FieldValues,
  type UseFormReturn,
} from "react-hook-form";

interface FormProps<
  TFieldValues extends FieldValues = FieldValues,
> extends Omit<React.FormHTMLAttributes<HTMLFormElement>, "onSubmit"> {
  form: UseFormReturn<TFieldValues>;
  onSubmit: (values: TFieldValues) => void | Promise<void>;
}

function Form<TFieldValues extends FieldValues = FieldValues>({
  form,
  onSubmit,
  className,
  children,
  ...props
}: FormProps<TFieldValues>) {
  return (
    <FormProvider {...form}>
      <form
        onSubmit={form.handleSubmit(onSubmit)}
        className={cn("space-y-4", className)}
        {...props}
      >
        {children}
      </form>
    </FormProvider>
  );
}

interface FormControlProps extends Omit<
  React.ComponentProps<typeof KobraFormControl>,
  "render" | "children"
> {
  /** The field control; receives the item's id, aria-invalid and aria-describedby. */
  children: React.ReactElement;
}

function FormControl({ children, ...props }: FormControlProps) {
  return <KobraFormControl render={children} {...props} />;
}

export {
  Form,
  FormControl,
  FormDescription,
  FormField,
  FormItem,
  FormLabel,
  FormMessage,
};
