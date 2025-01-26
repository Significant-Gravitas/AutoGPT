import React from "react";
import type { Meta, StoryObj } from "@storybook/react";
import {
  FieldValues,
  InternalFieldName,
  RegisterOptions,
  useForm,
  UseFormHandleSubmit,
  UseFormRegister,
  UseFormWatch,
} from "react-hook-form";
import { z } from "zod";
import { zodResolver } from "@hookform/resolvers/zod";

import {
  Form,
  FormField,
  FormItem,
  FormLabel,
  FormControl,
  FormDescription,
  FormMessage,
} from "./form";
import { Input } from "./input";
import { Button } from "./button";

const formSchema = z.object({
  username: z.string().min(2, {
    message: "Username must be at least 2 characters.",
  }),
});

const meta = {
  title: "UI/Form",
  component: Form,
  parameters: {
    layout: "centered",
  },
  tags: ["autodocs"],
} satisfies Meta<typeof Form>;

export default meta;
type Story = StoryObj<typeof meta>;

const FormExample = () => {
  const form = useForm<z.infer<typeof formSchema>>({
    resolver: zodResolver(formSchema),
    defaultValues: {
      username: "",
    },
  });

  function onSubmit(values: z.infer<typeof formSchema>) {
    console.log(values);
  }

  return (
    <Form {...form}>
      <form onSubmit={form.handleSubmit(onSubmit)} className="space-y-8">
        <FormField
          control={form.control}
          name="username"
          render={({ field }) => (
            <FormItem>
              <FormLabel>Username</FormLabel>
              <FormControl>
                <Input placeholder="Enter your username" {...field} />
              </FormControl>
              <FormDescription>
                This is your public display name.
              </FormDescription>
              <FormMessage />
            </FormItem>
          )}
        />
        <Button type="submit">Submit</Button>
      </form>
    </Form>
  );
};

export const Default: Story = {
  args: {
    children: <FormExample />,
    // watch(callback: (data, { name, type }) => void, defaultValues?: {[key:string]: unknown}): { unsubscribe: () => void }
    watch: (name?: any, defaultValue?: any) => {
      if (typeof name === "function") {
        return { unsubscribe: () => {} };
      }
      return defaultValue || {};
    },
    getValues: () => [],
    getFieldState: (name, formState) => ({
      invalid: false,
      isDirty: false,
      isTouched: false,
      isValidating: false,
      error: undefined,
    }),
    setError: () => {},
    setValue: () => {},
    trigger: async () => true,
    reset: () => {},
    clearErrors: () => {},
    formState: {
      errors: {},
      isDirty: false,
      isSubmitting: false,
      isValid: true,
      isLoading: false,
      isSubmitted: false,
      isSubmitSuccessful: false,
      isValidating: false,
      defaultValues: {},
      dirtyFields: {},
      touchedFields: {},
      disabled: false,
      submitCount: 0,
      validatingFields: {},
    },
    resetField: () => {},
    handleSubmit: (() => {
      return async (e?: React.BaseSyntheticEvent) => {
        e?.preventDefault();
        return Promise.resolve();
      };
    }) as unknown as UseFormHandleSubmit<any>,
    unregister: () => {},
    control: {
      _subjects: {
        state: {
          observers: [],
          subscribe: () => ({ unsubscribe: () => {} }),
          unsubscribe: () => {},
          next: () => {},
        },
        array: {
          observers: [],
          subscribe: () => ({ unsubscribe: () => {} }),
          unsubscribe: () => {},
          next: () => {},
        },
        values: {
          observers: [],
          subscribe: () => ({ unsubscribe: () => {} }),
          unsubscribe: () => {},
          next: () => {},
        },
      },
      _reset: () => {},
      _resetDefaultValues: () => {},
      _getFieldArray: () => [],
      _setErrors: () => {},
      _updateDisabledField: () => {},
      _executeSchema: () => Promise.resolve({ errors: {} }),
      handleSubmit: (onSubmit?: any) => (e?: React.BaseSyntheticEvent) => {
        e?.preventDefault();
        return Promise.resolve();
      },
      unregister: () => {},
      getFieldState: () => ({
        invalid: false,
        isDirty: false,
        isTouched: false,
        isValidating: false,
        error: undefined,
      }),
      setError: () => {},
      _disableForm: () => {},
      _removeUnmounted: () => {},
      _names: {
        mount: new Set(),
        array: new Set(),
        watch: new Set(),
        unMount: new Set(),
        disabled: new Set(),
      },
      _state: { mount: false, watch: false, action: false },
      _options: { mode: "onSubmit", defaultValues: {} },
      _formState: {
        isDirty: false,
        isSubmitted: false,
        submitCount: 0,
        isLoading: false,
        isSubmitSuccessful: false,
        isSubmitting: false,
        isValidating: false,
        isValid: true,
        disabled: false,
        dirtyFields: {},
        touchedFields: {},
        errors: {},
        validatingFields: {},
      },
      _fields: {},

      _defaultValues: {},
      _formValues: {},
      _proxyFormState: {
        isDirty: false,
        dirtyFields: false,
        touchedFields: false,
        errors: false,
        isValid: true,
        isValidating: false,
        validatingFields: false,
      },
      _getDirty: () => false,
      _updateValid: () => {},
      _updateFieldArray: () => {},
      _getWatch: () => ({}),
      _updateFormState: () => {},
      register: ((name: string, options?: RegisterOptions<any>) => ({
        name,
        onChange: (e: any) => Promise.resolve(),
        onBlur: (e: any) => Promise.resolve(),
        ref: () => {},
      })) as unknown as UseFormRegister<any>,
    },
    register: ((name: string) => ({
      name,
      onChange: (e: any) => Promise.resolve(),
      onBlur: (e: any) => Promise.resolve(),
      ref: () => {},
    })) as UseFormRegister<FieldValues>,
    setFocus: () => {},
  },
  render: () => <FormExample />,
};

export const WithError: Story = {
  args: {
    children: <FormExample />,
    // watch(callback: (data, { name, type }) => void, defaultValues?: {[key:string]: unknown}): { unsubscribe: () => void }
    watch: (name?: any, defaultValue?: any) => {
      if (typeof name === "function") {
        return { unsubscribe: () => {} };
      }
      return defaultValue || {};
    },
    getValues: () => [],
    getFieldState: (name, formState) => ({
      invalid: false,
      isDirty: false,
      isTouched: false,
      isValidating: false,
      error: undefined,
    }),
    setError: () => {},
    setValue: () => {},
    trigger: async () => true,
    reset: () => {},
    clearErrors: () => {},
    formState: {
      errors: {},
      isDirty: false,
      isSubmitting: false,
      isValid: true,
      isLoading: false,
      isSubmitted: false,
      isSubmitSuccessful: false,
      isValidating: false,
      defaultValues: {},
      dirtyFields: {},
      touchedFields: {},
      disabled: false,
      submitCount: 0,
      validatingFields: {},
    },
    resetField: () => {},
    handleSubmit: (() => {
      return async (e?: React.BaseSyntheticEvent) => {
        e?.preventDefault();
        return Promise.resolve();
      };
    }) as unknown as UseFormHandleSubmit<any>,
    unregister: () => {},
    control: {
      _subjects: {
        state: {
          observers: [],
          subscribe: () => ({ unsubscribe: () => {} }),
          unsubscribe: () => {},
          next: () => {},
        },
        array: {
          observers: [],
          subscribe: () => ({ unsubscribe: () => {} }),
          unsubscribe: () => {},
          next: () => {},
        },
        values: {
          observers: [],
          subscribe: () => ({ unsubscribe: () => {} }),
          unsubscribe: () => {},
          next: () => {},
        },
      },
      _reset: () => {},
      _resetDefaultValues: () => {},
      _getFieldArray: () => [],
      _setErrors: () => {},
      _updateDisabledField: () => {},
      _executeSchema: () => Promise.resolve({ errors: {} }),
      handleSubmit: (onSubmit?: any) => (e?: React.BaseSyntheticEvent) => {
        e?.preventDefault();
        return Promise.resolve();
      },
      unregister: () => {},
      getFieldState: () => ({
        invalid: false,
        isDirty: false,
        isTouched: false,
        isValidating: false,
        error: undefined,
      }),
      setError: () => {},
      _disableForm: () => {},
      _removeUnmounted: () => {},
      _names: {
        mount: new Set(),
        array: new Set(),
        watch: new Set(),
        unMount: new Set(),
        disabled: new Set(),
      },
      _state: { mount: false, watch: false, action: false },
      _options: { mode: "onSubmit", defaultValues: {} },
      _formState: {
        isDirty: false,
        isSubmitted: false,
        submitCount: 0,
        isLoading: false,
        isSubmitSuccessful: false,
        isSubmitting: false,
        isValidating: false,
        isValid: true,
        disabled: false,
        dirtyFields: {},
        touchedFields: {},
        errors: {},
        validatingFields: {},
      },
      _fields: {},

      _defaultValues: {},
      _formValues: {},
      _proxyFormState: {
        isDirty: false,
        dirtyFields: false,
        touchedFields: false,
        errors: false,
        isValid: true,
        isValidating: false,
        validatingFields: false,
      },
      _getDirty: () => false,
      _updateValid: () => {},
      _updateFieldArray: () => {},
      _getWatch: () => ({}),
      _updateFormState: () => {},
      register: ((name: string, options?: RegisterOptions<any>) => ({
        name,
        onChange: (e: any) => Promise.resolve(),
        onBlur: (e: any) => Promise.resolve(),
        ref: () => {},
      })) as unknown as UseFormRegister<any>,
    },
    register: ((name: string) => ({
      name,
      onChange: (e: any) => Promise.resolve(),
      onBlur: (e: any) => Promise.resolve(),
      ref: () => {},
    })) as UseFormRegister<FieldValues>,
    setFocus: () => {},
  },
  render: () => {
    const FormWithError = () => {
      const form = useForm<z.infer<typeof formSchema>>({
        resolver: zodResolver(formSchema),
        defaultValues: {
          username: "",
        },
      });

      React.useEffect(() => {
        form.setError("username", {
          type: "manual",
          message: "This username is already taken.",
        });
      }, [form]);

      function onSubmit(values: z.infer<typeof formSchema>) {
        console.log(values);
      }

      return (
        <Form {...form}>
          <form onSubmit={form.handleSubmit(onSubmit)} className="space-y-8">
            <FormField
              control={form.control}
              name="username"
              render={({ field }) => (
                <FormItem>
                  <FormLabel>Username</FormLabel>
                  <FormControl>
                    <Input placeholder="Enter your username" {...field} />
                  </FormControl>
                  <FormDescription>
                    This is your public display name.
                  </FormDescription>
                  <FormMessage />
                </FormItem>
              )}
            />
            <Button type="submit">Submit</Button>
          </form>
        </Form>
      );
    };

    return <FormWithError />;
  },
};

export const WithDefaultValue: Story = {
  args: {
    children: <FormExample />,
    // watch(callback: (data, { name, type }) => void, defaultValues?: {[key:string]: unknown}): { unsubscribe: () => void }
    watch: (name?: any, defaultValue?: any) => {
      if (typeof name === "function") {
        return { unsubscribe: () => {} };
      }
      return defaultValue || {};
    },
    getValues: () => [],
    getFieldState: (name, formState) => ({
      invalid: false,
      isDirty: false,
      isTouched: false,
      isValidating: false,
      error: undefined,
    }),
    setError: () => {},
    setValue: () => {},
    trigger: async () => true,
    reset: () => {},
    clearErrors: () => {},
    formState: {
      errors: {},
      isDirty: false,
      isSubmitting: false,
      isValid: true,
      isLoading: false,
      isSubmitted: false,
      isSubmitSuccessful: false,
      isValidating: false,
      defaultValues: {},
      dirtyFields: {},
      touchedFields: {},
      disabled: false,
      submitCount: 0,
      validatingFields: {},
    },
    resetField: () => {},
    handleSubmit: (() => {
      return async (e?: React.BaseSyntheticEvent) => {
        e?.preventDefault();
        return Promise.resolve();
      };
    }) as unknown as UseFormHandleSubmit<any>,
    unregister: () => {},
    control: {
      _subjects: {
        state: {
          observers: [],
          subscribe: () => ({ unsubscribe: () => {} }),
          unsubscribe: () => {},
          next: () => {},
        },
        array: {
          observers: [],
          subscribe: () => ({ unsubscribe: () => {} }),
          unsubscribe: () => {},
          next: () => {},
        },
        values: {
          observers: [],
          subscribe: () => ({ unsubscribe: () => {} }),
          unsubscribe: () => {},
          next: () => {},
        },
      },
      _reset: () => {},
      _resetDefaultValues: () => {},
      _getFieldArray: () => [],
      _setErrors: () => {},
      _updateDisabledField: () => {},
      _executeSchema: () => Promise.resolve({ errors: {} }),
      handleSubmit: (onSubmit?: any) => (e?: React.BaseSyntheticEvent) => {
        e?.preventDefault();
        return Promise.resolve();
      },
      unregister: () => {},
      getFieldState: () => ({
        invalid: false,
        isDirty: false,
        isTouched: false,
        isValidating: false,
        error: undefined,
      }),
      setError: () => {},
      _disableForm: () => {},
      _removeUnmounted: () => {},
      _names: {
        mount: new Set(),
        array: new Set(),
        watch: new Set(),
        unMount: new Set(),
        disabled: new Set(),
      },
      _state: { mount: false, watch: false, action: false },
      _options: { mode: "onSubmit", defaultValues: {} },
      _formState: {
        isDirty: false,
        isSubmitted: false,
        submitCount: 0,
        isLoading: false,
        isSubmitSuccessful: false,
        isSubmitting: false,
        isValidating: false,
        isValid: true,
        disabled: false,
        dirtyFields: {},
        touchedFields: {},
        errors: {},
        validatingFields: {},
      },
      _fields: {},

      _defaultValues: {},
      _formValues: {},
      _proxyFormState: {
        isDirty: false,
        dirtyFields: false,
        touchedFields: false,
        errors: false,
        isValid: true,
        isValidating: false,
        validatingFields: false,
      },
      _getDirty: () => false,
      _updateValid: () => {},
      _updateFieldArray: () => {},
      _getWatch: () => ({}),
      _updateFormState: () => {},
      register: ((name: string, options?: RegisterOptions<any>) => ({
        name,
        onChange: (e: any) => Promise.resolve(),
        onBlur: (e: any) => Promise.resolve(),
        ref: () => {},
      })) as unknown as UseFormRegister<any>,
    },
    register: ((name: string) => ({
      name,
      onChange: (e: any) => Promise.resolve(),
      onBlur: (e: any) => Promise.resolve(),
      ref: () => {},
    })) as UseFormRegister<FieldValues>,
    setFocus: () => {},
  },
  render: () => {
    const FormWithDefaultValue = () => {
      const form = useForm<z.infer<typeof formSchema>>({
        resolver: zodResolver(formSchema),
        defaultValues: {
          username: "johndoe",
        },
      });

      function onSubmit(values: z.infer<typeof formSchema>) {
        console.log(values);
      }

      return (
        <Form {...form}>
          <form onSubmit={form.handleSubmit(onSubmit)} className="space-y-8">
            <FormField
              control={form.control}
              name="username"
              render={({ field }) => (
                <FormItem>
                  <FormLabel>Username</FormLabel>
                  <FormControl>
                    <Input placeholder="Enter your username" {...field} />
                  </FormControl>
                  <FormDescription>
                    This is your public display name.
                  </FormDescription>
                  <FormMessage />
                </FormItem>
              )}
            />
            <Button type="submit">Submit</Button>
          </form>
        </Form>
      );
    };

    return <FormWithDefaultValue />;
  },
};
