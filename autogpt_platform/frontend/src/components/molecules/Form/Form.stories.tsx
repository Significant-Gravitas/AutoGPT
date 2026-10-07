import { Button } from "@/components/atoms/Button/Button";
import { Input } from "@/components/atoms/Input/Input";
import { Switch } from "@/components/atoms/Switch/Switch";
import { zodResolver } from "@hookform/resolvers/zod";
import type { Meta, StoryObj } from "@storybook/nextjs";
import { useForm } from "react-hook-form";
import { expect, fn, userEvent, within } from "storybook/test";
import { z } from "zod";
import {
  Form,
  FormControl,
  FormDescription,
  FormField,
  FormItem,
  FormLabel,
  FormMessage,
} from "./Form";

const profileSchema = z.object({
  name: z.string().trim().min(2, "Enter at least 2 characters"),
  email: z.string().trim().email("Enter a valid email address"),
  notifyOnFailure: z.boolean(),
});

type ProfileValues = z.infer<typeof profileSchema>;

const EMPTY_VALUES: ProfileValues = {
  name: "",
  email: "",
  notifyOnFailure: false,
};

const FILLED_VALUES: ProfileValues = {
  name: "Ada Lovelace",
  email: "ada@example.com",
  notifyOnFailure: true,
};

const meta = {
  title: "Molecules/Form",
  component: Form,
  tags: ["autodocs"],
  decorators: [
    (Story) => (
      <div className="w-96">
        <Story />
      </div>
    ),
  ],
  parameters: {
    layout: "centered",
    a11y: { test: "error" },
    docs: {
      description: {
        component:
          "react-hook-form bindings for the design system. `Form` takes a `useForm` instance and an `onSubmit`, and wires `handleSubmit`. Each field is a `FormField` (a `Controller`) rendering a `FormItem` with `FormControl` (adds `id`, `aria-invalid` and `aria-describedby` to the control), an optional `FormLabel` and `FormDescription`, and a `FormMessage` that shows the field's validation error. Validate with `zodResolver` and a zod schema.",
      },
    },
  },
  args: {
    onSubmit: fn(),
  },
} satisfies Meta<typeof Form>;

export default meta;
type Story = StoryObj<typeof Form>;

export const Default: Story = {
  render: (args) => <ProfileForm onSubmit={args.onSubmit} />,
};

export const Prefilled: Story = {
  render: (args) => (
    <ProfileForm onSubmit={args.onSubmit} defaultValues={FILLED_VALUES} />
  ),
};

export const ValidationErrors: Story = {
  render: (args) => <ProfileForm onSubmit={args.onSubmit} />,
  play: async ({ canvasElement, args }) => {
    const canvas = within(canvasElement);
    await userEvent.click(canvas.getByRole("button", { name: "Save profile" }));
    await expect(
      await canvas.findByText("Enter at least 2 characters"),
    ).toBeInTheDocument();
    await expect(
      await canvas.findByText("Enter a valid email address"),
    ).toBeInTheDocument();
    await expect(args.onSubmit).not.toHaveBeenCalled();
  },
};

export const Submitting: Story = {
  render: () => (
    <ProfileForm
      defaultValues={FILLED_VALUES}
      onSubmit={() => new Promise<void>(() => {})}
    />
  ),
  play: async ({ canvasElement }) => {
    await userEvent.click(
      within(canvasElement).getByRole("button", { name: "Save profile" }),
    );
  },
};

export const Disabled: Story = {
  render: (args) => (
    <ProfileForm
      onSubmit={args.onSubmit}
      defaultValues={FILLED_VALUES}
      disabled
    />
  ),
};

interface Props {
  onSubmit: (values: ProfileValues) => void | Promise<void>;
  defaultValues?: ProfileValues;
  disabled?: boolean;
}

function ProfileForm({
  onSubmit,
  defaultValues = EMPTY_VALUES,
  disabled = false,
}: Props) {
  const form = useForm<ProfileValues>({
    resolver: zodResolver(profileSchema),
    defaultValues,
    disabled,
  });

  return (
    <Form form={form} onSubmit={onSubmit} className="flex flex-col gap-4">
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
                placeholder="Ada Lovelace"
                wrapperClassName="mb-0!"
              />
            </FormControl>
            <FormDescription>Shown on agents you publish.</FormDescription>
            <FormMessage />
          </FormItem>
        )}
      />
      <FormField
        control={form.control}
        name="email"
        render={({ field }) => (
          <FormItem>
            <FormControl>
              <Input
                {...field}
                id={field.name}
                type="email"
                label="Email"
                placeholder="you@example.com"
                wrapperClassName="mb-0!"
              />
            </FormControl>
            <FormDescription>Where run notifications are sent.</FormDescription>
            <FormMessage />
          </FormItem>
        )}
      />
      <FormField
        control={form.control}
        name="notifyOnFailure"
        render={({ field }) => (
          <FormItem className="flex flex-row items-center justify-between gap-4 space-y-0">
            <div className="flex flex-col gap-1">
              <FormLabel>Email me when a run fails</FormLabel>
              <FormDescription>Sent at most once per hour.</FormDescription>
            </div>
            <FormControl>
              <Switch
                checked={field.value}
                onCheckedChange={field.onChange}
                disabled={field.disabled}
              />
            </FormControl>
          </FormItem>
        )}
      />
      <Button
        type="submit"
        variant="primary"
        size="lg"
        disabled={disabled}
        loading={form.formState.isSubmitting}
      >
        Save profile
      </Button>
    </Form>
  );
}
