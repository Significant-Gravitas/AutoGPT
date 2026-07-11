"use client";

import { Badge } from "@/components/atoms/Badge/Badge";
import { Button } from "@/components/atoms/Button/Button";
import { Input } from "@/components/atoms/Input/Input";
import { Text } from "@/components/atoms/Text/Text";
import {
  Form,
  FormControl,
  FormField,
  FormItem,
  FormMessage,
} from "@/components/molecules/Form/Form";

import { useAliasesSection } from "./useAliasesSection";

interface Props {
  orgId: string;
  isAdmin: boolean;
}

export function AliasesSection({ orgId, isAdmin }: Props) {
  const { form, aliases, canManage, isCreating, handleCreate } =
    useAliasesSection({ orgId, isAdmin });

  return (
    <section className="flex flex-col gap-4" data-testid="org-aliases-section">
      <div className="flex flex-col gap-1">
        <Text variant="h4" as="h2">
          Aliases
        </Text>
        <Text variant="small" className="text-zinc-500">
          Aliases are old slugs that keep resolving to this organization, so
          existing links stay valid after a rename.
        </Text>
      </div>

      {canManage ? (
        <Form
          form={form}
          onSubmit={handleCreate}
          className="flex max-w-xl items-start gap-3"
        >
          <FormField
            control={form.control}
            name="alias_slug"
            render={({ field }) => (
              <FormItem className="flex-1">
                <FormControl>
                  <Input
                    {...field}
                    id={field.name}
                    label=""
                    hideLabel
                    placeholder="old-slug"
                    wrapperClassName="!mb-0"
                  />
                </FormControl>
                <FormMessage />
              </FormItem>
            )}
          />
          <Button type="submit" loading={isCreating}>
            Add alias
          </Button>
        </Form>
      ) : null}

      {aliases.length > 0 ? (
        <ul className="flex flex-col divide-y divide-zinc-100">
          {aliases.map((alias) => (
            <li
              key={alias.id}
              className="flex items-center gap-3 py-3"
              data-testid="org-alias-row"
            >
              <div className="flex min-w-0 flex-1 flex-col">
                <span className="truncate text-sm font-medium">
                  {alias.alias_slug}
                </span>
                <span className="text-xs text-zinc-500">
                  Added {new Date(alias.created_at).toLocaleDateString()}
                </span>
              </div>
              {alias.alias_type === "RENAME" ? (
                <Badge variant="info">From rename</Badge>
              ) : null}
            </li>
          ))}
        </ul>
      ) : (
        <Text variant="small" className="text-zinc-500">
          No aliases yet.
        </Text>
      )}
    </section>
  );
}
