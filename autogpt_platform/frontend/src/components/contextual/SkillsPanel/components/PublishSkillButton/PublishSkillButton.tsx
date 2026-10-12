"use client";

import { Button } from "@/components/atoms/Button/Button";
import { Icon } from "@/components/atoms/Icon/Icon";
import { Select } from "@/components/atoms/Select/Select";
import { Text } from "@/components/atoms/Text/Text";
import { Dialog } from "@/components/molecules/Dialog/Dialog";
import { MultiToggle } from "@/components/molecules/MultiToggle/MultiToggle";
import { CheckmarkCircle02Icon, Store01Icon } from "@hugeicons/core-free-icons";
import { usePublishSkillButton } from "./usePublishSkillButton";

interface Props {
  skillName: string;
}

export function PublishSkillButton({ skillName }: Props) {
  const {
    isOpen,
    setIsOpen,
    open,
    submitted,
    isPublishing,
    canSubmit,
    category,
    setCategory,
    categoryOptions,
    providers,
    setProviders,
    providerItems,
    submit,
  } = usePublishSkillButton({ skillName });

  return (
    <>
      <Button
        variant="icon"
        size="icon"
        onClick={open}
        data-testid="skill-publish-button"
        aria-label="Publish skill to marketplace"
      >
        <Icon icon={Store01Icon} className="h-4 w-4" />
      </Button>

      <Dialog
        controlled={{ isOpen, set: setIsOpen }}
        styling={{ maxWidth: "32rem" }}
        title="Publish to marketplace"
      >
        <Dialog.Content>
          {submitted ? (
            <div
              className="flex items-start gap-2"
              data-testid="skill-publish-submitted"
            >
              <Icon
                icon={CheckmarkCircle02Icon}
                size={18}
                className="mt-0.5 text-emerald-600"
              />
              <Text variant="body">
                Submitted for review. Your skill appears in the marketplace once
                it is approved; what you published is a copy, so editing it here
                changes nothing until you publish again.
              </Text>
            </div>
          ) : (
            <div className="flex flex-col gap-4">
              <Text variant="body" className="!text-zinc-500">
                Publishing sends a copy of <strong>{skillName}</strong> for
                review. Say where it belongs and what it expects to be
                connected.
              </Text>
              <Select
                id="skill-category"
                label="Category"
                placeholder="Pick a category"
                value={category}
                onValueChange={setCategory}
                options={categoryOptions}
              />
              <div className="flex flex-col gap-2">
                <Text variant="large-medium">Works with</Text>
                <Text variant="small" className="!text-zinc-500">
                  Integrations your instructions assume. Installers see these
                  and connect them when they first need one.
                </Text>
                <MultiToggle
                  items={providerItems}
                  selectedValues={providers}
                  onChange={setProviders}
                  aria-label="Integrations this skill works with"
                />
              </div>
              <div className="flex justify-end">
                <Button
                  variant="primary"
                  onClick={submit}
                  disabled={!canSubmit}
                  loading={isPublishing}
                  data-testid="skill-publish-submit"
                >
                  Submit for review
                </Button>
              </div>
            </div>
          )}
        </Dialog.Content>
      </Dialog>
    </>
  );
}
