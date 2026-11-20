import { usePostV2PublishSkillToMarketplace } from "@/app/api/__generated__/endpoints/store/store";
import { useGetV1ListProviders } from "@/app/api/__generated__/endpoints/integrations/integrations";
import { formatProviderName } from "@/components/contextual/IntegrationsPanel/helpers";
import { useToast } from "@/components/molecules/Toast/use-toast";
import { useStoreCategories } from "@/hooks/useStoreCategories";
import { useState } from "react";

interface Args {
  skillName: string;
}

export function usePublishSkillButton({ skillName }: Args) {
  const { toast } = useToast();
  const { categories } = useStoreCategories();
  const [isOpen, setIsOpen] = useState(false);
  const [category, setCategory] = useState("");
  const [providers, setProviders] = useState<string[]>([]);
  const [submitted, setSubmitted] = useState(false);

  const providersQuery = useGetV1ListProviders({
    query: {
      enabled: isOpen,
      select: (response) => (response.status === 200 ? response.data : []),
    },
  });

  const { mutate: publish, isPending } = usePostV2PublishSkillToMarketplace({
    mutation: {
      onSuccess: (response) => {
        if (response.status !== 201) return;
        setSubmitted(true);
      },
      onError: (error) =>
        toast({
          title: "Couldn't publish this skill",
          description:
            error instanceof Error ? error.message : "Please try again.",
          variant: "destructive",
        }),
    },
  });

  function open() {
    setSubmitted(false);
    setIsOpen(true);
  }

  function submit() {
    publish({
      data: {
        skill_name: skillName,
        categories: [category],
        required_providers: providers,
      },
    });
  }

  return {
    isOpen,
    setIsOpen,
    open,
    submitted,
    isPublishing: isPending,
    canSubmit: category !== "" && !isPending,
    category,
    setCategory,
    categoryOptions: categories.map((c) => ({
      value: c.value,
      label: c.label,
    })),
    providers,
    setProviders,
    providerItems: (providersQuery.data ?? [])
      .filter((provider) => !provider.mcp_server)
      .map((provider) => ({
        value: provider.name,
        label: formatProviderName(provider.name),
      })),
    submit,
  };
}
