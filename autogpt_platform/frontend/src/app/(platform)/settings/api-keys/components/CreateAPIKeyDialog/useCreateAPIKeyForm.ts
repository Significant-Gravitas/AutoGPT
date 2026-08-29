"use client";

import { useState } from "react";
import { zodResolver } from "@hookform/resolvers/zod";
import { useForm } from "react-hook-form";

import { useCreateAPIKey } from "../hooks/useCreateAPIKey";
import { createAPIKeySchema, type CreateAPIKeyFormValues } from "./schema";

type View = "form" | "success";

interface Args {
  onClose: () => void;
}

export function useCreateAPIKeyForm({ onClose }: Args) {
  const [view, setView] = useState<View>("form");
  const [plainTextKey, setPlainTextKey] = useState("");
  const { createKey, isPending } = useCreateAPIKey();

  const form = useForm<CreateAPIKeyFormValues>({
    resolver: zodResolver(createAPIKeySchema),
    defaultValues: { name: "", description: "", permissions: [] },
    mode: "onChange",
  });

  async function handleSubmit(values: CreateAPIKeyFormValues) {
    try {
      const result = await createKey({
        name: values.name,
        description: values.description || null,
        permissions: values.permissions,
      });
      setPlainTextKey(result.plain_text_key);
      setView("success");
    } catch {
      // Toast is surfaced by useCreateAPIKey; keep form open so user can retry.
    }
  }

  function handleClose() {
    setView("form");
    setPlainTextKey("");
    form.reset();
    onClose();
  }

  return {
    form,
    view,
    plainTextKey,
    isPending,
    handleSubmit,
    handleClose,
  };
}
