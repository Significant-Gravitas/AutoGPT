"use client";

import { useState } from "react";
import { CaretDown } from "@phosphor-icons/react";
import type { LlmModel, LlmProvider } from "@/lib/autogpt-server-api/types";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "@/components/molecules/DropdownMenu/DropdownMenu";
import { Button } from "@/components/atoms/Button/Button";
import { AddProviderForm } from "./AddProviderForm";
import { AddModelForm } from "./AddModelForm";
import { ProviderList } from "./ProviderList";
import { ModelsTable } from "./ModelsTable";

interface Props {
  providers: LlmProvider[];
  models: LlmModel[];
}

type FormType = "model" | "provider" | null;

export function LlmRegistryDashboard({ providers, models }: Props) {
  const [activeForm, setActiveForm] = useState<FormType>(null);

  function handleFormSelect(type: FormType) {
    setActiveForm(activeForm === type ? null : type);
  }

  return (
    <div className="mx-auto p-6">
      <div className="flex flex-col gap-6">
        {/* Header */}
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-3xl font-bold">LLM Registry</h1>
            <p className="text-gray-500">
              Manage supported providers, models, and credit pricing
            </p>
          </div>
          <DropdownMenu>
            <DropdownMenuTrigger asChild>
              <Button variant="primary" size="small" className="gap-2">
                Add New
                <CaretDown className="h-4 w-4" />
              </Button>
            </DropdownMenuTrigger>
            <DropdownMenuContent align="end">
              <DropdownMenuItem onClick={() => handleFormSelect("model")}>
                Model
              </DropdownMenuItem>
              <DropdownMenuItem onClick={() => handleFormSelect("provider")}>
                Provider
              </DropdownMenuItem>
            </DropdownMenuContent>
          </DropdownMenu>
        </div>

        {/* Add Forms Section */}
        {activeForm && (
          <div className="rounded-lg border bg-white p-6 shadow-sm">
            <div className="mb-4 flex items-center justify-between">
              <h2 className="text-xl font-semibold">
                Add {activeForm === "model" ? "Model" : "Provider"}
              </h2>
              <button
                type="button"
                onClick={() => setActiveForm(null)}
                className="text-sm text-gray-600 hover:text-gray-900"
              >
                Close
              </button>
            </div>
            {activeForm === "model" ? (
              <AddModelForm providers={providers} />
            ) : (
              <AddProviderForm />
            )}
          </div>
        )}

        {/* Providers Section */}
        <div className="rounded-lg border bg-white p-6 shadow-sm">
          <div className="mb-4">
            <h2 className="text-xl font-semibold">Providers</h2>
            <p className="mt-1 text-sm text-gray-600">
              Default credentials and feature flags for upstream vendors
            </p>
          </div>
          <ProviderList providers={providers} />
        </div>

        {/* Models Section */}
        <div className="rounded-lg border bg-white p-6 shadow-sm">
          <div className="mb-4">
            <h2 className="text-xl font-semibold">Models</h2>
            <p className="mt-1 text-sm text-gray-600">
              Toggle availability, adjust context windows, and update credit
              pricing
            </p>
          </div>
          <ModelsTable models={models} providers={providers} />
        </div>
      </div>
    </div>
  );
}
