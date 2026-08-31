import { Expert } from "@/app/api/__generated__/models/expert";
import { customMutator } from "@/app/api/mutators/custom-mutator";
import { useMutation, useQuery } from "@tanstack/react-query";

export interface OfficeTemplateExpert {
  template_id: string;
  name: string;
  role: string;
  avatar_url: string | null;
  tagline: string | null;
  schedule_cron: string | null;
  intro_task_title: string | null;
}

export interface OfficeTemplate {
  id: string;
  name: string;
  description: string;
  experts: OfficeTemplateExpert[];
}

export interface HiredOfficeExpert {
  expert: Expert;
  intro_task_id: string | null;
  intro_task_title: string | null;
  schedule_created: boolean;
}

export interface HireOfficeResponse {
  office_template_id: string;
  office_name: string;
  hired: HiredOfficeExpert[];
}

interface ApiResponse<T> {
  data: T;
  status: number;
  headers: Headers;
}

export const OFFICE_TEMPLATES_URL = "/api/experts/office-templates";
export const HIRE_OFFICE_URL = "/api/experts/hire-office";

// Seam: replace these two hooks with the orval-generated ones once the
// client is regenerated; call sites stay unchanged.
export function useOfficeTemplatesQuery() {
  return useQuery({
    queryKey: [OFFICE_TEMPLATES_URL],
    queryFn: async () => {
      const res = await customMutator<ApiResponse<OfficeTemplate[]>>(
        OFFICE_TEMPLATES_URL,
        { method: "GET" },
      );
      return res.data;
    },
  });
}

export function useHireOfficeMutation(options?: {
  onSuccess?: (result: HireOfficeResponse) => void;
  onError?: (error: unknown) => void;
}) {
  return useMutation({
    mutationFn: async (body: { template_id: string }) => {
      const res = await customMutator<ApiResponse<HireOfficeResponse>>(
        HIRE_OFFICE_URL,
        { method: "POST", body: JSON.stringify(body) },
      );
      return res.data;
    },
    ...options,
  });
}
