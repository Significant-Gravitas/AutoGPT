import type { CardDecision } from "@/app/api/__generated__/models/cardDecision";
import type { TaskActionRequest } from "@/app/api/__generated__/models/taskActionRequest";
import {
  useInfiniteQuery,
  useMutation,
  useQuery,
  useQueryClient,
  type QueryClient,
} from "@tanstack/react-query";
import { tasksApi } from "./api";
import type {
  CreateTaskInput,
  TaskFilters,
  TaskPlan,
  UpdateTaskInput,
} from "./types";

export const tasksKeys = {
  all: ["tasks-v1"] as const,
  list: (filters: TaskFilters) => ["tasks-v1", "list", filters] as const,
  detail: (id: string) => ["tasks-v1", "detail", id] as const,
  activity: (id: string) => ["tasks-v1", "activity", id] as const,
  activityLog: (id: string) => ["tasks-v1", "activity-log", id] as const,
  summary: (id: string) => ["tasks-v1", "summary", id] as const,
  evidence: (id: string, evidenceId: string) =>
    ["tasks-v1", "evidence", id, evidenceId] as const,
  experts: ["tasks-v1", "experts"] as const,
  plan: (id: string) => ["tasks-v1", "plan", id] as const,
};

export function invalidateTasks(queryClient: QueryClient) {
  return queryClient.invalidateQueries({ queryKey: tasksKeys.all });
}

export function useTaskList(
  filters: TaskFilters,
  { enabled = true }: { enabled?: boolean } = {},
) {
  return useQuery({
    queryKey: tasksKeys.list(filters),
    queryFn: () => tasksApi.list(filters),
    enabled,
  });
}

export function useTaskInbox(
  filters: TaskFilters,
  { enabled = true }: { enabled?: boolean } = {},
) {
  return useQuery({
    queryKey: [...tasksKeys.list(filters), "inbox"],
    queryFn: () => tasksApi.listWithCounts(filters),
    enabled,
  });
}

export function useTasksNeedingMe() {
  return useQuery({
    queryKey: [...tasksKeys.all, "needs-you"],
    queryFn: () => tasksApi.needsYou(),
  });
}

export function useTaskDetail(id: string | null) {
  return useQuery({
    queryKey: tasksKeys.detail(id ?? ""),
    queryFn: () => tasksApi.get(id!),
    enabled: !!id,
  });
}

export function useTaskActivity(id: string | null) {
  return useQuery({
    queryKey: tasksKeys.activity(id ?? ""),
    queryFn: () => tasksApi.activity(id!),
    enabled: !!id,
  });
}

export function useTaskExperts() {
  return useQuery({
    queryKey: tasksKeys.experts,
    queryFn: () => tasksApi.experts(),
    staleTime: 60_000,
  });
}

export function useTaskPlan(planId: string | null) {
  return useQuery({
    queryKey: tasksKeys.plan(planId ?? ""),
    queryFn: () => tasksApi.getPlan(planId!),
    enabled: !!planId,
  });
}

export function useCreateTask() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (input: CreateTaskInput) => tasksApi.create(input),
    onSuccess: () => invalidateTasks(queryClient),
  });
}

export function useUpdateTask() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: ({ id, input }: { id: string; input: UpdateTaskInput }) =>
      tasksApi.update(id, input),
    onSuccess: () => invalidateTasks(queryClient),
  });
}

export function useStartTask() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (id: string) => tasksApi.start(id),
    onSuccess: () => invalidateTasks(queryClient),
  });
}

export function useCommentOnTask() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: ({ id, body }: { id: string; body: string }) =>
      tasksApi.comment(id, body),
    onSuccess: () => invalidateTasks(queryClient),
  });
}

export function useAnswerTaskQuestion() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: ({
      id,
      answer,
      inputId,
    }: {
      id: string;
      answer: string;
      inputId?: string;
    }) => tasksApi.answer(id, answer, inputId),
    onSuccess: () => invalidateTasks(queryClient),
  });
}

export function useTaskSummary(id: string | null) {
  return useQuery({
    queryKey: tasksKeys.summary(id ?? ""),
    queryFn: () => tasksApi.summary(id!),
    enabled: !!id,
  });
}

export function useTaskActivityLog(id: string, { enabled = true } = {}) {
  return useInfiniteQuery({
    queryKey: tasksKeys.activityLog(id),
    queryFn: ({ pageParam }) => tasksApi.activityPage(id, pageParam),
    initialPageParam: null as string | null,
    getNextPageParam: (last) => last.nextCursor,
    enabled,
  });
}

export function useTaskEvidence(
  id: string,
  evidenceId: string,
  { enabled = true } = {},
) {
  return useQuery({
    queryKey: tasksKeys.evidence(id, evidenceId),
    queryFn: () => tasksApi.evidence(id, evidenceId),
    enabled,
    staleTime: Infinity,
  });
}

export function useAnswerAsk() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: ({
      id,
      answer,
      askId,
    }: {
      id: string;
      answer: string;
      askId?: string;
    }) => tasksApi.answerAsk(id, answer, askId),
    onSuccess: () => invalidateTasks(queryClient),
  });
}

export function useAnswerApprovals() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: ({
      id,
      decisions,
    }: {
      id: string;
      decisions: CardDecision[];
    }) => tasksApi.answerApprovals(id, decisions),
    onSuccess: () => invalidateTasks(queryClient),
  });
}

export function useTaskAction() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: ({
      id,
      action,
      body,
    }: {
      id: string;
      action: string;
      body?: TaskActionRequest;
    }) => tasksApi.action(id, action, body),
    onSuccess: () => invalidateTasks(queryClient),
  });
}

export function useApproveTaskPlan() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (plan: TaskPlan) => tasksApi.approvePlan(plan),
    onSuccess: () => invalidateTasks(queryClient),
  });
}
