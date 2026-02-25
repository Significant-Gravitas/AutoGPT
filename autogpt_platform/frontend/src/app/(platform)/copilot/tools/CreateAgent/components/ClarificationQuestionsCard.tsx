"use client";

import { Button } from "@/components/atoms/Button/Button";
import { Card } from "@/components/atoms/Card/Card";
import { Input } from "@/components/atoms/Input/Input";
import { Text } from "@/components/atoms/Text/Text";
import { cn } from "@/lib/utils";
import { ChatTeardropDotsIcon, CheckCircleIcon } from "@phosphor-icons/react";
import { useEffect, useRef, useState } from "react";
import type { ClarifyingQuestion } from "../helpers";

interface Props {
  questions: ClarifyingQuestion[];
  message: string;
  sessionId?: string;
  onSubmitAnswers: (answers: Record<string, string>) => void;
  onCancel?: () => void;
  isAnswered?: boolean;
  className?: string;
}

export function ClarificationQuestionsCard({
  questions,
  message,
  sessionId,
  onSubmitAnswers,
  onCancel,
  isAnswered = false,
  className,
}: Props) {
  const [answers, setAnswers] = useState<Record<string, string>>({});
  const [isSubmitted, setIsSubmitted] = useState(false);
  const lastSessionIdRef = useRef<string | undefined>(undefined);

  useEffect(() => {
    const storageKey = getStorageKey(sessionId);
    if (!storageKey) {
      setAnswers({});
      setIsSubmitted(false);
      lastSessionIdRef.current = sessionId;
      return;
    }

    try {
      const saved = localStorage.getItem(storageKey);
      if (saved) {
        const parsed = JSON.parse(saved) as Record<string, string>;
        setAnswers(parsed);
      } else {
        setAnswers({});
      }
      setIsSubmitted(false);
    } catch {
      setAnswers({});
      setIsSubmitted(false);
    }
    lastSessionIdRef.current = sessionId;
  }, [sessionId]);

  useEffect(() => {
    if (lastSessionIdRef.current !== sessionId) {
      return;
    }
    const storageKey = getStorageKey(sessionId);
    if (!storageKey) return;

    const hasAnswers = Object.values(answers).some((v) => v.trim());
    try {
      if (hasAnswers) {
        localStorage.setItem(storageKey, JSON.stringify(answers));
      } else {
        localStorage.removeItem(storageKey);
      }
    } catch {}
  }, [answers, sessionId]);

  function handleAnswerChange(keyword: string, value: string) {
    setAnswers((prev) => ({ ...prev, [keyword]: value }));
  }

  function handleSubmit() {
    const allAnswered = questions.every((q) => answers[q.keyword]?.trim());
    if (!allAnswered) {
      return;
    }
    setIsSubmitted(true);
    onSubmitAnswers(answers);

    const storageKey = getStorageKey(sessionId);
    try {
      if (storageKey) {
        localStorage.removeItem(storageKey);
      }
    } catch {}
  }

  const allAnswered = questions.every((q) => answers[q.keyword]?.trim());

  if (isAnswered || isSubmitted) {
    return (
      <div
        className={cn(
          "group relative flex w-full justify-start gap-3 px-4 py-3",
          className,
        )}
      >
        <div className="flex w-full max-w-3xl gap-3">
          <div className="flex-shrink-0">
            <div className="flex h-7 w-7 items-center justify-center rounded-lg bg-green-500">
              <CheckCircleIcon className="h-4 w-4 text-white" weight="bold" />
            </div>
          </div>
          <div className="flex min-w-0 flex-1 flex-col">
            <Card className="p-4">
              <Text variant="h4" className="mb-1 text-slate-900">
                Answers submitted
              </Text>
              <Text variant="small" className="text-slate-600">
                Processing your responses...
              </Text>
            </Card>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div
      className={cn(
        "group relative flex w-full justify-start gap-3",
        className,
      )}
    >
      <div className="flex w-full max-w-3xl gap-3">
        <div className="flex min-w-0 flex-1 flex-col">
          <Card className="space-y-6 p-8">
            <div>
              <div className="flex gap-3">
                <ChatTeardropDotsIcon className="size-6" />
                <Text variant="h4" className="mb-1 text-slate-900">
                  I need more information
                </Text>
              </div>
              <Text variant="body" className="text-slate-600">
                {message}
              </Text>
            </div>

            <div className="space-y-6">
              {questions.map((q, index) => {
                const isAnswered = !!answers[q.keyword]?.trim();

                return (
                  <div
                    key={`${q.keyword}-${index}`}
                    className={cn(
                      "relative rounded-lg border border-dotted p-3",
                      isAnswered
                        ? "border-green-500 bg-green-50/50"
                        : "border-slate-100 bg-slate-50/50",
                    )}
                  >
                    <div className="mb-2 flex items-start gap-2">
                      {isAnswered ? (
                        <CheckCircleIcon
                          size={20}
                          className="mt-0.5 text-green-500"
                          weight="bold"
                        />
                      ) : (
                        <div className="mt-0 flex h-6 w-6 items-center justify-center rounded-full border border-slate-300 font-mono">
                          {index + 1}
                        </div>
                      )}
                      <div className="flex-1">
                        <Text
                          variant="h5"
                          className="mb-2 font-semibold text-slate-900"
                        >
                          {q.question}
                        </Text>
                        {q.example && (
                          <Text
                            variant="body"
                            className="mb-2 italic text-slate-500"
                          >
                            Example: {q.example}
                          </Text>
                        )}
                        <Input
                          type="textarea"
                          id={`clarification-${q.keyword}-${index}`}
                          label={q.question}
                          hideLabel
                          placeholder="Your answer..."
                          rows={2}
                          value={answers[q.keyword] || ""}
                          onChange={(e) =>
                            handleAnswerChange(q.keyword, e.target.value)
                          }
                        />
                      </div>
                    </div>
                  </div>
                );
              })}
            </div>

            <div className="flex max-w-[25rem] gap-2">
              <Button
                onClick={handleSubmit}
                disabled={!allAnswered}
                className="w-auto flex-1"
                variant="primary"
              >
                Submit Answers
              </Button>
              {onCancel && (
                <Button onClick={onCancel} variant="outline">
                  Cancel
                </Button>
              )}
            </div>
          </Card>
        </div>
      </div>
    </div>
  );
}

function getStorageKey(sessionId?: string): string | null {
  if (!sessionId) return null;
  return `clarification_answers_${sessionId}`;
}
