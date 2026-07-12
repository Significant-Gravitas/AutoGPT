"use client";

import {
  postExperimentalStartSessionRecording,
  postExperimentalStopSessionRecording,
} from "@/app/api/__generated__/endpoints/copilot/copilot";
import type { RecordingStartRequestInterpretationRoute } from "@/app/api/__generated__/models/recordingStartRequestInterpretationRoute";
import type { WorkflowRecording } from "@/app/api/__generated__/models/workflowRecording";
import { useMutation } from "@tanstack/react-query";
import { useEffect, useRef } from "react";
import type { RecordingSettings } from "./recording-helpers";

interface RecordingIdentity {
  sessionID: string;
  recordingID: string;
}

interface StartedRecording extends RecordingIdentity {
  interpretationRoute: RecordingStartRequestInterpretationRoute;
}

interface Args {
  onStarted: (recording: StartedRecording) => void;
  onStopped: (recording: WorkflowRecording) => void;
}

function stopInBackground({ sessionID, recordingID }: RecordingIdentity) {
  void postExperimentalStopSessionRecording(sessionID, {
    recording_id: recordingID,
  }).catch(() => undefined);
}

export function useRecordingRequests({ onStarted, onStopped }: Args) {
  const activeRecordingRef = useRef<RecordingIdentity | null>(null);
  const stopInFlightRef = useRef(false);
  const mountedRef = useRef(false);

  useEffect(() => {
    mountedRef.current = true;
    return () => {
      mountedRef.current = false;
      const activeRecording = activeRecordingRef.current;
      activeRecordingRef.current = null;
      if (activeRecording && !stopInFlightRef.current) {
        stopInBackground(activeRecording);
      }
    };
  }, []);

  const startMutation = useMutation({
    mutationFn: async ({
      sessionID,
      interpretationRoute,
      channels,
    }: RecordingSettings & { sessionID: string }) => {
      const response = await postExperimentalStartSessionRecording(sessionID, {
        mode: "demonstration",
        interpretation_route: interpretationRoute,
        channels,
      });
      if (response.status !== 200) {
        throw new Error("The Local PC executor did not start recording");
      }
      const { recording_id } = response.data;
      const startedRecording: StartedRecording = {
        sessionID,
        recordingID: recording_id,
        interpretationRoute,
      };
      activeRecordingRef.current = startedRecording;
      if (!mountedRef.current) {
        activeRecordingRef.current = null;
        stopInBackground(startedRecording);
        return null;
      }
      return startedRecording;
    },
    onSuccess: (startedRecording) => {
      if (startedRecording && mountedRef.current) onStarted(startedRecording);
    },
  });

  const stopMutation = useMutation({
    mutationFn: async (activeRecording: RecordingIdentity) => {
      stopInFlightRef.current = true;
      try {
        const response = await postExperimentalStopSessionRecording(
          activeRecording.sessionID,
          { recording_id: activeRecording.recordingID },
        );
        if (response.status !== 200) {
          throw new Error("The Local PC executor did not stop recording");
        }
        return { activeRecording, recording: response.data.recording };
      } catch (error) {
        stopInFlightRef.current = false;
        if (!mountedRef.current) {
          stopInBackground(activeRecording);
        }
        throw error;
      }
    },
    onSuccess: ({ activeRecording, recording }) => {
      stopInFlightRef.current = false;
      if (
        activeRecordingRef.current?.sessionID === activeRecording.sessionID &&
        activeRecordingRef.current.recordingID === activeRecording.recordingID
      ) {
        activeRecordingRef.current = null;
      }
      if (mountedRef.current) onStopped(recording);
    },
  });

  function start(sessionID: string, settings: RecordingSettings) {
    startMutation.reset();
    stopMutation.reset();
    startMutation.mutate({ sessionID, ...settings });
  }

  function stop() {
    const activeRecording = activeRecordingRef.current;
    if (!activeRecording) return;
    stopInFlightRef.current = true;
    stopMutation.reset();
    stopMutation.mutate(activeRecording);
  }

  function stopActive() {
    const activeRecording = activeRecordingRef.current;
    activeRecordingRef.current = null;
    if (activeRecording && !stopInFlightRef.current) {
      stopInBackground(activeRecording);
    }
  }

  function reset() {
    startMutation.reset();
    stopMutation.reset();
  }

  return {
    start,
    stop,
    stopActive,
    reset,
    isStarting: startMutation.isPending,
    isStopping: stopMutation.isPending,
    hasStartError: startMutation.isError,
    hasStopError: stopMutation.isError,
  };
}
