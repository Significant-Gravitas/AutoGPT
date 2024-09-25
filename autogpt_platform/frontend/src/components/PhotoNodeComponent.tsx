import React, { useState, useRef, useEffect, useCallback } from "react";
import { Button } from "@/components/ui/button";
import { Camera, X } from "lucide-react";

const PhotoNodeComponent = ({
  data,
  handleInputChange,
  generateOutputHandles,
}) => {
  const [isCameraActive, setIsCameraActive] = useState(false);
  const [errorMessage, setErrorMessage] = useState("");
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const streamRef = useRef(null);

  const stopCamera = useCallback(() => {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((track) => track.stop());
      streamRef.current = null;
    }
    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }
    setIsCameraActive(false);
  }, []);

  useEffect(() => {
    return () => {
      stopCamera();
    };
  }, [stopCamera]);

  const startCamera = async () => {
    setErrorMessage("");
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: true });
      streamRef.current = stream;
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        videoRef.current.onloadedmetadata = () => {
          videoRef.current.play();
          setIsCameraActive(true);
        };
      } else {
        throw new Error("Video element is not available");
      }
    } catch (error) {
      console.error("Error accessing camera:", error);
      setErrorMessage(`Failed to start camera: ${error.message}`);
    }
  };

  const capturePhoto = () => {
    if (!videoRef.current || !canvasRef.current) {
      setErrorMessage(
        "Failed to capture photo: Camera not initialized properly",
      );
      return;
    }
    const video = videoRef.current;
    const canvas = canvasRef.current;
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    canvas.getContext("2d").drawImage(video, 0, 0);
    const imageDataUrl = canvas.toDataURL("image/jpeg");
    handleInputChange("image_data", imageDataUrl.split(",")[1]);
    stopCamera();
  };

  return (
    <div className="p-3">
      {errorMessage && <div className="mb-2 text-red-500">{errorMessage}</div>}
      <input
        type="text"
        value={data.hardcodedValues?.image_data || ""}
        onChange={(e) => handleInputChange("image_data", e.target.value)}
        placeholder="Enter Image Data"
        className="mb-2 w-full rounded border p-2"
      />
      <div style={{ display: isCameraActive ? "block" : "none" }}>
        <video
          ref={videoRef}
          style={{ width: "100%", maxHeight: "200px", objectFit: "cover" }}
          playsInline
        />
      </div>
      <canvas ref={canvasRef} style={{ display: "none" }} />
      {!isCameraActive ? (
        <Button onClick={startCamera} className="w-full">
          <Camera className="mr-2 h-4 w-4" /> Open Camera
        </Button>
      ) : (
        <div className="flex justify-between">
          <Button onClick={capturePhoto} className="mr-2 flex-1">
            <Camera className="mr-2 h-4 w-4" /> Capture
          </Button>
          <Button
            onClick={stopCamera}
            variant="outline"
            className="ml-2 flex-1"
          >
            <X className="mr-2 h-4 w-4" /> Close Camera
          </Button>
        </div>
      )}
      {generateOutputHandles(data.outputSchema, data.uiType)}
    </div>
  );
};

export default PhotoNodeComponent;
