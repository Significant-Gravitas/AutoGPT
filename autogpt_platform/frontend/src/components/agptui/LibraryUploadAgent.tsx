"use client";
import { Upload, X } from "lucide-react";
import { Button } from "./Button";
import { useEffect, useState } from "react";
import { motion, useAnimation } from "framer-motion";
import { cn } from "@/lib/utils";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "../ui/dialog";
import { Input } from "../ui/input";
import { FileUploader } from "react-drag-drop-files";

const fileTypes = ["JSON"];

export const LibraryUploadAgent = () => {
  const [scrolled, setScrolled] = useState(false);
  const [file, setFile] = useState<File | null>(null);
  const [isDroped, setisDroped] = useState(false);
  const controls = useAnimation();
  const handleChange = (file: File) => {
    setTimeout(() => {
      setisDroped(false);
    }, 2000);
    setFile(file);
    setisDroped(false);
  };

  useEffect(() => {
    const handleScroll = () => {
      if (window.scrollY > 30) {
        setScrolled(true);
      } else {
        setScrolled(false);
      }
    };

    window.addEventListener("scroll", handleScroll);
    return () => window.removeEventListener("scroll", handleScroll);
  }, []);

  const handleUpload = () => {
    // Add upload logic here
    if (file) {
      console.log("Uploading file:", file);
    }
  };

  return (
    <Dialog>
      <DialogTrigger asChild>
        <Button
          variant="library_primary"
          size="library"
          className={cn(
            "max-w-[177px] transition-all duration-200 ease-in-out",
            scrolled ? "w-fit max-w-fit" : "w-fit sm:w-[177px]",
          )}
        >
          <motion.div animate={controls}>
            <Upload
              className={cn(
                "h-5 w-5 transition-all duration-200 ease-in-out",
                !scrolled && "sm:mr-2",
              )}
            />
          </motion.div>
          {!scrolled && (
            <motion.div
              initial={{ opacity: 1 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              className="hidden items-center transition-opacity duration-300 sm:inline-flex"
            >
              Upload an agent
            </motion.div>
          )}
        </Button>
      </DialogTrigger>
      <DialogContent>
        <DialogHeader>
          <DialogTitle className="mb-8 text-center">Upload Agent </DialogTitle>
        </DialogHeader>

        <div className="relative flex flex-col gap-4">
          <Input placeholder="Agent name" className="w-full rounded-[10px]" />
          <Input placeholder="Description" className="w-full rounded-[10px]" />

          {file ? (
            <div className="flex rounded-[10px] border p-2 font-sans text-sm font-medium text-[#525252] outline-none">
              <span className="line-clamp-1">{file.name}</span>
              <Button
                onClick={() => setFile(null)}
                className="absolute left-[-10px] top-[-16px] mt-2 h-fit border-none bg-red-200 p-1"
                size="library"
              >
                <X
                  className="m-0 h-[12px] w-[12px] text-red-600"
                  strokeWidth={3}
                />
              </Button>
            </div>
          ) : (
            <FileUploader
              handleChange={handleChange}
              name="file"
              types={fileTypes}
              label={"Upload your agent here..!!"}
              uploadedLabel={"Uploading Successful"}
              required={true}
              hoverTitle={"Drop your agent here...!!"}
              maxSize={10}
              classes={"drop-style"}
              onDrop={() => {
                setisDroped(true);
              }}
              onSelect={() => {
                setisDroped(true);
              }}
              children={
                <div
                  style={{
                    minHeight: "150px",
                    display: "flex",
                    flexDirection: "column",
                    justifyContent: "center",
                    alignItems: "center",
                    outline: "none",
                    fontFamily: "var(--font-geist-sans)",
                    color: "#525252",
                    fontSize: "14px",
                    fontWeight: "500",
                  }}
                >
                  {isDroped ? (
                    <div className="flex items-center justify-center py-4">
                      <div className="h-8 w-8 animate-spin rounded-full border-b-2 border-t-2 border-neutral-800"></div>
                    </div>
                  ) : (
                    <>
                      <span>Drop your agent here</span>
                      <span>or</span>
                      <span>Click to upload</span>
                    </>
                  )}
                </div>
              }
            />
          )}

          <Button
            onClick={handleUpload}
            variant="library_primary"
            size="library"
            className="mt-2 self-end"
            disabled={!file}
          >
            Upload Agent
          </Button>
        </div>
      </DialogContent>
    </Dialog>
  );
};
