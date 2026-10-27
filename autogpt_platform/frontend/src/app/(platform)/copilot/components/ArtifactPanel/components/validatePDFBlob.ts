const PDF_MIME_TYPE = "application/pdf";
const PDF_SIGNATURE = new Uint8Array([0x25, 0x50, 0x44, 0x46, 0x2d]);

export async function validatePDFBlob(blob: Blob): Promise<void> {
  const mimeType = blob.type.split(";", 1)[0]?.trim().toLowerCase();
  const header = new Uint8Array(
    await blob.slice(0, PDF_SIGNATURE.length).arrayBuffer(),
  );
  const hasPDFSignature = PDF_SIGNATURE.every(
    (byte, index) => header[index] === byte,
  );

  if (mimeType !== PDF_MIME_TYPE || !hasPDFSignature) {
    throw new Error("The file response is not a valid PDF");
  }
}
