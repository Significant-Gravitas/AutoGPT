<template>
  <el-upload
    :show-file-list="false"
    accept="image/*"
    :before-upload="onBeforeUpload"
  >
    <el-button>上传底图</el-button>
  </el-upload>
</template>

<script setup lang="ts">
import { useImageStore } from '../stores/imageStore'

const imageStore = useImageStore()

async function onBeforeUpload(file: File) {
  const bitmap = await createImageBitmap(file)
  // Generate Deep Zoom Image (DZI) XML and tiles in memory using OffscreenCanvas
  const tileSize = 256
  const tileOverlap = 1
  const levels = Math.ceil(Math.log2(Math.max(bitmap.width, bitmap.height))) + 1
  const dziId = 'dzi-' + Date.now()

  const tilesBaseUrl = await generateTiles(bitmap, { tileSize, tileOverlap, levels, dziId })
  const dziUrl = `${tilesBaseUrl}/descriptor.dzi`

  imageStore.setImage({
    id: dziId,
    name: file.name,
    width: bitmap.width,
    height: bitmap.height,
    tileSize,
    tileOverlap,
    levels,
    dziUrl,
  })

  return false
}

async function generateTiles(
  bitmap: ImageBitmap,
  opts: { tileSize: number; tileOverlap: number; levels: number; dziId: string }
): Promise<string> {
  const { tileSize, tileOverlap, levels, dziId } = opts
  // Use an in-memory URL space via Service Worker-less approach: Blob URLs per tile placed under an Object URL folder concept.
  // We'll create a temporary cache map and a base URL scheme using URL.createObjectURL for a manifest.
  const tileEntries: { path: string; blob: Blob }[] = []

  // Pyramid from top (smallest) to base (largest)
  for (let level = 0; level < levels; level++) {
    const scale = 2 ** (levels - level - 1)
    const levelWidth = Math.ceil(bitmap.width / scale)
    const levelHeight = Math.ceil(bitmap.height / scale)

    const cols = Math.ceil(levelWidth / tileSize)
    const rows = Math.ceil(levelHeight / tileSize)

    // Draw the scaled level image
    const levelCanvas = new OffscreenCanvas(levelWidth, levelHeight)
    const lctx = levelCanvas.getContext('2d')!
    lctx.imageSmoothingEnabled = true
    lctx.imageSmoothingQuality = 'high'
    lctx.drawImage(bitmap, 0, 0, levelWidth, levelHeight)

    for (let y = 0; y < rows; y++) {
      for (let x = 0; x < cols; x++) {
        const sx = x * tileSize
        const sy = y * tileSize
        const sw = Math.min(tileSize + (x < cols - 1 ? tileOverlap : 0), levelWidth - sx)
        const sh = Math.min(tileSize + (y < rows - 1 ? tileOverlap : 0), levelHeight - sy)

        const tile = new OffscreenCanvas(sw, sh)
        const tctx = tile.getContext('2d')!
        tctx.drawImage(levelCanvas, sx, sy, sw, sh, 0, 0, sw, sh)
        const blob = await tile.convertToBlob({ type: 'image/jpeg', quality: 0.9 })
        const path = `${dziId}/files/${level}/${x}_${y}.jpg`
        tileEntries.push({ path, blob })
      }
    }
  }

  // Create DZI descriptor
  const dziXml = `<?xml version="1.0" encoding="UTF-8"?>\n<Image TileSize="${tileSize}" Overlap="${tileOverlap}" Format="jpg" xmlns="http://schemas.microsoft.com/deepzoom/2008">\n  <Size Width="${bitmap.width}" Height="${bitmap.height}"/>\n</Image>`
  const dziBlob = new Blob([dziXml], { type: 'application/xml' })
  const dziPath = `${dziId}/descriptor.dzi`
  tileEntries.push({ path: dziPath, blob: dziBlob })

  // Build an in-memory URL space by returning a base URL that resolves via custom OSD tileSource with fetch hook.
  // We'll serve via a simple URL.createObjectURL for a zip-like bundle index.json + blobs map.
  const index = tileEntries.map(e => e.path)
  const blobs: Record<string, string> = {}
  for (const e of tileEntries) {
    blobs[e.path] = URL.createObjectURL(e.blob)
  }
  const manifest = { index, blobs }
  const manifestBlob = new Blob([JSON.stringify(manifest)], { type: 'application/json' })
  const baseUrl = URL.createObjectURL(manifestBlob)

  // Monkey patch global fetch for this base URL scope
  const originalFetch = window.fetch.bind(window)
  window.fetch = async (input: RequestInfo | URL, init?: RequestInit): Promise<Response> => {
    try {
      const url = typeof input === 'string' ? input : (input as URL).toString()
      if (url.startsWith(baseUrl)) {
        const manifestText = await originalFetch(baseUrl).then(r => r.text())
        const { blobs } = JSON.parse(manifestText)
        const suffix = url.substring(baseUrl.length)
        const key = decodeURIComponent(suffix.replace(/^\//, ''))
        const blobUrl = blobs[key]
        if (blobUrl) return originalFetch(blobUrl, init)
      }
    } catch (e) {
      // fallthrough
    }
    return originalFetch(input as any, init)
  }

  return baseUrl
}
</script>