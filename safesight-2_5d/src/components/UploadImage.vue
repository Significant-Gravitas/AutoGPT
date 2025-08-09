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
  const tileSize = 256
  const tileOverlap = 1
  const levels = Math.ceil(Math.log2(Math.max(bitmap.width, bitmap.height))) + 1
  const id = 'img-' + Date.now()

  const { map, getTileUrl } = await generateTiles(bitmap, { tileSize, tileOverlap, levels })

  const tileSource = {
    width: bitmap.width,
    height: bitmap.height,
    tileSize,
    tileOverlap,
    minLevel: 0,
    maxLevel: levels - 1,
    getTileUrl(level: number, x: number, y: number) {
      return getTileUrl(level, x, y)
    },
    // OpenSeadragon 4+ may look for fileFormat; but Blob URLs include mime type already.
  }

  imageStore.setImage({
    id,
    name: file.name,
    width: bitmap.width,
    height: bitmap.height,
    tileSize,
    tileOverlap,
    levels,
    tileSource,
  })

  return false
}

async function generateTiles(
  bitmap: ImageBitmap,
  opts: { tileSize: number; tileOverlap: number; levels: number }
): Promise<{ map: Record<string, string>, getTileUrl: (level: number, x: number, y: number) => string }>{
  const { tileSize, tileOverlap, levels } = opts
  const blobMap: Record<string, string> = {}

  for (let level = 0; level < levels; level++) {
    const scale = 2 ** (levels - level - 1)
    const levelWidth = Math.ceil(bitmap.width / scale)
    const levelHeight = Math.ceil(bitmap.height / scale)

    const cols = Math.ceil(levelWidth / tileSize)
    const rows = Math.ceil(levelHeight / tileSize)

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
        const key = `${level}/${x}_${y}.jpg`
        blobMap[key] = URL.createObjectURL(blob)
      }
    }
  }

  return {
    map: blobMap,
    getTileUrl(level: number, x: number, y: number) {
      const key = `${level}/${x}_${y}.jpg`
      return blobMap[key]
    }
  }
}
</script>