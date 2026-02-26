# SECRT-2047: Add Nano Banana 2 to Image Generator, Customizer, and Editor Blocks

## Summary
Add `google/nano-banana-2` (Gemini 3.1 Flash Image) to the three image blocks. NB2 shares the same Replicate API schema as NB/NBP (prompt, aspect_ratio, output_format, image_input, resolution) with minor additions (new aspect ratios, 4K resolution option).

---

## File 1: `ai_image_generator_block.py`

### Changes

1. **Add enum value** to `ImageGenModel` (line ~107):
   ```python
   NANO_BANANA_2 = "Nano Banana 2"
   ```

2. **Add generation branch** in `generate_image()` method (after the `NANO_BANANA_PRO` branch, ~line 202):
   ```python
   elif input_data.model == ImageGenModel.NANO_BANANA_2:
       input_params = {
           "prompt": modified_prompt,
           "aspect_ratio": SIZE_TO_NANO_BANANA_RATIO[input_data.size],
           "resolution": "2K",
           "output_format": "jpg",
           "safety_filter_level": "block_only_high",
       }
       output = await self._run_client(
           credentials, "google/nano-banana-2", input_params
       )
       return output
   ```

   This is identical to the NBP branch except the model name. The existing `SIZE_TO_NANO_BANANA_RATIO` mapping (line ~67) already works for NB2 — same supported ratios.

3. **Update test_input** — no change needed (test uses RECRAFT).

### Notes
- NB2 supports additional aspect ratios (1:4, 4:1, 1:8, 8:1) and 4K resolution, but the generator block uses semantic `ImageSize` enums mapped to standard ratios, so these extras aren't exposed. This is fine — matching existing behavior.

---

## File 2: `ai_image_customizer.py`

### Changes

1. **Add enum value** to `GeminiImageModel` (line ~20):
   ```python
   NANO_BANANA_2 = "google/nano-banana-2"
   ```

2. **No other changes needed.** The customizer already passes `model.value` directly as the Replicate model name (line ~120: `model_name=input_data.model.value`), and the `run_model()` method (line ~128) uses it generically. NB2 accepts the same parameters (prompt, aspect_ratio, output_format, image_input).

3. **Update block description** (line ~82) to mention Nano Banana 2:
   ```python
   description=(
       "Generate and edit custom images using Google's Nano-Banana models from Gemini. "
       "Provide a prompt and optional reference images to create or modify images."
   ),
   ```

---

## File 3: `flux_kontext.py` (AI Image Editor Block)

### Changes

This block currently only supports Flux Kontext models. Adding NB2 requires a model-aware branching approach since NB2 has different API parameters than Flux Kontext.

1. **Add new model enum or extend existing** — two options:

   **Option A (Recommended): Extend `FluxKontextModelName`** by renaming it to something generic and adding NB Pro + NB2:
   ```python
   class ImageEditorModel(str, Enum):
       FLUX_KONTEXT_PRO = "Flux Kontext Pro"
       FLUX_KONTEXT_MAX = "Flux Kontext Max"
       NANO_BANANA_PRO = "Nano Banana Pro"
       NANO_BANANA_2 = "Nano Banana 2"

       @property
       def api_name(self) -> str:
           _map = {
               "FLUX_KONTEXT_PRO": "black-forest-labs/flux-kontext-pro",
               "FLUX_KONTEXT_MAX": "black-forest-labs/flux-kontext-max",
               "NANO_BANANA_PRO": "google/nano-banana-pro",
               "NANO_BANANA_2": "google/nano-banana-2",
           }
           return _map[self.name]
   ```

   **Option B: Keep `FluxKontextModelName` and add a separate enum.** This is messier — Option A is cleaner.

2. **Update `run_model()`** (line ~133) to handle NB2's different input schema:
   - Flux Kontext uses `input_image` (single image, base64)
   - NB2 uses `image_input` (list of images), `aspect_ratio`, `output_format`
   - NB2 doesn't use `seed`

   ```python
   async def run_model(self, ...):
       client = ReplicateClient(api_token=api_key.get_secret_value())

       if "nano-banana" in model_name:
           input_params = {
               "prompt": prompt,
               "aspect_ratio": aspect_ratio,
               "output_format": "jpg",
           }
           if input_image_b64:
               input_params["image_input"] = [input_image_b64]
       else:
           # Existing Flux Kontext logic
           input_params = {
               "prompt": prompt,
               "input_image": input_image_b64,
               "aspect_ratio": aspect_ratio,
               **({"seed": seed} if seed is not None else {}),
           }
       # ... rest unchanged
   ```

3. **Update `AspectRatio` enum** — the existing enum in this file already covers all NB2 ratios (1:1, 16:9, 9:16, 4:3, 3:4, 3:2, 2:3, 4:5, 5:4, 21:9). No changes needed.

4. **Update block description and credentials description** to mention NB2 alongside Flux Kontext.

5. **Rename class references** if renaming the enum (Input schema field type, default value).

---

## NB2-Specific Parameters & Differences

| Parameter | NB | NBP | NB2 | Notes |
|---|---|---|---|---|
| `prompt` | ✅ | ✅ | ✅ | Same |
| `aspect_ratio` | ✅ | ✅ | ✅ | NB2 adds 1:4, 4:1, 1:8, 8:1 |
| `output_format` | ✅ | ✅ | ✅ | jpg/png |
| `image_input` | ✅ | ✅ | ✅ | Up to 14 images (was fewer) |
| `resolution` | ✅ | ✅ | ✅ | NB2 adds 4K option |
| `safety_filter_level` | ✅ | ✅ | ✅ | Same |

NB2 is API-compatible with NB/NBP. No new required parameters. The main improvements are quality/speed, not API surface.

---

## Test Considerations

1. **Unit tests**: Existing test mocks return data URIs — same pattern works for NB2. Add NB2 to test inputs where models are iterated.
2. **Integration test**: Verify `google/nano-banana-2` model name resolves on Replicate.
3. **Manual test**: Generate an image with NB2 in each block to verify output quality and parameter handling.
4. **Edge cases**:
   - Editor block: NB2 with no input image (pure generation) — should work since `image_input` is optional
   - Editor block: NB2 ignores `seed` parameter — ensure it's not passed

---

## Implementation Order

1. `ai_image_customizer.py` — simplest, just add enum value + description update
2. `ai_image_generator_block.py` — add enum + copy/adapt NBP branch
3. `flux_kontext.py` — most complex, needs model-aware branching

Estimated effort: ~1 hour total.

---

## Verified

- **`GeminiImageModel` enum value format**: Existing entries use full API paths (`"google/nano-banana"`, `"google/nano-banana-pro"`). Plan's `NANO_BANANA_2 = "google/nano-banana-2"` matches this pattern. ✅
- **`SIZE_TO_NANO_BANANA_RATIO` coverage**: Mapping covers all 5 `ImageSize` enum values (SQUARE→1:1, LANDSCAPE→4:3, PORTRAIT→3:4, WIDE→16:9, TALL→9:16). ✅
- **`model_name` in `flux_kontext.py` `run_model()`**: Called via `model_name=input_data.model.api_name` (line ~125), so it receives the API name (e.g., `"google/nano-banana-2"`). The string check `"nano-banana" in model_name` is correct. ✅
- **Flux Kontext `input_image` optionality**: `input_image` field is `Optional[MediaFileType]` with `default=None`. The `run()` method already handles the None case with a conditional (line ~121). The existing Flux Kontext else-branch passes `input_image_b64` which may be None — this is fine as Flux Kontext accepts it. ✅
- **Nano Banana Pro in Editor block**: Added to plan per ticket requirement (sub-item 3.1). Both NBP and NB2 share the same API schema, so the same branching logic handles both. ✅
