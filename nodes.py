import numpy as np
from PIL import Image, ImageSequence, ImageOps
from PIL.PngImagePlugin import PngInfo # Added import
from sklearn.cluster import KMeans
from skimage import color
import torch
import comfy.utils
from concurrent.futures import ThreadPoolExecutor
import cv2
from typing import Dict, Tuple
import numba
import os
import random
import folder_paths
from nodes import PreviewImage
import io
import base64
import colorsys
import json
import math # Added for ceil/floor
import struct
import hashlib
import node_helpers

# Numba optimized functions
@numba.jit(nopython=True)
def _find_closest_color_numba(pixel_rgb, palette_rgb):
    """Numba-optimized function to find the closest color index in a palette."""
    diff = palette_rgb - pixel_rgb # Broadcasting
    distances_sq = np.sum(diff * diff, axis=1)
    return np.argmin(distances_sq)

@numba.jit(nopython=True)
def _apply_floyd_steinberg_dithering_numba(img_float, palette, palette_rgb, dithering_amount, has_alpha, transparent_idx):
    """Numba-optimized Floyd-Steinberg dithering."""
    height, width, channels = img_float.shape
    result = img_float.copy() # Work on a copy

    # Define error weights (constants are fine in nopython mode)
    err_weights_dx = np.array([1, -1, 0, 1], dtype=np.int32)
    err_weights_dy = np.array([0, 1, 1, 1], dtype=np.int32)
    err_weights_val = np.array([7/16, 3/16, 5/16, 1/16], dtype=np.float32) * dithering_amount # Apply amount here

    alpha_mask = np.empty((height, width), dtype=numba.boolean)
    if has_alpha:
        # Precompute alpha mask
        alpha_mask = result[:, :, 3] < 1.0
        # Set transparent pixels directly
        # transparent_idx != -1 implies palette is RGBA and has a defined transparent entry
        if transparent_idx != -1:
             transparent_palette_entry = palette[transparent_idx].astype(np.float32)
             for y in numba.prange(height): # Use prange for potential parallelization
                 for x in range(width):
                     if alpha_mask[y, x]:
                         # Since transparent_idx != -1, palette[transparent_idx] is RGBA.
                         # result is also RGBA because has_alpha is true.
                         result[y, x] = transparent_palette_entry
    else:
        # Create a dummy mask if no alpha
        alpha_mask = np.zeros((height, width), dtype=numba.boolean)


    for y in range(height):
        for x in range(width):
            # Skip transparent pixels already set
            if has_alpha and alpha_mask[y, x]:
                continue

            old_pixel = result[y, x].copy() # Copy necessary for error calc
            old_pixel_rgb = old_pixel[:3]

            # Find closest color in the RGB palette
            closest_idx = _find_closest_color_numba(old_pixel_rgb, palette_rgb)
            
            # Get the RGB values from palette_rgb (which is always N,3)
            quantized_rgb = palette_rgb[closest_idx]
            result[y, x, :3] = quantized_rgb # Assign new RGB

            # Handle alpha channel for the current pixel in result
            if has_alpha: # If original image has alpha channel
                if palette.shape[1] == 4: # If palette also has alpha channel
                    result[y, x, 3] = palette[closest_idx, 3] # Use alpha from palette
                # Else (image has alpha, but palette is RGB): 
                # alpha of result[y,x,3] retains its original value from img_float (result is a copy).
            # Else (original image is RGB): result has only 3 channels, nothing to do for alpha.

            # Calculate error only for RGB channels
            error_rgb = old_pixel_rgb - quantized_rgb # error_rgb is a 3-element array

            # Distribute the RGB error to RGB channels of neighbors
            for i in range(len(err_weights_dx)):
                nx, ny = x + err_weights_dx[i], y + err_weights_dy[i]

                # Check bounds
                if 0 <= nx < width and 0 <= ny < height:
                     # Don't distribute error to transparent pixels (those pre-filled or originally transparent)
                    if not (has_alpha and alpha_mask[ny, nx]):
                         result[ny, nx, :3] += error_rgb * err_weights_val[i]

    # Clip values at the end
    # Numba doesn't directly support np.clip with min/max arrays easily across dimensions
    # So we clip per channel (channels is from original img_float)
    for i in range(channels):
        result[:, :, i] = np.maximum(0.0, np.minimum(255.0, result[:, :, i]))

    return result.astype(np.uint8)

@numba.jit(nopython=True)
def _color_distance_numba(color1, color2, max_channels):
    """Numba-optimized Euclidean distance between colors (up to max_channels)."""
    dist_sq = 0.0
    for i in range(max_channels):
        diff = color1[i] - color2[i]
        dist_sq += diff * diff
    return np.sqrt(dist_sq)

@numba.jit(nopython=True)
def _flood_fill_core_numba(img, start_x, start_y, bg_color, max_color_diff, max_channels):
    """Numba-optimized core flood fill logic using a list as a queue."""
    h, w = img.shape[:2]
    # Prepare mask: 0=unprocessed, 1=background (filled), 2=processed but not filled
    mask = np.zeros((h, w), dtype=np.uint8)

    # Use a standard Python list for the queue - Numba supports simple list operations
    queue = [(start_x, start_y)] # Use list of tuples
    mask[start_y, start_x] = 1  # Mark start point as background

    # Define 4-connected neighbors (constants are fine)
    neighbors_dx = np.array([-1, 1, 0, 0], dtype=np.int32)
    neighbors_dy = np.array([0, 0, -1, 1], dtype=np.int32)

    processed_count = 0 # Keep track of queue additions to avoid excessive growth
    max_queue_size = h * w # Theoretical max

    queue_head = 0
    while queue_head < len(queue):
        x, y = queue[queue_head]
        queue_head += 1 # Dequeue by advancing head index (more efficient than pop(0))

        # No need to read current_color again, we know it matches bg_color

        for i in range(len(neighbors_dx)):
            nx, ny = x + neighbors_dx[i], y + neighbors_dy[i]

            # Check bounds
            if 0 <= nx < w and 0 <= ny < h:
                # Check if already processed
                if mask[ny, nx] == 0:
                    neighbor_color = img[ny, nx]
                    distance = _color_distance_numba(neighbor_color, bg_color, max_channels)

                    if distance <= max_color_diff:
                        mask[ny, nx] = 1  # Mark as background
                        queue.append((nx, ny)) # Enqueue
                    else:
                        mask[ny, nx] = 2 # Mark as processed but not background

    # Return only the background mask (where mask == 1)
    return (mask == 1).astype(np.uint8)  

class SmartImagePaletteConvert:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "num_colors": ("INT", {"default": 8, "min": 2, "max": 256, "step": 1}),
                "dithering_amount": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05, "display": "slider"}),
            },
            "optional": {
                "reference_image": ("IMAGE",),
                "additional_colors": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("IMAGE", "IMAGE")
    RETURN_NAMES = ("image", "palette")
    FUNCTION = "convert_image"
    CATEGORY = "SmartImageTools"

    @staticmethod
    def _sort_palette(palette):
        """Sorts a palette primarily by hue, then luminance, then saturation.
           Assigns special hue values for gray/transparent colors.
        """
        if palette.shape[0] <= 1:
            return palette

        has_alpha = palette.shape[1] == 4
        color_data = [] # List to store (sort_key, original_color)

        for color_val in palette:
            rgb = color_val[:3] / 255.0
            alpha = color_val[3] / 255.0 if has_alpha else 1.0
            is_transparent = has_alpha and color_val[3] == 0

            h, l, s = colorsys.rgb_to_hls(rgb[0], rgb[1], rgb[2])

            # Define hue_value for sorting
            # -2: Transparent
            # -1: Achromatic (Gray/Black/White)
            # 0-1: Chromatic hue
            if is_transparent:
                hue_value = -2
                # Use luminance of black for sorting transparent
                l = 0.0
                s = 0.0
            elif s < 0.05 or l < 0.01 or l > 0.99: # Achromatic threshold (low saturation or near black/white)
                hue_value = -1
                 # Keep actual luminance and saturation for sorting grays
            else:
                hue_value = h

            # Sort key: Hue -> Luminance -> Saturation
            sort_key = (hue_value, l, s)
            color_data.append((sort_key, color_val))

        # Sort based on the calculated key
        color_data.sort(key=lambda x: x[0])

        # Extract sorted colors
        sorted_palette_list = [item[1] for item in color_data]

        return np.array(sorted_palette_list, dtype=palette.dtype)

    @staticmethod
    def find_closest_color_vectorized(pixels, palette):
        """Find the closest color in the palette for each pixel (vectorized)."""
        # Reshape pixels to 2D if they aren't already
        original_shape = pixels.shape
        if len(original_shape) > 2:
            pixels_reshaped = pixels.reshape(-1, original_shape[-1])
        else:
            pixels_reshaped = pixels

        # Compute distances between each pixel and each palette color
        # Uses broadcasting for efficient computation
        diff = pixels_reshaped[:, np.newaxis, :3] - palette[np.newaxis, :, :3]
        distances = np.sum(diff**2, axis=2)
        indices = np.argmin(distances, axis=1)

        # Get the closest colors
        closest_colors = palette[indices]

        return closest_colors, indices

    def extract_palette_from_reference(self, reference_img, num_colors=None, has_transparency=True):
        """Extract palette from a reference image."""
        # Extract exact palette from reference image
        h, w = reference_img.shape[:2]
        has_alpha = reference_img.shape[2] == 4

        # Reshape the image to get all pixels
        img_flat = reference_img.reshape(-1, reference_img.shape[2])

        # For large images, sample a subset of pixels to avoid memory issues
        max_pixels = 100000
        if len(img_flat) > max_pixels:
            indices = np.random.choice(len(img_flat), max_pixels, replace=False)
            img_flat = img_flat[indices]

        # Handle alpha channel if present
        if has_alpha:
            # Separate transparent from non-transparent pixels
            mask = img_flat[:, 3] > 0
            rgb_opaque = img_flat[mask]

            # Get unique RGB colors (ignore alpha variations for same RGB)
            # Work with float representation internally for potential negative values
            unique_colors_float = np.unique(rgb_opaque[:, :3], axis=0).astype(np.float32)

            # Create final palette with alpha channel using float32
            palette = np.zeros((len(unique_colors_float) + 1, 4), dtype=np.float32)
            palette[:len(unique_colors_float), :3] = unique_colors_float
            palette[:len(unique_colors_float), 3] = 255.0  # Full opacity
            # Use large negative RGB for transparent color temporarily (this is needed to cancel the transparent color from picking it while searching for the closest color)
            palette[-1] = [-500.0, -500.0, -500.0, 0.0]
        else:
            # No alpha channel, just get unique RGB colors
            unique_colors = np.unique(img_flat, axis=0)
            # Convert to float32 for consistency
            palette = unique_colors.astype(np.float32)

        # Sorting works fine with floats
        sorted_palette_float = self._sort_palette(palette)
        # Return as uint8 after sorting and potential modification
        return np.clip(sorted_palette_float, 0, 255).astype(np.uint8)

    def merge_palettes(self, palette1, palette2):
        """Merge two palettes while avoiding duplicates."""
        # Ensure both palettes are float32 for processing
        palette1_float = palette1.astype(np.float32)
        palette2_float = palette2.astype(np.float32)

        # Determine if we need to handle alpha channels
        has_alpha1 = palette1_float.shape[1] == 4
        has_alpha2 = palette2_float.shape[1] == 4

        if has_alpha1 and has_alpha2:
            # Both have alpha - merge properly
            # Combine all colors
            combined = np.vstack([palette1_float, palette2_float])

            # Find unique colors based on RGB values (ignore alpha for uniqueness)
            rgb_colors = combined[:, :3]

            # Create a set to track unique RGB combinations
            unique_indices = []
            seen_rgb = set()

            for i, rgb in enumerate(rgb_colors):
                rgb_tuple = tuple(rgb)
                if rgb_tuple not in seen_rgb:
                    seen_rgb.add(rgb_tuple)
                    unique_indices.append(i)

            # Extract unique colors
            unique_palette = combined[unique_indices]

        elif has_alpha1 or has_alpha2:
            # One has alpha, the other doesn't - convert both to RGBA
            if not has_alpha1:
                # Add alpha channel to palette1
                alpha_col = np.full((palette1_float.shape[0], 1), 255.0, dtype=np.float32)
                palette1_float = np.hstack([palette1_float, alpha_col])

            if not has_alpha2:
                # Add alpha channel to palette2
                alpha_col = np.full((palette2_float.shape[0], 1), 255.0, dtype=np.float32)
                palette2_float = np.hstack([palette2_float, alpha_col])

            # Now both have alpha, use the same logic as above
            combined = np.vstack([palette1_float, palette2_float])
            rgb_colors = combined[:, :3]

            unique_indices = []
            seen_rgb = set()

            for i, rgb in enumerate(rgb_colors):
                rgb_tuple = tuple(rgb)
                if rgb_tuple not in seen_rgb:
                    seen_rgb.add(rgb_tuple)
                    unique_indices.append(i)

            unique_palette = combined[unique_indices]

        else:
            # Neither has alpha - simple RGB merging
            combined = np.vstack([palette1_float, palette2_float])
            unique_palette = np.unique(combined, axis=0)

        return unique_palette

    def generate_palette(self, img, num_colors, has_transparency=True):
        """Generate an optimal color palette using K-means clustering in CIELAB space."""
        # Downsample large images for faster processing
        h, w = img.shape[:2]
        max_dim = 256
        if max(h, w) > max_dim:
            scale = max_dim / max(h, w)
            new_size = (int(w * scale), int(h * scale))
            img_small = cv2.resize(img, new_size, interpolation=cv2.INTER_AREA)
        else:
            img_small = img

        # Convert to CIELAB for better perceptual clustering
        img_flat = img_small.reshape(-1, 4) if img_small.shape[2] == 4 else img_small.reshape(-1, 3)

        # Separate alpha channel if present
        if img_small.shape[2] == 4:
            rgb = img_flat[:, :3]
            alpha = img_flat[:, 3]
            # Only consider non-transparent pixels for color extraction
            rgb = rgb[alpha > 0]
        else:
            rgb = img_flat

        if len(rgb) == 0:
            # Handle completely transparent images
            if has_transparency:
                if img.shape[2] == 4:
                    return np.zeros((1, 4), dtype=np.uint8)
                else:
                    return np.zeros((1, 3), dtype=np.uint8)

        # Use a subset of pixels for large images
        max_pixels = 10000
        if len(rgb) > max_pixels:
            indices = np.random.choice(len(rgb), max_pixels, replace=False)
            rgb_subset = rgb[indices]
        else:
            rgb_subset = rgb

        # Convert RGB to CIELAB
        rgb_normalized = rgb_subset / 255.0
        lab = color.rgb2lab(rgb_normalized.reshape(-1, 1, 3)).reshape(-1, 3)

        # Use KMeans to cluster colors
        colors_to_extract = num_colors - 1 if has_transparency else num_colors
        kmeans = KMeans(n_clusters=colors_to_extract, random_state=42, n_init=10)
        kmeans.fit(lab)

        # Convert centers back to RGB
        centers_lab = kmeans.cluster_centers_
        centers_rgb = color.lab2rgb(centers_lab.reshape(-1, 1, 3)).reshape(-1, 3)
        centers_rgb = (centers_rgb * 255).astype(np.uint8)

        # Use float32 for palette to accommodate potential negative values
        centers_rgb_float = centers_rgb.astype(np.float32)

        # Add transparency color if needed
        if has_transparency:
            # Use large negative RGB for transparent color temporarily, use float32
            transparent_color = np.array([-99999.0, -99999.0, -99999.0, 0.0], dtype=np.float32)
            if img.shape[2] == 4:
                # Create palette with RGBA values using float32
                palette = np.zeros((colors_to_extract + 1, 4), dtype=np.float32)
                palette[:-1, :3] = centers_rgb_float
                palette[:-1, 3] = 255.0  # Full opacity for color entries
                palette[-1] = transparent_color  # Last entry is transparent (with modified RGB)
            else:
                # Create palette with RGB values using float32
                palette = np.zeros((colors_to_extract + 1, 3), dtype=np.float32)
                palette[:-1] = centers_rgb_float
                palette[-1] = transparent_color[:3] # RGB part of modified transparent
        else:
            # No transparency needed
            if img.shape[2] == 4:
                # Use float32
                palette = np.zeros((colors_to_extract, 4), dtype=np.float32)
                palette[:, :3] = centers_rgb_float
                palette[:, 3] = 255.0
            else:
                # Use float32
                palette = centers_rgb_float

        # Sorting works fine with floats
        sorted_palette_float = self._sort_palette(palette)
        # Return palette as uint8 after processing is complete in convert_image
        # Keep it as float32 internally for dithering/mapping
        return sorted_palette_float # Return float palette

    def apply_floyd_steinberg_dithering(self, img, palette, dithering_amount=1.0):
        """Apply Floyd-Steinberg dithering with variable amount and optimizations."""
        # Ensure palette is float32 for internal processing
        palette_float = palette.astype(np.float32)

        if dithering_amount <= 0:
            # If no dithering, just do direct color mapping (already vectorized)
            has_alpha = img.shape[2] == 4
            img_flat = img.reshape(-1, img.shape[2])

            # Determine the transparent color index from the palette
            transparent_idx = -1
            if has_alpha: # img has alpha
                # Check if palette ALSO has alpha before trying to access its alpha channel
                if palette_float.shape[1] == 4:
                    alpha_channel = palette_float[:, 3]
                    transparent_indices = np.where(alpha_channel == 0.0)[0]
                    if len(transparent_indices) > 0:
                        transparent_idx = transparent_indices[0] # Take the first one
                # else: palette is RGB, so no transparent_idx to find in it.
            # Use palette RGB for distance calculation (will include -99999 if present)
            palette_rgb = palette_float[:, :3]

            if has_alpha:
                # Handle transparent pixels in the source image
                mask = img_flat[:, 3] < 1 # Check alpha channel of source image

                # Find closest color for non-transparent pixels using palette_rgb
                closest_colors_indices = np.zeros(img_flat.shape[0], dtype=int) # Placeholder shape
                non_transparent_mask = ~mask
                if np.any(non_transparent_mask):
                     closest_colors_non_transparent, indices_non_transparent = SmartImagePaletteConvert.find_closest_color_vectorized(
                         img_flat[non_transparent_mask, :3], palette_rgb
                     )
                     closest_colors_indices[non_transparent_mask] = indices_non_transparent

                # Create result array (needs to be float initially if palette is float)
                result_flat = np.zeros_like(img_flat, dtype=np.float32)

                # Map non-transparent pixels using the full palette (incl alpha and modified RGB)
                if np.any(non_transparent_mask):
                     result_flat[non_transparent_mask] = palette_float[closest_colors_indices[non_transparent_mask]]

                # Set transparent pixels to the palette's transparent color (using index)
                if transparent_idx != -1 and np.any(mask):
                    result_flat[mask] = palette_float[transparent_idx]
                elif np.any(mask): # If transparent pixels exist but no transparent color in palette
                     # Fallback: Make them black with 0 alpha? Or handle error?
                     # Let's make them black and transparent
                     result_flat[mask, :3] = 0.0
                     result_flat[mask, 3] = 0.0


            else:
                # No transparency - simple mapping
                closest_colors, indices = SmartImagePaletteConvert.find_closest_color_vectorized(img_flat, palette_rgb)
                result_flat = palette_float[indices]


            # Reshape and convert to uint8 at the end
            return np.clip(result_flat, 0, 255).reshape(img.shape).astype(np.uint8)


        # --- Use Numba for dithering > 0 ---
        has_alpha = img.shape[2] == 4
        img_float = img.astype(np.float32)

        # Create a palette view without alpha for color comparisons if needed
        # Numba needs contiguous arrays sometimes
        palette_rgb = np.ascontiguousarray(palette_float[:, :3])

        # Pre-calculate transparent index if needed
        transparent_idx = -1
        if has_alpha:
             # Check if palette ALSO has alpha
             if palette_float.shape[1] == 4:
                 alpha_channel = palette_float[:, 3]
                 transparent_indices = np.where(alpha_channel == 0.0)[0]
                 if len(transparent_indices) > 0:
                     transparent_idx = transparent_indices[0]
             # else: # Handle case where no transparent color found if necessary
             #     print("Warning: No transparent color (alpha=0) found in palette for dithering.")


        # Call the Numba JIT function (expects float32 palette)
        result_dithered = _apply_floyd_steinberg_dithering_numba(
            img_float,
            palette_float, # Pass full float palette (with modified RGB, correct alpha)
            palette_rgb,   # Pass float RGB part (with modified RGB) for distance calc
            dithering_amount,
            has_alpha,
            transparent_idx
        )
        # Numba function already returns uint8
        return result_dithered

    def process_image(self, image, palette, dithering_amount):
        """Process a single image with the given palette."""
        # Palette is expected to be float32 here
        return self.apply_floyd_steinberg_dithering(image, palette, dithering_amount)

    def convert_image(self, image, num_colors, dithering_amount, reference_image=None, additional_colors=None):
        # Convert from tensor to numpy array
        input_image = 255. * image.cpu().numpy()

        # Process reference image if provided
        if reference_image is not None:
            reference_np = 255. * reference_image.cpu().numpy()
            # Extract exact palette from reference image, ignoring num_colors
            # This will return uint8 initially
            palette_uint8 = self.extract_palette_from_reference(reference_np[0])
            # Convert to float32 for internal processing
            palette_float = palette_uint8.astype(np.float32)
             # Ensure transparent color has modified RGB if present
            if palette_float.shape[1] == 4: # Check if alpha channel exists
                alpha_channel = palette_float[:, 3]
                transparent_indices = np.where(alpha_channel == 0.0)[0]
                if len(transparent_indices) > 0:
                    palette_float[transparent_indices, :3] = -99999.0
        else:
            # Generate optimal palette from the input image using num_colors
            # Returns float32 palette with modified transparent color already
            palette_float = self.generate_palette(input_image[0], num_colors)

        # Process additional colors if provided
        if additional_colors is not None:
            additional_np = 255. * additional_colors.cpu().numpy()
            # Extract exact palette from additional colors image
            additional_palette_uint8 = self.extract_palette_from_reference(additional_np[0])
            # Convert to float32 for processing
            additional_palette_float = additional_palette_uint8.astype(np.float32)

            # Ensure transparent color has modified RGB if present in additional palette
            if additional_palette_float.shape[1] == 4:
                alpha_channel = additional_palette_float[:, 3]
                transparent_indices = np.where(alpha_channel == 0.0)[0]
                if len(transparent_indices) > 0:
                    additional_palette_float[transparent_indices, :3] = -99999.0

            # Merge the palettes while avoiding duplicates
            palette_float = self.merge_palettes(palette_float, additional_palette_float)


        # Create output array with same batch size as input
        batch_size = input_image.shape[0]

        # Use ThreadPoolExecutor for parallel processing of batch images
        with ThreadPoolExecutor() as executor:
            futures = [
                # Pass the float palette to process_image
                executor.submit(self.process_image, input_image[i], palette_float, dithering_amount)
                for i in range(batch_size)
            ]

            output_images = [future.result() for future in futures]

        # Convert back to tensor
        output_tensor = torch.from_numpy(np.stack(output_images) / 255.0).float()

        # --- Restore transparent color RGB in the palette before output ---
        # Work on a copy to not affect potential future uses if palette is cached etc.
        final_palette = palette_float.copy()
        if final_palette.shape[1] == 4: # Check if alpha channel exists
            alpha_channel = final_palette[:, 3]
            transparent_indices = np.where(alpha_channel == 0.0)[0]
            if len(transparent_indices) > 0:
                 # Set RGB of transparent colors back to 0
                 final_palette[transparent_indices, :3] = 0.0

        # Clip and convert the final palette to uint8 for the output tensor
        final_palette_uint8 = np.clip(final_palette, 0, 255).astype(np.uint8)
        # --- End Restoration ---

        # Create palette image (1 pixel high, with each color in the **sorted** palette)
        palette_height = 1
        palette_width = len(final_palette_uint8)
        # Create a NumPy array for the palette image using the restored uint8 palette
        palette_img = np.zeros((palette_height, palette_width, final_palette_uint8.shape[1]), dtype=np.uint8)

        # Fill each column with a color from the final uint8 palette
        if palette_width > 0:
            for i in range(palette_width):
                palette_img[:, i] = final_palette_uint8[i]

        # Convert to tensor (add batch dimension and normalize to 0-1 range)
        palette_tensor = torch.from_numpy(palette_img[np.newaxis, ...] / 255.0).float()

        return (output_tensor, palette_tensor)


class SmartImagesProcessor:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE",),
                "cut_start": ("INT", {"default": 0, "min": 0, "step": 1}),
                "cut_end": ("INT", {"default": 0, "min": 0, "step": 1}),
                "fps_reduction_factor": ("INT", {"default": 1, "min": 1, "step": 1}),
            },
            "optional": {
                # Keep type FLOAT, remove widget configuration to force dot input
                "fps": ("FLOAT", ),
            }
        }

    # Added IMAGE outputs for first and last frames
    RETURN_TYPES = ("IMAGE", "FLOAT", "IMAGE", "IMAGE")
    RETURN_NAMES = ("images", "fps", "first_frame", "last_frame") # Added names
    FUNCTION = "process_images"
    CATEGORY = "SmartImageTools"

    # Default fps to None to detect if it was connected
    def process_images(self, images, cut_start, cut_end, fps_reduction_factor, fps=None):
        # If fps is not connected or is None, default to 0.0 for calculations
        input_fps = fps if fps is not None else 0.0
        batch_size = images.shape[0]
        original_dtype = images.dtype
        device = images.device
        empty_frame_shape = (0,) + images.shape[1:] # Shape for empty frame tensor

        # Ensure cut values are not negative
        start_frame = max(0, cut_start)
        end_frame_offset = max(0, cut_end)

        # Calculate the slice indices for cutting, clamping to valid range
        start_idx = min(start_frame, batch_size)
        end_idx = max(start_idx, batch_size - end_frame_offset)
        end_idx = min(end_idx, batch_size) # Ensure end_idx doesn't exceed batch_size

        # Initialize output_fps as float 0.0, preserving floating-point precision
        output_fps = 0.0
        if input_fps > 0:
            # If a valid FPS is input, use it directly (keep as float)
            output_fps = input_fps

        # Create empty tensors for outputs initially
        empty_frames = torch.empty(empty_frame_shape, dtype=original_dtype, device=device)
        empty_first_frame = torch.empty(empty_frame_shape, dtype=original_dtype, device=device)
        empty_last_frame = torch.empty(empty_frame_shape, dtype=original_dtype, device=device)

        if start_idx >= end_idx:
            # Handle cases where the cuts result in no frames left
            # Return an empty tensor with the correct shape dimensions except batch
            print(f"Warning: Cut parameters ({cut_start}, {cut_end}) resulted in zero frames for batch size {batch_size}. Returning empty tensors.")
            # Return empty tensor and original/default (potentially rounded) fps
            # return (torch.empty(empty_frame_shape, dtype=original_dtype, device=device), output_fps)
            return (empty_frames, output_fps, empty_first_frame, empty_last_frame)

        # Apply start/end cut first
        trimmed_images = images[start_idx:end_idx]

        # Apply FPS reduction factor
        if fps_reduction_factor > 1 and trimmed_images.shape[0] > 0:
             processed_images = trimmed_images[::fps_reduction_factor]
             # Calculate new FPS if original FPS was provided (>0)
             if input_fps > 0:
                 # Use float division, ensure at least 1.0, keep as float
                 output_fps = max(1.0, input_fps / fps_reduction_factor)
             # If original fps was 0 or less, keep output_fps as 0.0

             if processed_images.shape[0] == 0:
                 # Handle cases where reduction factor removes all remaining frames
                 print(f"Warning: FPS reduction factor ({fps_reduction_factor}) resulted in zero frames after trimming. Returning empty tensors.")
                 # Return empty tensor and calculated/original fps
                 # return (torch.empty(empty_frame_shape, dtype=original_dtype, device=device), output_fps)
                 return (empty_frames, output_fps, empty_first_frame, empty_last_frame)
        else:
             processed_images = trimmed_images # No reduction needed or possible
             # output_fps remains the same as input fps (potentially rounded initially)

        # Extract first and last frames if images exist
        if processed_images.shape[0] > 0:
            first_frame = processed_images[0:1] # Slice to keep batch dimension
            last_frame = processed_images[-1:]  # Slice to keep batch dimension
        else:
            # This case should technically be caught earlier, but as a fallback
            first_frame = empty_first_frame
            last_frame = empty_last_frame

        # Return output_fps as float
        return (processed_images, output_fps, first_frame, last_frame)


class SmartGenerateImage:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "color": (["red", "green", "blue", "black", "white", "yellow", "cyan", "magenta",
                           "gray", "dark gray", "orange", "purple", "brown", "pink", "transparent"],),
                "alpha": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "use_alpha": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "width": ("INT", {"default": 512, "min": 1, "max": 8192, "step": 1}),
                "height": ("INT", {"default": 512, "min": 1, "max": 8192, "step": 1}),
                "get_image_size": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "generate_image"
    CATEGORY = "SmartImageTools"

    def generate_image(self, width=512, height=512, color="white", alpha=1.0, use_alpha=True, get_image_size=None):
        # Define color map with RGB(A) values
        color_map = {
            "red": [255, 0, 0],
            "green": [0, 255, 0],
            "blue": [0, 0, 255],
            "black": [0, 0, 0],
            "white": [255, 255, 255],
            "yellow": [255, 255, 0],
            "cyan": [0, 255, 255],
            "magenta": [255, 0, 255],
            "gray": [128, 128, 128],
            "dark gray": [64, 64, 64],
            "orange": [255, 165, 0],
            "purple": [128, 0, 128],
            "brown": [165, 42, 42],
            "pink": [255, 192, 203],
            "transparent": [0, 0, 0, 0],  # Special case with alpha=0
        }

        # Get dimensions from input image if provided
        if get_image_size is not None:
            height, width = get_image_size.shape[1:3]

        # Create image with or without alpha channel
        channels = 4 if use_alpha else 3

        # Handle special case for transparent color
        if color == "transparent":
            # Force alpha=0 for transparent color
            alpha_value = 0
            rgb_value = color_map["transparent"][:3]
        else:
            # Use the specified alpha
            alpha_value = alpha
            rgb_value = color_map[color]

        # Create the image array
        if channels == 4:
            # With alpha channel
            img_array = np.zeros((1, height, width, channels), dtype=np.float32)
            img_array[0, :, :, :3] = np.array(rgb_value) / 255.0
            img_array[0, :, :, 3] = alpha_value
        else:
            # RGB only
            img_array = np.zeros((1, height, width, channels), dtype=np.float32)
            img_array[0, :, :, :] = np.array(rgb_value) / 255.0

        # Convert to tensor
        tensor = torch.from_numpy(img_array).float()

        return (tensor,)


class SmartBackgroundRemove:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "fill_size": ("INT", {"default": 1, "min": 1, "max": 100, "step": 1}),
                "tolerance": ("FLOAT", {"default": 0.01, "min": 0.0, "max": 1.0, "step": 0.01}),
                "each_point_own_color": ("BOOLEAN", {"default": True}),
                "debug_points": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "start_point": ("POINT",),
                "start_points": ("POINT_SET",),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "flood_remove"
    CATEGORY = "SmartImageTools"

    def color_distance(self, color1, color2, max_channels=3):
        """Calculate Euclidean distance between colors"""
        return np.sqrt(np.sum((color1[:max_channels] - color2[:max_channels])**2))

    def custom_flood_fill(self, img, start_coords, bg_color, tolerance, fill_size, edge_mask=None):
        """
        Custom flood fill implementation using Numba for core logic and OpenCV for morphology.
        Uses a specific background color.
        """
        h, w = img.shape[:2]
        channels = img.shape[2]
        max_channels = min(3, channels) # Consider at most 3 channels (RGB) for distance

        # Ensure input image is contiguous uint8 for Numba compatibility
        img_uint8 = img.astype(np.uint8)
        if not img_uint8.flags['C_CONTIGUOUS']:
            img_uint8 = np.ascontiguousarray(img_uint8)

        # Convert tolerance from 0-1 to actual color distance
        # Use float32 for Numba compatibility
        max_color_diff = np.float32(tolerance * 255.0 * np.sqrt(max_channels))
        bg_color_float = bg_color.astype(np.float32) # Numba prefers consistent types
        start_x, start_y = start_coords

        # Call the Numba-optimized core flood fill logic
        background_mask = _flood_fill_core_numba(
            img_uint8,
            start_x,
            start_y,
            bg_color_float,
            max_color_diff,
            max_channels
        )

        # --- Apply morphological operations outside Numba ---
        if fill_size > 1:
            # Ensure edge_mask exists if fill_size > 1 (should be handled by caller)
            if edge_mask is None:
                 edge_mask = np.zeros((h, w), dtype=np.uint8)
                 edge_mask[:fill_size, :] = 1; edge_mask[-fill_size:, :] = 1
                 edge_mask[:, :fill_size] = 1; edge_mask[:, -fill_size:] = 1

            # Create a kernel for morphological operations
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (fill_size, fill_size))

            # Apply morphological closing to foreground (invert mask)
            foreground_mask = 1 - background_mask # background_mask is already uint8
            closed_foreground = cv2.morphologyEx(foreground_mask, cv2.MORPH_CLOSE, kernel)

            # Calculate new background
            new_background = 1 - closed_foreground

            # Apply edge mask logic
            edge_background = background_mask * edge_mask # Original fill in edge areas
            non_edge_background = new_background * (1 - edge_mask) # Closed fill elsewhere
            # Combine and ensure uint8
            background_mask = np.clip(edge_background + non_edge_background, 0, 1).astype(np.uint8)


        return background_mask # Return the final uint8 background mask

    def flood_remove(self, image, fill_size, tolerance, each_point_own_color, debug_points, start_point=None, start_points=None):
        # Convert from tensor to numpy array
        input_image = 255. * image.cpu().numpy()
        batch_size = input_image.shape[0]
        output_images = []

        # Determine which points to use
        points_to_use = []
        if start_points is not None:
            points_to_use = start_points
        elif start_point is not None:
            points_to_use = [start_point]
        else:
            # Default if no points provided
            points_to_use = [[0, 0]]

        for i in range(batch_size):
            img = input_image[i].copy()
            h, w = img.shape[:2]
            
            # Create initial mask: 1 for foreground (keep), 0 for background (remove)
            mask = np.ones((h, w), dtype=np.uint8)
            
            # Create edge mask for infinite corridor size near edges (shared by all points)
            if fill_size > 1:
                edge_mask = np.zeros((h, w), dtype=np.uint8)
                edge_mask[:fill_size, :] = 1  # Top edge
                edge_mask[-fill_size:, :] = 1  # Bottom edge
                edge_mask[:, :fill_size] = 1  # Left edge
                edge_mask[:, -fill_size:] = 1  # Right edge
            else:
                edge_mask = None
            
            # Sample bg color from first point if not using individual colors
            first_bg_color = None
            if not each_point_own_color and len(points_to_use) > 0:
                # Convert first point from normalized (0-1) to pixel coordinates
                first_x = int(points_to_use[0][0] * w)
                first_y = int((1 - points_to_use[0][1]) * h)
                
                # Clamp coordinates to valid range
                first_x = max(0, min(first_x, w - 1))
                first_y = max(0, min(first_y, h - 1))
                
                # Get background color from first point
                first_bg_color = img[first_y, first_x].copy()
            
            # Process each point
            for point_idx, point in enumerate(points_to_use):
                # Convert point from normalized (0-1) to pixel coordinates
                x = int(point[0] * w)
                y = int((1 - point[1]) * h)  # Invert y as flood fill works from top-left
                
                # Clamp coordinates to valid range
                x = max(0, min(x, w - 1))
                y = max(0, min(y, h - 1))
                
                # Get background color from point or use first point's color
                if each_point_own_color or point_idx == 0:
                    bg_color = img[y, x].copy()
                else:
                    bg_color = first_bg_color
                
                # Use custom flood fill with the sampled background color
                fill_mask = self.custom_flood_fill(
                    img, 
                    (x, y), 
                    bg_color, 
                    tolerance, 
                    fill_size, 
                    edge_mask
                )
                
                # Update the mask - any area filled should be considered background
                # We use logical_and to keep track of areas that haven't been filled
                mask = np.logical_and(mask, 1 - fill_mask).astype(np.uint8)
            
            # Apply mask to image
            if img.shape[2] == 4:  # RGBA
                # Just modify alpha channel
                result = img.copy()
                result[:, :, 3] = result[:, :, 3] * mask
            else:  # RGB
                # For RGB, we need to add alpha channel
                result = np.zeros((h, w, 4), dtype=np.uint8)
                result[:, :, :3] = img
                result[:, :, 3] = mask * 255
            
            # Draw debug points if requested
            if debug_points and points_to_use:
                # Ensure result is uint8 for OpenCV drawing functions
                result_uint8 = result.astype(np.uint8)
                
                # Draw each point
                for point in points_to_use:
                    # Convert point from normalized (0-1) to pixel coordinates
                    px = int(point[0] * w)
                    py = int((1 - point[1]) * h)  # Invert y-coordinate
                    
                    # Clamp coordinates to valid range
                    px = max(0, min(px, w - 1))
                    py = max(0, min(py, h - 1))
                    
                    # Draw black outline first (3px radius)
                    cv2.circle(result_uint8, (px, py), 3, (0, 0, 0, 255), -1)
                    # Draw red point on top (2px radius)
                    cv2.circle(result_uint8, (px, py), 2, (255, 0, 0, 255), -1)
                
                # Convert back to float32 for output
                result = result_uint8.astype(np.float32)
            
            output_images.append(result)
        
        # Convert back to tensor
        output_tensor = torch.from_numpy(np.stack(output_images) / 255.0).float()
        
        return (output_tensor,)     


class SmartPoint:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "x": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "y": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("POINT",)
    FUNCTION = "create_point"
    CATEGORY = "SmartImageTools"

    def create_point(self, x, y):
        return ([x, y],)


class SmartPointSet:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "point": ("POINT",),
            }
        }

    RETURN_TYPES = ("POINT_SET",)
    FUNCTION = "create_point_set"
    CATEGORY = "SmartImageTools"

    def create_point_set(self, point):
        # Create a list containing just this point
        return ([point],)


class SmartPointSetMerge:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "point_set": ("POINT_SET",),
                "point": ("POINT",),
            }
        }

    RETURN_TYPES = ("POINT_SET",)
    FUNCTION = "merge_point"
    CATEGORY = "SmartImageTools"

    def merge_point(self, point_set, point):
        # Create a new list with all existing points plus the new one
        new_set = point_set.copy()
        new_set.append(point)
        return (new_set,)


class SmartImagePreviewScaled(PreviewImage):
    def __init__(self):
        super().__init__()

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE",),
                # Change to FLOAT widget
                "scale_by": ("FLOAT", {"default": 1.0, "min": 0.01, "max": 100.0, "step": 0.1}),
                "transparency": (["original", "black", "white", "orange"], {"default": "original"}),
            },
            "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
        }

    RETURN_TYPES = ()
    FUNCTION = "execute" # Use execute which then calls save_images internally
    OUTPUT_NODE = True
    CATEGORY = "SmartImageTools"

    # Use execute to potentially modify images before calling the parent's save logic
    def execute(self, images, scale_by=1.0, transparency="original", filename_prefix="SmartScaledPreview", prompt=None, extra_pnginfo=None):
        # Store original dimensions before any scaling
        original_height = images.shape[1]
        original_width = images.shape[2]
        
        if abs(scale_by - 1.0) < 1e-6 or scale_by <= 0:
            # No scaling or invalid scale factor, use original save_images
            result = self.save_images(images, filename_prefix, prompt, extra_pnginfo)
            # Add original dimensions to the result
            if "ui" in result:
                result["ui"]["original_dimensions"] = [[original_width, original_height]]
            return result

        # Perform scaling
        scaled_images_list = []
        original_dtype = images.dtype
        device = images.device

        for img_tensor in images: # Process each image in the batch
            # Convert tensor (B, H, W, C) or (H, W, C) -> numpy (H, W, C)
            img_np = img_tensor.cpu().numpy()

            # Ensure it's float 0-1 range before converting to uint8
            img_np = np.clip(img_np, 0.0, 1.0)
            img_np_uint8 = (img_np * 255.0).astype(np.uint8)

            h, w = img_np_uint8.shape[:2]
            new_h = max(1, int(round(h * scale_by)))
            new_w = max(1, int(round(w * scale_by)))

            if h == 0 or w == 0:
                # Handle empty image case - create an empty image of target size
                resized_img = np.zeros((new_h, new_w, img_np_uint8.shape[2]), dtype=np.uint8)
            else:
                # Resize using nearest neighbor
                resized_img = cv2.resize(img_np_uint8, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

            # --- Transparency Blending ---
            if transparency != "original":
                # Convert to PIL for alpha compositing
                resized_pil = Image.fromarray(resized_img)

                # Ensure image is RGBA for compositing
                if resized_pil.mode != 'RGBA':
                    resized_pil = resized_pil.convert('RGBA')

                # Get background color from BG_COLOR_MAP
                if transparency in BG_COLOR_MAP and BG_COLOR_MAP[transparency] is not None:
                    blend_rgb = BG_COLOR_MAP[transparency]
                    # Create a background image of the solid color
                    background = Image.new('RGBA', resized_pil.size, blend_rgb + (255,))

                    # Composite the original image over the background
                    # This automatically handles the alpha blending
                    resized_pil = Image.alpha_composite(background, resized_pil)

                # Convert back to numpy array
                resized_img = np.array(resized_pil)
            # --- End Transparency Blending ---

            # Handle potential shape issues after resize (e.g., grayscale losing channel dim)
            if len(resized_img.shape) == 2 and len(img_np_uint8.shape) == 3:
                resized_img = np.expand_dims(resized_img, axis=-1) # Add channel dim back
                # If original had more channels (e.g., RGB), replicate the channel
                if img_np_uint8.shape[2] > 1:
                     resized_img = resized_img.repeat(img_np_uint8.shape[2], axis=-1)
            elif len(resized_img.shape) == 3 and resized_img.shape[2] != img_np_uint8.shape[2]:
                 # If channel count changed unexpectedly, try to force match (e.g., add alpha or take RGB)
                 target_channels = img_np_uint8.shape[2]
                 if target_channels == 4 and resized_img.shape[2] == 3: # Add alpha
                     alpha_channel = np.full((new_h, new_w, 1), 255, dtype=np.uint8)
                     resized_img = np.concatenate((resized_img, alpha_channel), axis=-1)
                 elif target_channels == 3 and resized_img.shape[2] == 4: # Drop alpha
                     resized_img = resized_img[:, :, :3]
                 # Add other potential conversions if needed, otherwise, there might be an issue

            # Convert back to float32 tensor [0, 1]
            resized_img_float = resized_img.astype(np.float32) / 255.0
            # Ensure tensor shape is HWC before converting
            scaled_images_list.append(torch.from_numpy(resized_img_float).to(device).to(original_dtype))

        # Stack the list of tensors back into a batch (B, H, W, C)
        if not scaled_images_list: # Handle case where input images might be empty
            return {"ui": {"images": []}} # Return empty preview data

        scaled_images_batch = torch.stack(scaled_images_list)

        # Call the parent's save_images method with the scaled images
        result = self.save_images(scaled_images_batch, filename_prefix, prompt, extra_pnginfo)
        # Add original dimensions to the result
        if "ui" in result:
            result["ui"]["original_dimensions"] = [[original_width, original_height]]
        return result


class SmartPreviewPalette:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": { "palette_image": ("IMAGE",) },
            "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
        }

    RETURN_TYPES = ()
    FUNCTION = "execute"
    OUTPUT_NODE = True
    CATEGORY = "SmartImageTools"

    def execute(self, palette_image, prompt=None, extra_pnginfo=None):
        colors_list = []
        if palette_image is not None and palette_image.shape[0] > 0:
            # Assuming BxHxWxC, and H=1
            img_np = palette_image[0].cpu().numpy()
            if img_np.shape[0] == 1: # Check if height is 1
                # Iterate through the width
                for w in range(img_np.shape[1]):
                    pixel = img_np[0, w, :] # Get pixel data (H=0, W=w)
                    # Convert float [0,1] to int [0,255]
                    rgb = (np.clip(pixel[:3], 0.0, 1.0) * 255).astype(np.uint8)
                    # Format as hex string #RRGGBB
                    hex_color = "#{:02x}{:02x}{:02x}".format(rgb[0], rgb[1], rgb[2])
                    colors_list.append(hex_color)

        return {"ui": {"colors": [colors_list]}} # Wrap in another list to match PreviewImage structure


class SmartImagePoint:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                # Hidden input to store dot positions as JSON string
                "points_data": ("STRING", {"default": '[{"x":0.5,"y":0.5}]', "multiline": False, "hidden": True}),
            },
            "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
        }

    # Define outputs for the workflow graph
    RETURN_TYPES = ("POINT", "POINT_SET")
    RETURN_NAMES = ("first_point", "points")
    FUNCTION = "get_points_and_update_ui"
    OUTPUT_NODE = True
    CATEGORY = "SmartImageTools"

    # Instance method to access properties
    def get_points_and_update_ui(self, image, points_data, prompt=None, extra_pnginfo=None):
        # 1. Prepare UI data (image preview)
        img_tensor = image[0]
        h, w = img_tensor.shape[0:2]
        img_np = np.clip(img_tensor.cpu().numpy(), 0.0, 1.0)
        img_uint8 = (img_np * 255.0).astype(np.uint8)
        pil_img = Image.fromarray(img_uint8)
        buffer = io.BytesIO()
        pil_img.save(buffer, format="PNG")
        buffer.seek(0)
        img_base64 = base64.b64encode(buffer.read()).decode('utf-8')
        ui_data = {
            "images": [f"data:image/png;base64,{img_base64}"],
            "dimensions": [[w, h]],
            "points_data": points_data  # Send points_data to JS
        }

        # 2. Parse points_data JSON
        try:
            dots = json.loads(points_data)
            if not isinstance(dots, list):
                dots = [{'x': 0.5, 'y': 0.5}]
        except Exception as e:
            print(f"Error parsing points_data: {e}")
            dots = [{'x': 0.5, 'y': 0.5}]

        # Ensure dots have x and y
        valid_dots = []
        for dot in dots:
            if isinstance(dot, dict) and 'x' in dot and 'y' in dot:
                valid_dots.append(dot)
            else:
                # Add default if an invalid dot is found
                valid_dots.append({'x': 0.5, 'y': 0.5})

        if not valid_dots: # Ensure we always have at least one dot
            valid_dots = [{'x': 0.5, 'y': 0.5}]

        # 3. Format outputs
        first_dot = valid_dots[0]
        first_point = [first_dot.get('x', 0.5), first_dot.get('y', 0.5)]

        point_set = [[d.get('x', 0.5), d.get('y', 0.5)] for d in valid_dots]

        # Return results for the workflow graph AND data for the UI
        return {"result": (first_point, point_set), "ui": ui_data}


class SmartImageRegion:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "type": (["Rectangle", "Polygon"],), # Added Polygon type
                # Hidden input to store region data as JSON string
                "region_data": ("STRING", {"default": '{"x1":0.25,"y1":0.25,"x2":0.75,"y2":0.75}', "multiline": False, "hidden": True}),
            },
            "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
        }

    # Define outputs for the workflow graph
    RETURN_TYPES = ("MASK", "BBOX")
    RETURN_NAMES = ("mask", "bbox")
    FUNCTION = "get_region_and_update_ui"
    OUTPUT_NODE = True
    CATEGORY = "SmartImageTools"

    # Instance method to process region data
    def get_region_and_update_ui(self, image, type, region_data, prompt=None, extra_pnginfo=None):
        # 1. Prepare UI data (image preview)
        img_tensor = image[0]
        h, w = img_tensor.shape[0:2]
        img_np = np.clip(img_tensor.cpu().numpy(), 0.0, 1.0)
        img_uint8 = (img_np * 255.0).astype(np.uint8)
        pil_img = Image.fromarray(img_uint8)
        buffer = io.BytesIO()
        pil_img.save(buffer, format="PNG")
        buffer.seek(0)
        img_base64 = base64.b64encode(buffer.read()).decode('utf-8')
        ui_data = {
            "images": [f"data:image/png;base64,{img_base64}"],
            "dimensions": [[w, h]],
            "region_data": region_data,  # Send region_data to JS
            "region_type": type  # Send type to JS
        }

        # 2. Parse region_data JSON
        try:
            region = json.loads(region_data)
            if type == "Rectangle" and not isinstance(region, dict):
                region = {"x1": 0.25, "y1": 0.25, "x2": 0.75, "y2": 0.75}
            elif type == "Polygon" and not isinstance(region, list):
                # Default triangle if invalid polygon data
                region = [{"x": 0.25, "y": 0.25}, {"x": 0.75, "y": 0.25}, {"x": 0.5, "y": 0.75}]
        except Exception as e:
            print(f"Error parsing region_data: {e}")
            if type == "Rectangle":
                region = {"x1": 0.25, "y1": 0.25, "x2": 0.75, "y2": 0.75}
            else:  # Polygon
                region = [{"x": 0.25, "y": 0.25}, {"x": 0.75, "y": 0.25}, {"x": 0.5, "y": 0.75}]

        # 3. Generate mask image
        # Create a binary mask where the region is 1.0 and everywhere else is 0.0
        # Standard MASK format is [batch, height, width] (no channels dimension)
        mask_np = np.zeros((h, w), dtype=np.float32)

        if type == "Rectangle":
            # Get pixel coordinates (convert from 0-1 to pixel coords)
            # Note: ComfyUI and our JS use top-left as origin with y increasing downward
            # so we need to flip the y-coordinates for the mask
            x1 = max(0, min(w-1, int(region.get("x1", 0.25) * w)))
            y1 = max(0, min(h-1, int(region.get("y1", 0.25) * h)))  # No y-flip for consistency with bbox
            x2 = max(0, min(w-1, int(region.get("x2", 0.75) * w)))
            y2 = max(0, min(h-1, int(region.get("y2", 0.75) * h)))  # No y-flip for consistency with bbox

            # Ensure x1 <= x2 and y1 <= y2
            x1, x2 = min(x1, x2), max(x1, x2)
            y1, y2 = min(y1, y2), max(y1, y2)

            # Fill the region in the mask
            mask_np[y1:y2+1, x1:x2+1] = 1.0

            # 4. Create bbox output (normalized coordinates)
            # Use original normalized coordinates from region data for consistency
            x1_norm = region.get("x1", 0.25)
            y1_norm = region.get("y1", 0.25)
            x2_norm = region.get("x2", 0.75)
            y2_norm = region.get("y2", 0.75)

            # Ensure normalized coordinates are in correct order and range
            x1_norm, x2_norm = min(x1_norm, x2_norm), max(x1_norm, x2_norm)
            y1_norm, y2_norm = min(y1_norm, y2_norm), max(y1_norm, y2_norm)
            x1_norm = max(0.0, min(1.0, x1_norm))
            y1_norm = max(0.0, min(1.0, y1_norm))
            x2_norm = max(0.0, min(1.0, x2_norm))
            y2_norm = max(0.0, min(1.0, y2_norm))

            # Calculate width and height
            width_norm = x2_norm - x1_norm
            height_norm = y2_norm - y1_norm

            # BBOX format: [x, y, width, height] (normalized 0-1)
            bbox = [x1_norm, y1_norm, width_norm, height_norm]

        else:  # Polygon type
            # Convert normalized polygon points to pixel coordinates
            if not region or len(region) < 3:  # Ensure we have at least 3 points for a valid polygon
                # Default triangle if invalid polygon data
                region = [{"x": 0.25, "y": 0.25}, {"x": 0.75, "y": 0.25}, {"x": 0.5, "y": 0.75}]
                
            poly_points = []
            for point in region:
                x = max(0, min(w-1, int(point.get("x", 0.5) * w)))
                y = max(0, min(h-1, int(point.get("y", 0.5) * h)))
                poly_points.append([x, y])
                
            # Convert to numpy array for OpenCV
            poly_points_np = np.array(poly_points, dtype=np.int32)
            
            # Fill polygon in the mask using OpenCV
            cv2.fillPoly(mask_np, [poly_points_np], 1.0)
            
            # Calculate bounding box of the polygon
            min_x, min_y = 1.0, 1.0
            max_x, max_y = 0.0, 0.0
            
            for point in region:
                x_norm = max(0.0, min(1.0, point.get("x", 0.5)))
                y_norm = max(0.0, min(1.0, point.get("y", 0.5)))
                min_x = min(min_x, x_norm)
                min_y = min(min_y, y_norm)
                max_x = max(max_x, x_norm)
                max_y = max(max_y, y_norm)
            
            # Calculate width and height
            width_norm = max_x - min_x
            height_norm = max_y - min_y
            
            # BBOX format: [x, y, width, height] (normalized 0-1)
            bbox = [min_x, min_y, width_norm, height_norm]

        # Convert to tensor with batch dimension [batch, height, width]
        mask_tensor = torch.from_numpy(mask_np).unsqueeze(0)

        # Return results for the workflow graph AND data for the UI
        return {"result": (mask_tensor, bbox), "ui": ui_data}


class SmartImagePaletteExtract:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
            },
            "optional": {
                 "max_colors": ("INT", {"default": 255, "min": 1, "max": 255, "step": 1}),
            }
        }

    RETURN_TYPES = ("IMAGE", "INT")
    RETURN_NAMES = ("palette", "num_colors")
    FUNCTION = "extract_palette"
    CATEGORY = "SmartImageTools"

    # Copied from SmartImagePaletteConvert for consistent sorting
    @staticmethod
    def _sort_palette(palette):
        """Sorts a palette primarily by hue, then luminance, then saturation.
           Assigns special hue values for gray/transparent colors.
        """
        if palette.shape[0] <= 1:
            return palette

        has_alpha = palette.shape[1] == 4
        color_data = [] # List to store (sort_key, original_color)

        for color_val in palette:
            rgb = color_val[:3] / 255.0
            alpha = color_val[3] / 255.0 if has_alpha else 1.0
            is_transparent = has_alpha and color_val[3] == 0

            h, l, s = colorsys.rgb_to_hls(rgb[0], rgb[1], rgb[2])

            # Define hue_value for sorting
            # -2: Transparent
            # -1: Achromatic (Gray/Black/White)
            # 0-1: Chromatic hue
            if is_transparent:
                hue_value = -2
                l = 0.0 # Use luminance of black for sorting transparent
                s = 0.0
            elif s < 0.05 or l < 0.01 or l > 0.99: # Achromatic threshold
                hue_value = -1
            else:
                hue_value = h

            sort_key = (hue_value, l, s)
            color_data.append((sort_key, color_val))

        color_data.sort(key=lambda x: x[0])
        sorted_palette_list = [item[1] for item in color_data]
        return np.array(sorted_palette_list, dtype=palette.dtype)

    def extract_palette(self, image, max_colors=255):
        input_image = 255. * image.cpu().numpy()
        # Process only the first image in the batch
        img = input_image[0].astype(np.uint8)
        h, w, channels = img.shape
        has_alpha = channels == 4

        # Reshape to get all pixels
        img_flat = img.reshape(-1, channels)

        # Get unique colors
        unique_colors = np.unique(img_flat, axis=0)
        num_unique = len(unique_colors)

        if num_unique > max_colors:
            raise ValueError(f"Image contains {num_unique} unique colors, exceeding the limit of {max_colors}.")

        # Sort the palette
        sorted_palette = self._sort_palette(unique_colors)

        # Create palette image (1 pixel high)
        palette_height = 1
        palette_width = len(sorted_palette)
        palette_img = np.zeros((palette_height, palette_width, channels), dtype=np.uint8)

        if palette_width > 0:
             for i in range(palette_width):
                 palette_img[:, i] = sorted_palette[i]

        # Convert to tensor (add batch dimension and normalize)
        palette_tensor = torch.from_numpy(palette_img[np.newaxis, ...] / 255.0).float()

        return (palette_tensor, num_unique)


class SmartSemiTransparenceRemove:
    # Define color map with RGB values (0-1 range for easier blending)
    COLOR_MAP = {
        "red": [1.0, 0.0, 0.0],
        "green": [0.0, 1.0, 0.0],
        "blue": [0.0, 0.0, 1.0],
        "black": [0.0, 0.0, 0.0],
        "white": [1.0, 1.0, 1.0],
        "yellow": [1.0, 1.0, 0.0],
        "cyan": [0.0, 1.0, 1.0],
        "magenta": [1.0, 0.0, 1.0],
        "gray": [0.5, 0.5, 0.5],
        "dark gray": [0.25, 0.25, 0.25],
        "orange": [1.0, 0.647, 0.0],
        "purple": [0.5, 0.0, 0.5],
        "brown": [0.647, 0.165, 0.165],
        "pink": [1.0, 0.753, 0.796],
    }

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "blend_color": (list(s.COLOR_MAP.keys()),),
                "threshold": ("FLOAT", {"default": 0.1, "min": 0.0, "max": 1.0, "step": 0.01}),
                "over_threshold": (["Blend", "Fill blend color"],),
                "fill_threshold": ("FLOAT", {"default": 0.95, "min": 0.0, "max": 1.0, "step": 0.01}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "remove_transparence"
    CATEGORY = "SmartImageTools"

    def remove_transparence(self, image, blend_color, threshold, over_threshold="Blend", fill_threshold=0.95):
        batch_size, height, width, channels = image.shape
        device = image.device
        dtype = image.dtype

        # Get blend color RGB values
        blend_rgb = torch.tensor(self.COLOR_MAP[blend_color], device=device, dtype=dtype)

        # Clone image to avoid modifying original tensor
        result_image = image.clone()

        # Ensure image has alpha channel
        if channels == 3:
            # Add alpha channel, initialize to fully opaque
            alpha_channel = torch.ones((batch_size, height, width, 1), device=device, dtype=dtype)
            result_image = torch.cat((result_image, alpha_channel), dim=3)
        elif channels != 4:
            raise ValueError("Input image must have 3 (RGB) or 4 (RGBA) channels.")

        # Process pixels
        alpha = result_image[:, :, :, 3]
        rgb = result_image[:, :, :, :3]

        # Condition 1: Alpha is 1 (fully opaque) - do nothing, alpha already 1

        # Condition 2: Alpha < threshold - make fully transparent
        mask_transparent = alpha < threshold # Shape [B, H, W]
        num_transparent_pixels = mask_transparent.sum()

        if num_transparent_pixels > 0:
            # Set alpha channel to 0 using masked assignment on the slice
            result_image[:, :, :, 3][mask_transparent] = 0.0
            # Set RGB channels to 0 using masked assignment on the slice
            result_image[:, :, :, :3][mask_transparent] = 0.0

        if over_threshold == "Blend":
            # Original behavior: blend all semi-transparent pixels
            # Condition 3: threshold <= Alpha < 1 - blend and make fully opaque
            mask_blend = (alpha >= threshold) & (alpha < 1.0) # Shape [B, H, W]
            num_blend_pixels = mask_blend.sum()

            if num_blend_pixels > 0:
                # Expand alpha and blend_rgb for broadcasting
                alpha_expanded = alpha[mask_blend].unsqueeze(-1) # Shape: [num_blend_pixels, 1]
                blend_rgb_expanded = blend_rgb.unsqueeze(0)     # Shape: [1, 3]

                # Calculate blended RGB for the relevant pixels
                original_rgb_subset = rgb[mask_blend]           # Shape: [num_blend_pixels, 3]
                blended_rgb = original_rgb_subset * alpha_expanded + blend_rgb_expanded * (1.0 - alpha_expanded) # Shape: [num_blend_pixels, 3]

                # Assign blended RGB back to the result image using the mask on the sliced tensor
                result_image[:, :, :, :3][mask_blend] = blended_rgb

                # Set alpha to 1 for these pixels using the mask on the sliced tensor
                result_image[:, :, :, 3][mask_blend] = 1.0
        else:
            # "Fill blend color" mode
            # Pixels with threshold <= alpha < fill_threshold: filled with blend color
            mask_fill = (alpha >= threshold) & (alpha < fill_threshold) # Shape [B, H, W]
            num_fill_pixels = mask_fill.sum()

            if num_fill_pixels > 0:
                # Fill with blend color (no blending)
                result_image[:, :, :, :3][mask_fill] = blend_rgb.unsqueeze(0).unsqueeze(0).unsqueeze(0)
                # Set alpha to 1
                result_image[:, :, :, 3][mask_fill] = 1.0

            # Pixels with fill_threshold <= alpha < 1.0: blended normally
            mask_blend = (alpha >= fill_threshold) & (alpha < 1.0) # Shape [B, H, W]
            num_blend_pixels = mask_blend.sum()

            if num_blend_pixels > 0:
                # Expand alpha and blend_rgb for broadcasting
                alpha_expanded = alpha[mask_blend].unsqueeze(-1) # Shape: [num_blend_pixels, 1]
                blend_rgb_expanded = blend_rgb.unsqueeze(0)     # Shape: [1, 3]

                # Calculate blended RGB for the relevant pixels
                original_rgb_subset = rgb[mask_blend]           # Shape: [num_blend_pixels, 3]
                blended_rgb = original_rgb_subset * alpha_expanded + blend_rgb_expanded * (1.0 - alpha_expanded) # Shape: [num_blend_pixels, 3]

                # Assign blended RGB back to the result image using the mask on the sliced tensor
                result_image[:, :, :, :3][mask_blend] = blended_rgb

                # Set alpha to 1 for these pixels using the mask on the sliced tensor
                result_image[:, :, :, 3][mask_blend] = 1.0

        return (result_image,)


# Color map for background blending (0-255 range)
BG_COLOR_MAP = {
    "transparent": None, # Special case
    "red": (255, 0, 0),
    "green": (0, 255, 0),
    "blue": (0, 0, 255),
    "black": (0, 0, 0),
    "white": (255, 255, 255),
    "yellow": (255, 255, 0),
    "cyan": (0, 255, 255),
    "magenta": (255, 0, 255),
    "gray": (128, 128, 128),
    "dark gray": (64, 64, 64),
    "orange": (255, 165, 0),
    "purple": (128, 0, 128),
    "brown": (165, 42, 42),
    "pink": (255, 192, 203),
}

class SmartVideoPreviewScaled:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE",),
                "fps": ("FLOAT", {"default": 12.0, "min": 0.1, "max": 60.0, "step": 1}),
                "scale_by": ("FLOAT", {"default": 2.0, "min": 0.1, "max": 10.0, "step": 0.1}),
                "bg_color": (list(BG_COLOR_MAP.keys()), {"default": "transparent"}), # Added bg_color
            },
            "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
        }

    RETURN_TYPES = ()
    FUNCTION = "preview_video"
    OUTPUT_NODE = True
    CATEGORY = "SmartNodes/Image"

    @classmethod
    def IS_CHANGED(s, images, fps, scale_by, bg_color, prompt=None, extra_pnginfo=None):
        # Include all parameters that affect the output to prevent incorrect caching
        # Create a hash of the parameters to ensure unique cache keys
        m = hashlib.sha256()
        m.update(str(fps).encode())
        m.update(str(scale_by).encode())
        m.update(str(bg_color).encode())
        # Add a random component to force re-execution for preview nodes
        # This ensures that even with identical parameters, preview updates properly
        m.update(str(random.random()).encode())
        return m.digest().hex()

    def preview_video(self, images, fps, scale_by=2.0, bg_color="transparent", prompt=None, extra_pnginfo=None):
        # Convert tensors to PIL images and prepare for web UI
        image_data = []
        original_dimensions = []
        scaled_dimensions = []

        for image_tensor in images:
            img_np = image_tensor.cpu().numpy()
            # Handle potential non-standard ranges by clipping before conversion
            img_np = np.clip(img_np, 0.0, 1.0)
            img_pil = Image.fromarray((img_np * 255.0).astype(np.uint8))

            # Store original dimensions
            orig_w, orig_h = img_pil.size
            original_dimensions.append([orig_w, orig_h])

            # --- Scaling happens here ---
            if abs(scale_by - 1.0) > 1e-6 and scale_by > 0:
                new_w = max(1, int(round(orig_w * scale_by)))
                new_h = max(1, int(round(orig_h * scale_by)))
                # Resize using NEAREST neighbor
                img_pil = img_pil.resize((new_w, new_h), Image.NEAREST)
            else:
                # If no scaling, scaled dimensions are same as original
                new_w, new_h = orig_w, orig_h
            # --- End Scaling ---

            # Store scaled dimensions
            scaled_dimensions.append([new_w, new_h])

            # --- Background Blending ---
            if bg_color != "transparent" and bg_color in BG_COLOR_MAP:
                blend_rgb = BG_COLOR_MAP[bg_color]
                if blend_rgb is not None:
                    # Ensure image is RGBA for compositing
                    if img_pil.mode != 'RGBA':
                        img_pil = img_pil.convert('RGBA')

                    # Create a background image of the solid color
                    background = Image.new('RGBA', img_pil.size, blend_rgb + (255,))

                    # Composite the original image over the background
                    # This automatically handles the alpha blending
                    img_pil = Image.alpha_composite(background, img_pil)
            # --- End Background Blending ---

            # Save the (potentially scaled and blended) image to a temporary location
            output_dir = folder_paths.get_temp_directory()
            filename = f"smart_video_frame_{random.randint(1000000, 9999999)}.png" # Add randomness to filename
            file_path = os.path.join(output_dir, filename) # Use os.path.join
            img_pil.save(file_path)
            image_data.append({
                "filename": filename,
                "subfolder": "", # Saved directly in temp
                "type": "temp"
            })

        # Return images, fps, and both original and scaled dimensions
        # Use a unique key 'video_frames' to avoid conflicts with default preview handlers
        return {"ui": {
            "video_frames": image_data,
            "fps": [fps],
            "original_dimensions": original_dimensions,
            "scaled_dimensions": scaled_dimensions
        }}


class SmartVideoPreviewScaledMasked:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE",),
                "fps": ("FLOAT", {"default": 12.0, "min": 0.1, "max": 60.0, "step": 1}),
                "scale_by": ("FLOAT", {"default": 2.0, "min": 0.1, "max": 10.0, "step": 0.1}),
                "mask_color": ("STRING", {"default": "#FF0000FF"}),  # RGBA color as hex
            },
            "optional": {
                "mask": ("MASK",),
            },
            "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("composed_image",)
    FUNCTION = "preview_video"
    OUTPUT_NODE = True
    CATEGORY = "SmartNodes/Image"

    @classmethod
    def IS_CHANGED(s, images, fps, scale_by, mask_color, mask=None, prompt=None, extra_pnginfo=None):
        # Include all parameters to prevent incorrect caching
        m = hashlib.sha256()
        m.update(str(fps).encode())
        m.update(str(scale_by).encode())
        m.update(str(mask_color).encode())
        # Add random component for preview nodes
        m.update(str(random.random()).encode())
        return m.digest().hex()

    def hex_to_rgba(self, hex_color):
        """Convert hex color string to RGBA tuple (0-255 range)."""
        hex_color = hex_color.lstrip('#')
        if len(hex_color) == 6:
            # RGB only, add full opacity
            r = int(hex_color[0:2], 16)
            g = int(hex_color[2:4], 16)
            b = int(hex_color[4:6], 16)
            a = 255
        elif len(hex_color) == 8:
            # RGBA
            r = int(hex_color[0:2], 16)
            g = int(hex_color[2:4], 16)
            b = int(hex_color[4:6], 16)
            a = int(hex_color[6:8], 16)
        else:
            # Default to red if invalid
            r, g, b, a = 255, 0, 0, 255
        return (r, g, b, a)

    def preview_video(self, images, fps, scale_by=2.0, mask_color="#FF0000FF", mask=None, prompt=None, extra_pnginfo=None):
        # Convert tensors to numpy
        images_np = images.cpu().numpy()
        mask_np = mask.cpu().numpy() if mask is not None else None
        
        # Parse mask color
        rgba_color = self.hex_to_rgba(mask_color)
        
        image_data = []
        original_dimensions = []
        scaled_dimensions = []
        composed_images = []
        
        batch_size = images_np.shape[0]
        
        for i in range(batch_size):
            # Get image
            img_np = images_np[i]
            img_np = np.clip(img_np, 0.0, 1.0)
            
            # Convert image to uint8
            img_uint8 = (img_np * 255.0).astype(np.uint8)
            h, w = img_uint8.shape[:2]
            
            # Convert to PIL
            img_pil = Image.fromarray(img_uint8)
            
            # Store original dimensions
            orig_w, orig_h = img_pil.size
            original_dimensions.append([orig_w, orig_h])
            
            # Apply mask overlay if mask is provided
            if mask_np is not None:
                # Get mask (handle single mask or batch)
                if i < mask_np.shape[0]:
                    mask_frame = mask_np[i]
                else:
                    mask_frame = mask_np[0]  # Use first mask if batch is smaller
                
                # Resize mask to match image dimensions if needed
                if mask_frame.shape[0] != h or mask_frame.shape[1] != w:
                    mask_frame = cv2.resize(mask_frame, (w, h), interpolation=cv2.INTER_LINEAR)
                
                # Ensure image is RGBA for compositing
                if img_pil.mode != 'RGBA':
                    img_pil = img_pil.convert('RGBA')
                
                # Create mask overlay layer
                # mask_frame values are 0-1, where 1 = fully masked
                mask_overlay = np.zeros((h, w, 4), dtype=np.uint8)
                mask_overlay[:, :, 0] = rgba_color[0]  # R
                mask_overlay[:, :, 1] = rgba_color[1]  # G
                mask_overlay[:, :, 2] = rgba_color[2]  # B
                # Alpha channel of overlay = mask * color_alpha
                mask_overlay[:, :, 3] = (mask_frame * rgba_color[3]).astype(np.uint8)
                
                # Convert mask overlay to PIL
                mask_overlay_pil = Image.fromarray(mask_overlay, 'RGBA')
                
                # Composite: base image + mask overlay
                img_pil = Image.alpha_composite(img_pil, mask_overlay_pil)
            
            # --- Scaling happens here ---
            if abs(scale_by - 1.0) > 1e-6 and scale_by > 0:
                new_w = max(1, int(round(orig_w * scale_by)))
                new_h = max(1, int(round(orig_h * scale_by)))
                # Resize using NEAREST neighbor
                img_pil = img_pil.resize((new_w, new_h), Image.NEAREST)
            else:
                # If no scaling, scaled dimensions are same as original
                new_w, new_h = orig_w, orig_h
            # --- End Scaling ---
            
            # Store scaled dimensions
            scaled_dimensions.append([new_w, new_h])
            
            # Convert back to tensor for output
            img_array = np.array(img_pil).astype(np.float32) / 255.0
            composed_images.append(torch.from_numpy(img_array))
            
            # Save the image to a temporary location for preview
            output_dir = folder_paths.get_temp_directory()
            filename = f"smart_video_masked_{random.randint(1000000, 9999999)}.png"
            file_path = os.path.join(output_dir, filename)
            img_pil.save(file_path)
            image_data.append({
                "filename": filename,
                "subfolder": "",
                "type": "temp"
            })
        
        # Stack composed images into batch
        composed_batch = torch.stack(composed_images)
        
        # Return with both output tensor and UI preview
        return {
            "ui": {
                "video_frames": image_data,
                "fps": [fps],
                "original_dimensions": original_dimensions,
                "scaled_dimensions": scaled_dimensions
            },
            "result": (composed_batch,)
        }


class SmartSaveAnimatedPNG:
    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()
        self.type = "output"

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE", ),
                "fps": ("FLOAT", {"default": 12.0, "min": 0.1, "max": 1000.0, "step": 0.1}),
                "filename_prefix": ("STRING", {"default": "ComfyUI"}),
                # Keep lossy_quality for potential future use, but set default high
                "lossless": ("BOOLEAN", {"default": True}),
                "save_metadata": ("BOOLEAN", {"default": True}),
                "format": (["APNG", "PNGif", "GIF"], {"default": "APNG"}),
             },
            "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
        }

    RETURN_TYPES = ()
    FUNCTION = "save_apng"
    OUTPUT_NODE = True
    CATEGORY = "SmartImageTools"
    
    def save_animated_gif(self, images, filename, fps, loop, optimize, quality, preserve_transparency, alpha_threshold):
        try:
            from PIL import Image
            import numpy as np
            import torch
        except ImportError as e:
            raise RuntimeError(f"Required libraries not available: {e}. Please install Pillow.")

        # Ensure filename has .gif extension
        if not filename.lower().endswith('.gif'):
            if filename.endswith('.'):
                filename = filename + 'gif'
            else:
                filename = filename + '.gif'

        # Convert relative path to absolute path within ComfyUI's output directory
        output_dir = folder_paths.get_output_directory()
        full_path = os.path.join(output_dir, filename)
        
        # Create directories if they don't exist
        directory = os.path.dirname(full_path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        
        # Convert images to PIL format
        pil_frames = []
        has_transparent_pixel = False
        used_colors = set()
        alpha_thresh = alpha_threshold * 255

        if isinstance(images, torch.Tensor) and images.dim() == 4:
            # Handle batch of images
            batch_size = images.shape[0]
        else:
            # If not a batch tensor, treat as list
            batch_size = len(images)

        for i in range(batch_size):
            if isinstance(images, torch.Tensor):
                img = images[i]
            else:
                img = images[i]

            if isinstance(img, torch.Tensor):
                img = img.detach().cpu()
                if img.dim() == 3:
                    if img.shape[0] in [1, 3, 4]:  # CHW
                        img = img.permute(1, 2, 0)
                if img.max() <= 1.0:
                    img = img * 255
                img = img.round().clamp(0, 255).byte().numpy()
                mode = 'RGBA' if img.shape[-1] == 4 else 'RGB' if img.shape[-1] == 3 else 'L' if img.shape[-1] == 1 else None
                if mode is None:
                    raise ValueError(f"Unsupported tensor channel count: {img.shape[-1]}")
                if mode == 'L':
                    img = np.repeat(img[:, :, np.newaxis], 3, axis=-1)
                    mode = 'RGB'
                pil_img = Image.fromarray(img, mode)
            elif isinstance(img, np.ndarray):
                if img.max() <= 1.0:
                    img = img * 255
                img = np.round(img).clip(0, 255).astype(np.uint8)
                mode = 'RGBA' if img.shape[-1] == 4 else 'RGB' if img.shape[-1] == 3 else 'L' if img.shape[-1] == 1 else None
                if mode is None:
                    raise ValueError(f"Unsupported array channel count: {img.shape[-1]}")
                if mode == 'L':
                    img = np.repeat(img[:, :, np.newaxis], 3, axis=-1)
                    mode = 'RGB'
                pil_img = Image.fromarray(img, mode)
            elif isinstance(img, Image.Image):
                pil_img = img
            else:
                raise ValueError(f"Unsupported image type: {type(img)}")

            # Convert to RGBA for consistency if transparency might be present
            if pil_img.mode not in ['RGB', 'RGBA']:
                pil_img = pil_img.convert('RGB')

            if preserve_transparency and pil_img.mode == 'RGBA':
                data = np.array(pil_img)
                alpha = data[:, :, 3]
                if np.any(alpha < alpha_thresh):
                    has_transparent_pixel = True
                opaque_mask = alpha >= alpha_thresh
                opaque_rgb = data[opaque_mask, :3]
                for color in opaque_rgb:
                    used_colors.add(tuple(map(int, color)))

            pil_frames.append(pil_img)

        if not pil_frames:
            raise ValueError("No valid images provided")

        if not preserve_transparency or not has_transparent_pixel:
            # Composite to RGB with white background
            for i in range(len(pil_frames)):
                if pil_frames[i].mode == 'RGBA':
                    background = Image.new('RGB', pil_frames[i].size, (255, 255, 255))
                    background.paste(pil_frames[i], mask=pil_frames[i].getchannel('A'))
                    pil_frames[i] = background
            pil_images = pil_frames
        else:
            # Find global key_color not used in any opaque pixel
            key_color = None
            attempts = 0
            while attempts < 10000 and key_color is None:
                r = random.randint(0, 255)
                g = random.randint(0, 255)
                b = random.randint(0, 255)
                if (r, g, b) not in used_colors:
                    key_color = (r, g, b)
                attempts += 1

            if key_color is None:
                key_color = (255, 0, 255)
                # Replace occurrences of key_color in opaque areas
                for i in range(len(pil_frames)):
                    if pil_frames[i].mode == 'RGBA':
                        data = np.array(pil_frames[i])
                        alpha = data[:, :, 3]
                        match = (data[:, :, 0] == 255) & (data[:, :, 1] == 0) & (data[:, :, 2] == 255) & (alpha >= alpha_thresh)
                        data[match, 0] = 254
                        pil_frames[i] = Image.fromarray(data, 'RGBA')

            # Create RGB frames with key_color in transparent areas
            rgb_frames = []
            for frame in pil_frames:
                if frame.mode == 'RGBA':
                    data = np.array(frame)
                    transparent_mask = data[:, :, 3] < alpha_thresh
                    rgb_data = data[:, :, :3]
                    rgb_data[transparent_mask] = key_color
                    frame_rgb = Image.fromarray(rgb_data, 'RGB')
                else:
                    frame_rgb = frame.convert('RGB')
                rgb_frames.append(frame_rgb)

            # Create large image for global palette
            widths, heights = zip(*(f.size for f in rgb_frames))
            max_width = max(widths)
            total_height = sum(heights) + 32  # Extra for key block
            large_img = Image.new('RGB', (max_width, total_height))
            y = 0
            for f in rgb_frames:
                large_img.paste(f, (0, y))
                y += f.height
            key_block = Image.new('RGB', (32, 32), key_color)
            large_img.paste(key_block, (0, y))

            # Quantize large image
            large_quant = large_img.quantize(colors=256, method=Image.Quantize.MAXCOVERAGE)
            global_palette = large_quant.getpalette()[:256*3]

            # Find trans_index
            trans_index = -1
            for i in range(256):
                if (global_palette[i*3] == key_color[0] and
                    global_palette[i*3 + 1] == key_color[1] and
                    global_palette[i*3 + 2] == key_color[2]):
                    trans_index = i
                    break

            if trans_index == -1:
                # Fallback to per-frame palettes
                pil_images = []
                for frame_rgb in rgb_frames:
                    frame_p = frame_rgb.quantize(colors=256)
                    palette = frame_p.getpalette()[:256*3]
                    for j in range(256):
                        if (palette[j*3] == key_color[0] and
                            palette[j*3 + 1] == key_color[1] and
                            palette[j*3 + 2] == key_color[2]):
                            frame_p.info['transparency'] = j
                            frame_p.info['disposal'] = 2
                            break
                        pil_images.append(frame_p)
                    if pil_images:
                        pil_images[0].info['background'] = pil_images[0].info.get('transparency', 0)
            else:
                # Use global palette
                palette_img = Image.new('P', (1, 1))
                palette_img.putpalette(global_palette)
                pil_images = []
                for frame_rgb in rgb_frames:
                    frame_p = frame_rgb.quantize(palette=palette_img)
                    frame_p.info['transparency'] = trans_index
                    frame_p.info['disposal'] = 2
                    pil_images.append(frame_p)
                if pil_images:
                    pil_images[0].info['background'] = trans_index

        # Calculate duration per frame in milliseconds
        duration = int(1000 / fps)
        
        # Save as animated GIF
        pil_images[0].save(
            full_path,
            save_all=True,
            append_images=pil_images[1:],
            duration=duration,
            loop=loop,
            optimize=optimize,
            quality=quality
        )
        
        return (full_path,)	

    def save_apng(self, images, fps, filename_prefix="ComfyUI", lossless=True, save_metadata=True, format="APNG", prompt=None, extra_pnginfo=None):
        full_output_folder, filename, counter, subfolder, filename_prefix_ = folder_paths.get_save_image_path(filename_prefix, self.output_dir, images[0].shape[1], images[0].shape[0])
        # Extract the base filename part from the potentially path-like filename_prefix_
        base_filename = os.path.basename(filename_prefix_)
        # Use a single filename, not numbered sequence
        if format == "APNG":
            ext = "apng"
        elif format == "PNGif":
            ext = "pngif"
        else:
            ext = "gif"
        file = f"{base_filename}.{ext}" # Use the extracted base name
        file_path = os.path.join(full_output_folder, file)

        # GIF branch: delegate to save_animated_gif
        if format == "GIF":
            results = list()
            try:
                relative_filename = os.path.join(subfolder, file) if subfolder else file
                self.save_animated_gif(
                    images=images,
                    filename=relative_filename,
                    fps=fps,
                    loop=0,
                    optimize=True,
                    quality=95,
                    preserve_transparency=True,
                    alpha_threshold=0.01,
                )

                results.append({
                    "filename": file,
                    "subfolder": subfolder,
                    "type": self.type
                })

            except Exception as e:
                print(f"Error saving GIF file: {e}")
                raise

            return {"ui": {"apng": results}}

        num_frames = images.shape[0]
        fps_int = int(round(fps * 100)) # Multiply by 100 and convert to int

        results = list()

        try:
            with open(file_path, 'wb') as f:
                # Write header: num_frames (int), fps_int (int)
                # Use '<i' for 4-byte signed integer, little-endian
                f.write(struct.pack('<i', num_frames))
                f.write(struct.pack('<i', fps_int))

                # Process and write each frame
                for image_tensor in images:
                    # Convert tensor [0, 1] float -> numpy [0, 255] uint8
                    img_np = image_tensor.cpu().numpy()
                    img_np = np.clip(img_np, 0.0, 1.0)
                    img_uint8 = (img_np * 255.0).astype(np.uint8)
                    img_pil = Image.fromarray(img_uint8)

                    # Save PIL image to bytes buffer as PNG
                    buffer = io.BytesIO()
                    # Determine PNG save options based on lossless flag
                    pnginfo = None
                    if save_metadata and prompt is not None:
                        pnginfo = PngInfo()
                        if extra_pnginfo is not None:
                            for key in extra_pnginfo:
                                pnginfo.add_text(key, json.dumps(extra_pnginfo[key]))
                        pnginfo.add_text("prompt", json.dumps(prompt))

                    save_opts = {"pnginfo": pnginfo}
                    if not lossless:
                        # While PNG is typically lossless, PIL allows 'compress_level'
                        # 0 = no compression, 1 = fastest, 9 = best compression (slowest)
                        # Let's default to a moderate level if lossless=False, though it's still lossless compression
                        # If a truly lossy format were desired, we'd need WebP or similar.
                        # For simplicity here, we just vary compression level.
                        save_opts["compress_level"] = 4 # Moderate compression if "lossless" is false
                    else:
                        save_opts["compress_level"] = 1 # Faster compression if lossless is true

                    img_pil.save(buffer, format="PNG", **save_opts)
                    png_data = buffer.getvalue()
                    png_length = len(png_data)

                    # Write frame data: png_length (int), png_data (binary)
                    f.write(struct.pack('<i', png_length))
                    f.write(png_data)

            results.append({
                "filename": file,
                "subfolder": subfolder,
                "type": self.type
            })

        except Exception as e:
            print(f"Error saving custom APNG file: {e}")
            # Optionally re-raise or handle the error appropriately
            raise

        # Return UI data indicating the saved file
        return {"ui": {"apng": results}}


class SmartSavePNG:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE", ),
                "filename": ("STRING", {"default": "C:/output.png"}),
             },
            "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "save_png"
    OUTPUT_NODE = True
    CATEGORY = "SmartImageTools"

    def save_png(self, images, filename, prompt=None, extra_pnginfo=None):
        # Get the first image from the batch
        image_tensor = images[0] if len(images.shape) > 3 else images
        
        # Convert tensor [0, 1] float -> numpy [0, 255] uint8
        img_np = image_tensor.cpu().numpy()
        img_np = np.clip(img_np, 0.0, 1.0)
        img_uint8 = (img_np * 255.0).astype(np.uint8)
        img_pil = Image.fromarray(img_uint8)
        
        # Create PngInfo metadata if needed
        pnginfo = None
        if prompt is not None:
            pnginfo = PngInfo()
            if extra_pnginfo is not None:
                for key in extra_pnginfo:
                    pnginfo.add_text(key, json.dumps(extra_pnginfo[key]))
            pnginfo.add_text("prompt", json.dumps(prompt))
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(filename)), exist_ok=True)
        
        # Save the image
        img_pil.save(filename, format="PNG", pnginfo=pnginfo)
        
        return (images,)


class SmartDrawPoints:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "points": ("POINT_SET",),
                "color": (list(BG_COLOR_MAP.keys()),),
                "circle_size": ("INT", {"default": 5, "min": 1, "max": 100, "step": 1}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "draw_points"
    CATEGORY = "SmartImageTools"

    def draw_points(self, image, points, color, circle_size):
        # Convert from tensor to numpy array
        input_image = 255. * image.cpu().numpy()
        batch_size = input_image.shape[0]
        output_images = []

        for i in range(batch_size):
            img = input_image[i].copy()
            h, w = img.shape[:2]
            
            # Convert image to uint8 for OpenCV
            img_uint8 = img.astype(np.uint8)
            
            # Get color values
            if color == "transparent":
                # For transparent, use black with alpha=0
                circle_color = (0, 0, 0, 0)
            else:
                # Get RGB values and add alpha=255
                rgb = BG_COLOR_MAP[color]
                circle_color = rgb + (255,)
            
            # Draw circles for each point
            for point in points:
                # Convert point from normalized (0-1) to pixel coordinates
                x = int(point[0] * w)
                y = int((1 - point[1]) * h)  # Invert y as OpenCV uses top-left origin
                
                # Clamp coordinates to valid range
                x = max(0, min(x, w - 1))
                y = max(0, min(y, h - 1))
                
                # Draw filled circle
                cv2.circle(img_uint8, (x, y), circle_size, circle_color, -1)
            
            output_images.append(img_uint8)
        
        # Convert back to tensor
        output_tensor = torch.from_numpy(np.stack(output_images) / 255.0).float()
        
        return (output_tensor,)


class SmartLoadGIFImage:
    @classmethod
    def INPUT_TYPES(s):
        input_dir = folder_paths.get_input_directory()
        files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]
        files = folder_paths.filter_files_content_types(files, ["image", "animated"])
        return {"required":
                    {"image": (sorted(files), {"image_upload": True}),
                     "custom_frame": ("INT", {"default": 0, "min": 0, "max": 1000, "step": 1})},
                }

    CATEGORY = "SmartImageTools"
    RETURN_TYPES = ("IMAGE", "MASK", "IMAGE", "MASK", "IMAGE", "MASK", "IMAGE", "MASK")
    RETURN_NAMES = ("all_images", "all_masks", "first_frame", "first_mask", "last_frame", "last_mask", "custom_frame", "custom_mask")
    FUNCTION = "load_gif_image"

    def load_gif_image(self, image, custom_frame):
        image_path = folder_paths.get_annotated_filepath(image)

        img = node_helpers.pillow(Image.open, image_path)

        output_images = []
        output_masks = []
        w, h = None, None

        excluded_formats = ['MPO']

        for i in ImageSequence.Iterator(img):
            i = node_helpers.pillow(ImageOps.exif_transpose, i)

            if i.mode == 'I':
                i = i.point(lambda i: i * (1 / 255))

            if len(output_images) == 0:
                w = i.size[0]
                h = i.size[1]

            if i.size[0] != w or i.size[1] != h:
                continue

            # Handle transparency properly - output RGBA when transparency exists
            if 'A' in i.getbands():
                # Image has alpha channel - use RGBA directly
                rgba_image = i.convert("RGBA")
                alpha = np.array(rgba_image.getchannel('A')).astype(np.float32) / 255.0
                mask = 1. - torch.from_numpy(alpha)
                # Output RGBA image to preserve transparency
                image = np.array(rgba_image).astype(np.float32) / 255.0
                image = torch.from_numpy(image)[None,]
            elif i.mode == 'P' and 'transparency' in i.info:
                # Palette image with transparency - use RGBA
                rgba_image = i.convert('RGBA')
                alpha = np.array(rgba_image.getchannel('A')).astype(np.float32) / 255.0
                mask = 1. - torch.from_numpy(alpha)
                # Output RGBA image to preserve transparency
                image = np.array(rgba_image).astype(np.float32) / 255.0
                image = torch.from_numpy(image)[None,]
            else:
                # No transparency - output RGB with zero mask
                mask = torch.zeros((h, w), dtype=torch.float32, device="cpu")
                rgb_image = i.convert("RGB")
                image = np.array(rgb_image).astype(np.float32) / 255.0
                image = torch.from_numpy(image)[None,]

            output_images.append(image)
            output_masks.append(mask.unsqueeze(0))

        if len(output_images) == 0:
            raise ValueError("No valid frames found in the image")

        # All images concatenated (like original LoadImage behavior)
        if len(output_images) > 1 and img.format not in excluded_formats:
            all_images = torch.cat(output_images, dim=0)
            all_masks = torch.cat(output_masks, dim=0)
        else:
            all_images = output_images[0]
            all_masks = output_masks[0]

        # First frame and mask
        first_frame = output_images[0]
        first_mask = output_masks[0]

        # Last frame and mask
        last_frame = output_images[-1]
        last_mask = output_masks[-1]

        # Custom frame and mask
        if custom_frame < len(output_images):
            custom_frame_img = output_images[custom_frame]
            custom_frame_mask = output_masks[custom_frame]
        else:
            # If custom_frame is out of range, default to last frame
            custom_frame_img = output_images[-1]
            custom_frame_mask = output_masks[-1]

        return (all_images, all_masks, first_frame, first_mask, last_frame, last_mask, custom_frame_img, custom_frame_mask)

    @classmethod
    def IS_CHANGED(s, image, custom_frame):
        image_path = folder_paths.get_annotated_filepath(image)
        m = hashlib.sha256()
        with open(image_path, 'rb') as f:
            m.update(f.read())
        m.update(str(custom_frame).encode())
        return m.digest().hex()

    @classmethod
    def VALIDATE_INPUTS(s, image, custom_frame):
        if not folder_paths.exists_annotated_filepath(image):
            return "Invalid image file: {}".format(image)

        return True


class SmartImagePaletteCreate:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "colors": ("INT", {"default": 5, "min": 1, "max": 5, "step": 1}),
                "color_1": ("COLOR", {"default": "#000000"}),  # Black
                "color_2": ("COLOR", {"default": "#000000"}),  # Black
                "color_3": ("COLOR", {"default": "#000000"}),  # Black
                "color_4": ("COLOR", {"default": "#000000"}),  # Black
                "color_5": ("COLOR", {"default": "#000000"}),  # Black
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("palette_image",)
    FUNCTION = "create_palette"
    CATEGORY = "SmartImageTools"

    def hex_to_rgb(self, hex_color):
        """Convert hex color string to RGB tuple."""
        hex_color = hex_color.lstrip('#')
        return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

    def create_palette(self, colors, color_1, color_2, color_3, color_4, color_5):
        """Create a palette image with specified colors."""
        # Collect all color inputs
        color_inputs = [color_1, color_2, color_3, color_4, color_5]

        # Take only the number of colors specified
        selected_colors = color_inputs[:colors]

        # Convert hex colors to RGB
        rgb_colors = []
        for hex_color in selected_colors:
            rgb = self.hex_to_rgb(hex_color)
            rgb_colors.append(rgb)

        # Create palette image: 1 pixel height, colors pixels width
        height = 1
        width = len(rgb_colors)

        # Create numpy array for the image
        palette_img = np.zeros((height, width, 3), dtype=np.uint8)

        # Fill each pixel with the corresponding color
        for i, rgb in enumerate(rgb_colors):
            palette_img[0, i] = rgb

        # Convert to torch tensor and add batch dimension
        palette_tensor = torch.from_numpy(palette_img).float() / 255.0
        palette_tensor = palette_tensor.unsqueeze(0)  # Add batch dimension

        return (palette_tensor,)


class SmartBackgroundFill:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "background_color": (list(BG_COLOR_MAP.keys()), {"default": "white"}),
            },
            "optional": {
                "mask": ("MASK",),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "fill_background"
    CATEGORY = "SmartImageTools"

    def fill_background(self, image, background_color, mask=None):
        # Convert from tensor to numpy array
        input_image = 255. * image.cpu().numpy()
        batch_size = input_image.shape[0]
        output_images = []

        # Get background color RGB values (0-255 range)
        if background_color == "transparent":
            # For transparent, use black with alpha=0, but since we're filling background,
            # this doesn't make sense - default to white
            bg_rgb = (255, 255, 255)
        else:
            bg_rgb = BG_COLOR_MAP[background_color]
            if bg_rgb is None:
                # Fallback to white if color is not found
                bg_rgb = (255, 255, 255)

        for i in range(batch_size):
            img = input_image[i].copy()
            channels = img.shape[2] if len(img.shape) > 2 else 0

            if channels == 3:
                # RGB image - pass through without modification
                output_images.append(img)
            elif channels == 4:
                # RGBA image - process transparency and output RGB
                h, w = img.shape[:2]

                # Get alpha channel - either from image or from mask
                if mask is not None:
                    # Use mask's alpha values
                    # Mask: 0.0 = keep original (opaque), 1.0 = fill background (transparent)
                    # Convert to alpha: 0.0 = transparent, 1.0 = opaque
                    mask_np = mask[i].cpu().numpy()
                    alpha = 1.0 - mask_np  # Invert mask values for proper alpha interpretation
                else:
                    # Use image's alpha channel (0.0 = transparent, 1.0 = opaque)
                    alpha = img[:, :, 3] / 255.0

                # Create RGB result image (always 3 channels)
                result = np.zeros((h, w, 3), dtype=np.uint8)

                # For fully transparent pixels (alpha = 0), set to background color
                transparent_mask = alpha == 0.0
                result[transparent_mask] = bg_rgb

                # For semi-transparent pixels, blend with background color
                semi_transparent_mask = (alpha > 0.0) & (alpha < 1.0)

                if semi_transparent_mask.any():
                    # Blend: result = (original * alpha) + (background * (1 - alpha))
                    original_rgb = img[semi_transparent_mask, :3]
                    alpha_values = alpha[semi_transparent_mask]

                    # Expand alpha and bg_rgb for broadcasting
                    alpha_expanded = alpha_values[:, np.newaxis]  # Shape: [num_pixels, 1]
                    bg_rgb_expanded = np.array(bg_rgb)[np.newaxis, :]  # Shape: [1, 3]

                    # Perform blending
                    blended_rgb = original_rgb * alpha_expanded + bg_rgb_expanded * (1.0 - alpha_expanded)

                    # Set blended RGB values back
                    result[semi_transparent_mask] = blended_rgb.astype(np.uint8)

                # For fully opaque pixels, copy original RGB
                opaque_mask = alpha == 1.0
                result[opaque_mask] = img[opaque_mask, :3]

                output_images.append(result)
            else:
                # Unexpected number of channels - pass through
                output_images.append(img)

        # Convert back to tensor
        output_tensor = torch.from_numpy(np.stack(output_images) / 255.0).float()

        return (output_tensor,)


class SmartGetMask:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("MASK",)
    FUNCTION = "get_mask"
    CATEGORY = "SmartImageTools"

    def get_mask(self, image):
        # Convert from tensor to numpy array
        input_image = image.cpu().numpy()
        batch_size = input_image.shape[0]
        output_masks = []

        for i in range(batch_size):
            img = input_image[i]
            channels = img.shape[2] if len(img.shape) > 2 else 0

            if channels == 3:
                # RGB image - create pure white mask (all 1.0) representing opaque areas
                h, w = img.shape[:2]
                mask = np.ones((h, w), dtype=np.float32)
                output_masks.append(mask)
            elif channels == 4:
                # RGBA image - extract and invert alpha channel as mask
                # Alpha channel is 0.0 (transparent) to 1.0 (opaque)
                # Invert so transparent areas (0.0) become white (1.0) in mask
                # And opaque areas (1.0) become black (0.0) in mask
                alpha = img[:, :, 3]  # Alpha channel
                mask = 1.0 - alpha    # Invert the mask
                output_masks.append(mask)
            else:
                # Unexpected number of channels - create white mask as fallback
                h, w = img.shape[:2] if len(img.shape) >= 2 else (64, 64)  # Default size if unknown
                mask = np.ones((h, w), dtype=np.float32)
                output_masks.append(mask)

        # Convert to tensor with batch dimension [batch, height, width]
        output_tensor = torch.from_numpy(np.stack(output_masks))

        return (output_tensor,)


class SmartImagePadding:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "left": ("INT", {"default": 0, "min": 0, "max": 10000, "step": 1}),
                "right": ("INT", {"default": 0, "min": 0, "max": 10000, "step": 1}),
                "top": ("INT", {"default": 0, "min": 0, "max": 10000, "step": 1}),
                "bottom": ("INT", {"default": 0, "min": 0, "max": 10000, "step": 1}),
                "fill": (["Transparent", "Black", "White", "Grey", "Stretch"], {"default": "Transparent"}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "add_padding"
    CATEGORY = "SmartImageTools"

    def add_padding(self, image, left, right, top, bottom, fill):
        # Convert from tensor to numpy array
        input_image = 255. * image.cpu().numpy()
        batch_size = input_image.shape[0]
        output_images = []

        for i in range(batch_size):
            img = input_image[i].copy()
            h, w = img.shape[:2]
            channels = img.shape[2] if len(img.shape) > 2 else 0

            # Convert RGB to RGBA if needed
            if channels == 3:
                # Add alpha channel - fully opaque
                alpha_channel = np.full((h, w, 1), 255, dtype=np.uint8)
                img = np.concatenate([img, alpha_channel], axis=2)
            elif channels != 4:
                # Handle unexpected channel count - create RGBA with opaque alpha
                rgba_img = np.zeros((h, w, 4), dtype=np.uint8)
                if channels >= 3:
                    rgba_img[:, :, :3] = img[:, :, :3]
                else:
                    # Grayscale or other - replicate to RGB
                    rgba_img[:, :, :3] = img[:, :, 0] if channels >= 1 else 255
                rgba_img[:, :, 3] = 255  # Fully opaque
                img = rgba_img

            # Calculate new dimensions
            new_h = h + top + bottom
            new_w = w + left + right

            # Create new image with appropriate fill
            if fill == "Transparent":
                padded_img = np.zeros((new_h, new_w, 4), dtype=np.uint8)
            elif fill == "Black":
                padded_img = np.zeros((new_h, new_w, 4), dtype=np.uint8)
                padded_img[:, :, 3] = 255  # Fully opaque alpha
            elif fill == "White":
                padded_img = np.full((new_h, new_w, 4), 255, dtype=np.uint8)
            elif fill == "Grey":
                padded_img = np.full((new_h, new_w, 4), 128, dtype=np.uint8)
                padded_img[:, :, 3] = 255  # Fully opaque alpha
            elif fill == "Stretch":
                padded_img = np.zeros((new_h, new_w, 4), dtype=np.uint8)
                # Fill with stretched edge pixels
                # Top padding
                if top > 0:
                    for y in range(top):
                        padded_img[y, left:left + w] = img[0, :]
                # Bottom padding
                if bottom > 0:
                    for y in range(top + h, new_h):
                        padded_img[y, left:left + w] = img[h - 1, :]
                # Left padding
                if left > 0:
                    for x in range(left):
                        padded_img[top:top + h, x] = img[:, 0]
                # Right padding
                if right > 0:
                    for x in range(left + w, new_w):
                        padded_img[top:top + h, x] = img[:, w - 1]
                # Fill corners
                if top > 0 and left > 0:
                    padded_img[:top, :left] = img[0, 0]
                if top > 0 and right > 0:
                    padded_img[:top, left + w:] = img[0, w - 1]
                if bottom > 0 and left > 0:
                    padded_img[top + h:, :left] = img[h - 1, 0]
                if bottom > 0 and right > 0:
                    padded_img[top + h:, left + w:] = img[h - 1, w - 1]

            # Copy original image to the center (accounting for padding)
            padded_img[top:top + h, left:left + w] = img

            output_images.append(padded_img)

        # Convert back to tensor
        output_tensor = torch.from_numpy(np.stack(output_images) / 255.0).float()

        return (output_tensor,)


class SmartGradientDeformation:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "type": (["vertical", "horizontal"],),
                "start_pos": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "end_pos": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "offset": ("INT", {"default": 20, "min": -8192, "max": 8192, "step": 1}),
                "exponent": ("FLOAT", {"default": 1.0, "min": 0.01, "max": 10.0, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("IMAGE", "IMAGE")
    RETURN_NAMES = ("image", "gradient")
    FUNCTION = "deform"
    CATEGORY = "SmartImageTools"

    def deform(self, image, type, start_pos, end_pos, offset, exponent):
        if start_pos > end_pos:
            start_pos, end_pos = end_pos, start_pos

        batch_size, height, width, _ = image.shape
        
        deformed_images = []
        gradient_images = []

        for i in range(batch_size):
            img_tensor = image[i]
            img_np = img_tensor.cpu().numpy()

            if type == "vertical":
                size = height
            else:
                size = width
            
            start = int(start_pos * size)
            end = int(end_pos * size)

            grad_1d = np.zeros((size,), dtype=np.float32)

            if start < end:
                ramp = np.linspace(0, 1, num=end - start, dtype=np.float32)
                grad_1d[start:end] = offset * (ramp ** exponent)
            
            if end < size:
                grad_1d[end:] = offset

            map_x, map_y = np.meshgrid(np.arange(width, dtype=np.float32), np.arange(height, dtype=np.float32))

            if type == "vertical":
                displacement = grad_1d.reshape(-1, 1)
                map_y = map_y - displacement
                gradient_map = np.tile(displacement, (1, width))
            else:
                map_x = map_x - grad_1d
                gradient_map = np.tile(grad_1d, (height, 1))

            deformed_img_np = cv2.remap(img_np, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
            
            min_val, max_val = np.min(gradient_map), np.max(gradient_map)
            if max_val == min_val:
                 gradient_map_norm = np.zeros_like(gradient_map, dtype=np.float32)
            else:
                 gradient_map_norm = (gradient_map - min_val) / (max_val - min_val)
            
            gradient_img_np = np.stack([gradient_map_norm] * 3, axis=-1)
            
            deformed_images.append(torch.from_numpy(deformed_img_np))
            gradient_images.append(torch.from_numpy(gradient_img_np.astype(np.float32)))

        deformed_batch = torch.stack(deformed_images)
        gradient_batch = torch.stack(gradient_images)

        return (deformed_batch, gradient_batch)


class SmartColorMatch:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "target": ("IMAGE",),
                "reference": ("IMAGE",),
                "method": (
                    [   
                        'mkl',
                        'hm', 
                        'reinhard', 
                        'mvgd', 
                        'hm-mvgd-hm', 
                        'hm-mkl-hm',
                    ], {
                        "default": 'mkl'
                    }),
            },
            "optional": {
                "reference_end": ("IMAGE",),
                "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
                "multithread": ("BOOLEAN", {"default": True}),
            }
        }
    
    CATEGORY = "SmartImageTools"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "colormatch"
    DESCRIPTION = """
Smart Color Match with progressive reference blending.

If reference_end is not provided, works like standard color matching.
If reference_end is provided with a batch of target images, the reference 
gradually transitions from 'reference' to 'reference_end' across the sequence.

Based on color-matcher by hahnec:
https://github.com/hahnec/color-matcher/
"""
    
    def colormatch(self, target, reference, method, reference_end=None, strength=1.0, multithread=True):
        try:
            from color_matcher import ColorMatcher
        except:
            raise Exception("Can't import color-matcher, did you install requirements.txt? Manual install: pip install color-matcher")
        
        target = target.cpu()
        reference = reference.cpu()
        if reference_end is not None:
            reference_end = reference_end.cpu()
        
        batch_size = target.size(0)
        
        target_images = target.squeeze()
        reference_images = reference.squeeze()
        
        target_np = target_images.numpy()
        reference_np = reference_images.numpy()
        
        if reference_end is not None:
            reference_end_images = reference_end.squeeze()
            reference_end_np = reference_end_images.numpy()
        else:
            reference_end_np = None

        def process(i):
            cm = ColorMatcher()
            target_np_i = target_np if batch_size == 1 else target_images[i].numpy()
            
            # Store original alpha if present
            target_has_alpha = target_np_i.shape[-1] == 4
            if target_has_alpha:
                target_alpha = target_np_i[:, :, 3].copy()
                target_rgb = target_np_i[:, :, :3].copy()
                # Fill fully transparent pixels with black
                transparent_mask = target_alpha < 0.01
                target_rgb[transparent_mask] = 0.0
            else:
                target_rgb = target_np_i
            
            try:
                if reference_end_np is None:
                    # No blending - single color match
                    reference_np_i = reference_np if reference.size(0) == 1 else reference_images[i].numpy()
                    
                    # Handle reference alpha channel
                    ref_has_alpha = reference_np_i.shape[-1] == 4
                    if ref_has_alpha:
                        ref_alpha = reference_np_i[:, :, 3]
                        ref_rgb = reference_np_i[:, :, :3].copy()
                        # Fill fully transparent pixels with black
                        ref_transparent_mask = ref_alpha < 0.01
                        ref_rgb[ref_transparent_mask] = 0.0
                    else:
                        ref_rgb = reference_np_i
                    
                    # Perform color matching on RGB channels only
                    image_result = cm.transfer(src=target_rgb, ref=ref_rgb, method=method)
                    image_result = target_rgb + strength * (image_result - target_rgb)
                    image_result = np.clip(image_result, 0.0, 1.0)
                else:
                    # Progressive blending - do two color matches and blend results
                    if batch_size > 1:
                        progress = i / (batch_size - 1)
                    else:
                        progress = 0.0
                    
                    # Get reference images (use first frame if single frame reference)
                    ref1 = reference_np if reference.size(0) == 1 else reference_images[i if i < reference.size(0) else 0].numpy()
                    ref2 = reference_end_np if reference_end.size(0) == 1 else reference_end_images[i if i < reference_end.size(0) else 0].numpy()
                    
                    # Process ref1
                    ref1_has_alpha = ref1.shape[-1] == 4
                    if ref1_has_alpha:
                        ref1_alpha = ref1[:, :, 3]
                        ref1_rgb = ref1[:, :, :3].copy()
                        ref1_transparent_mask = ref1_alpha < 0.01
                        ref1_rgb[ref1_transparent_mask] = 0.0
                    else:
                        ref1_rgb = ref1
                    
                    # Process ref2
                    ref2_has_alpha = ref2.shape[-1] == 4
                    if ref2_has_alpha:
                        ref2_alpha = ref2[:, :, 3]
                        ref2_rgb = ref2[:, :, :3].copy()
                        ref2_transparent_mask = ref2_alpha < 0.01
                        ref2_rgb[ref2_transparent_mask] = 0.0
                    else:
                        ref2_rgb = ref2
                    
                    # Perform color matching with ref1
                    result1 = cm.transfer(src=target_rgb, ref=ref1_rgb, method=method)
                    result1 = target_rgb + strength * (result1 - target_rgb)
                    result1 = np.clip(result1, 0.0, 1.0)
                    
                    # Perform color matching with ref2
                    result2 = cm.transfer(src=target_rgb, ref=ref2_rgb, method=method)
                    result2 = target_rgb + strength * (result2 - target_rgb)
                    result2 = np.clip(result2, 0.0, 1.0)
                    
                    # Blend the two results based on progress
                    image_result = result1 * (1.0 - progress) + result2 * progress
                    image_result = np.clip(image_result, 0.0, 1.0)
                
                # If target had alpha, restore it
                if target_has_alpha:
                    result_with_alpha = np.zeros((image_result.shape[0], image_result.shape[1], 4), dtype=np.float32)
                    result_with_alpha[:, :, :3] = image_result
                    result_with_alpha[:, :, 3] = target_alpha
                    return torch.from_numpy(result_with_alpha)
                else:
                    return torch.from_numpy(image_result)
            except Exception as e:
                print(f"Frame {i} color match error: {e}")
                return torch.from_numpy(target_np_i)  # fallback

        if multithread and batch_size > 1:
            max_threads = min(os.cpu_count() or 1, batch_size)
            with ThreadPoolExecutor(max_workers=max_threads) as executor:
                out = list(executor.map(process, range(batch_size)))
        else:
            out = [process(i) for i in range(batch_size)]

        out = torch.stack(out, dim=0).to(torch.float32)
        out.clamp_(0, 1)
        return (out,)


class SmartImageCrop:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "measurement": (["pixels", "percent"],),
                "left": ("INT", {"default": 0, "min": -8192, "max": 8192, "step": 1}),
                "right": ("INT", {"default": 0, "min": -8192, "max": 8192, "step": 1}),
                "top": ("INT", {"default": 0, "min": -8192, "max": 8192, "step": 1}),
                "bottom": ("INT", {"default": 0, "min": -8192, "max": 8192, "step": 1}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "crop_image"
    CATEGORY = "SmartImageTools"

    def crop_image(self, image, measurement, left, right, top, bottom):
        batch_size = image.shape[0]
        output_images = []

        for i in range(batch_size):
            img_tensor = image[i]
            h, w = img_tensor.shape[:2]
            channels = img_tensor.shape[2]
            has_alpha = channels == 4

            # Convert percent to pixels if needed
            if measurement == "percent":
                left_px = int(round(w * left / 100.0))
                right_px = int(round(w * right / 100.0))
                top_px = int(round(h * top / 100.0))
                bottom_px = int(round(h * bottom / 100.0))
            else:  # pixels
                left_px = left
                right_px = right
                top_px = top
                bottom_px = bottom

            # Calculate new dimensions
            new_w = w - left_px - right_px
            new_h = h - top_px - bottom_px

            # Ensure dimensions are at least 1
            new_w = max(1, new_w)
            new_h = max(1, new_h)

            # Create output image
            if has_alpha:
                output_img = torch.zeros((new_h, new_w, 4), dtype=img_tensor.dtype, device=img_tensor.device)
            else:
                output_img = torch.zeros((new_h, new_w, 3), dtype=img_tensor.dtype, device=img_tensor.device)

            # Calculate source and destination regions
            # Source region in original image
            src_top = max(0, top_px)
            src_bottom = min(h, h - bottom_px)
            src_left = max(0, left_px)
            src_right = min(w, w - right_px)

            # Destination region in output image
            dst_top = max(0, -top_px)
            dst_left = max(0, -left_px)

            # Calculate how much we can actually copy
            copy_h = min(src_bottom - src_top, new_h - dst_top)
            copy_w = min(src_right - src_left, new_w - dst_left)

            # Ensure we don't copy negative amounts
            if copy_h > 0 and copy_w > 0:
                # Copy the overlapping region
                output_img[dst_top:dst_top+copy_h, dst_left:dst_left+copy_w] = \
                    img_tensor[src_top:src_top+copy_h, src_left:src_left+copy_w]

            output_images.append(output_img)

        # Stack all processed images
        output_batch = torch.stack(output_images)
        return (output_batch,)


class SmartFillTransparentHoles:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "target": ("IMAGE",),
                "source": ("IMAGE",),
                "min_hole_size": ("INT", {"default": 1, "min": 1, "max": 100000, "step": 1}),
                "max_hole_size": ("INT", {"default": 15, "min": 1, "max": 100000, "step": 1}),
                "threshold": ("FLOAT", {"default": 0.9, "min": 0.0, "max": 1.0, "step": 0.01}),
                "blend": ("BOOLEAN", {"default": True}),
                "ignore_left": ("INT", {"default": 0, "min": 0, "max": 8192, "step": 1}),
                "ignore_right": ("INT", {"default": 0, "min": 0, "max": 8192, "step": 1}),
                "ignore_top": ("INT", {"default": 0, "min": 0, "max": 8192, "step": 1}),
                "ignore_bottom": ("INT", {"default": 0, "min": 0, "max": 8192, "step": 1}),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("image", "filled_mask")
    FUNCTION = "fill_holes"
    CATEGORY = "SmartImageTools"
    
    def process_single_image(self, target_img, source_img, min_hole_size, max_hole_size, threshold, blend, 
                            ignore_left, ignore_right, ignore_top, ignore_bottom):
        """Process a single image - optimized with scipy connected components."""
        from scipy import ndimage
        
        h, w = target_img.shape[:2]
        
        # Resize source if dimensions don't match
        sh, sw = source_img.shape[:2]
        if sh != h or sw != w:
            source_img = cv2.resize(source_img, (w, h), interpolation=cv2.INTER_LINEAR)
        
        # Ensure source has same channel count as target
        if source_img.shape[2] != target_img.shape[2]:
            if target_img.shape[2] == 4 and source_img.shape[2] == 3:
                alpha = np.ones((h, w, 1), dtype=source_img.dtype)
                source_img = np.concatenate([source_img, alpha], axis=2)
            elif target_img.shape[2] == 3 and source_img.shape[2] == 4:
                source_img = source_img[:, :, :3]
        
        # Check if target has alpha channel
        if target_img.shape[2] == 4:
            # Copy input to output buffer
            output_img = target_img.copy()
            
            # Extract transparency mask (alpha channel)
            alpha = target_img[:, :, 3]
            
            # Threshold the transparency mask (0 = transparent, 1 = opaque)
            binary_mask = (alpha < threshold).astype(np.uint8)
            
            # Create processing area mask (areas not to be ignored)
            processing_mask = np.ones((h, w), dtype=bool)
            # Apply ignore margins
            if ignore_left > 0:
                processing_mask[:, :ignore_left] = False
            if ignore_right > 0:
                processing_mask[:, -ignore_right:] = False
            if ignore_top > 0:
                processing_mask[:ignore_top, :] = False
            if ignore_bottom > 0:
                processing_mask[-ignore_bottom:, :] = False
            
            # Create filled area mask
            filled_area_mask = np.zeros((h, w), dtype=np.float32)
            
            # Use scipy's label to find connected components (much faster than manual flood fill)
            # structure defines 4-connectivity
            structure = np.array([[0, 1, 0],
                                  [1, 1, 1],
                                  [0, 1, 0]], dtype=np.uint8)
            labeled, num_features = ndimage.label(binary_mask, structure=structure)
            
            if num_features > 0:
                # Get properties of each region efficiently
                # Calculate sizes
                sizes = ndimage.sum(binary_mask, labeled, range(1, num_features + 1))
                
                # Check which regions touch edges
                edge_mask = np.zeros((h, w), dtype=bool)
                edge_mask[0, :] = True
                edge_mask[-1, :] = True
                edge_mask[:, 0] = True
                edge_mask[:, -1] = True
                
                touches_edge = np.zeros(num_features + 1, dtype=bool)
                touches_ignore = np.zeros(num_features + 1, dtype=bool)
                
                for label_id in range(1, num_features + 1):
                    region_mask = (labeled == label_id)
                    # Check if region touches image edges
                    if np.any(region_mask & edge_mask):
                        touches_edge[label_id] = True
                    # Check if region overlaps with ignored areas
                    if np.any(region_mask & ~processing_mask):
                        touches_ignore[label_id] = True
                
                # Get source data
                if source_img.shape[2] == 4:
                    source_rgb = source_img[:, :, :3]
                    source_alpha = source_img[:, :, 3]
                else:
                    source_rgb = source_img
                    source_alpha = np.ones((h, w), dtype=np.float32)
                
                # Process each region
                for label_id in range(1, num_features + 1):
                    area_size = sizes[label_id - 1]
                    
                    # Skip if touches edge
                    if touches_edge[label_id]:
                        continue
                    
                    # Skip if touches ignored area
                    if touches_ignore[label_id]:
                        continue
                    
                    # Skip if out of size range
                    if area_size < min_hole_size or area_size > max_hole_size:
                        continue
                    
                    # Get mask for this region
                    fill_mask = (labeled == label_id)
                    
                    # Copy original alpha to filled area mask
                    filled_area_mask[fill_mask] = alpha[fill_mask]
                    
                    # Perform fill/blend in output buffer
                    if blend:
                        # Alpha blend mode
                        target_rgb = target_img[:, :, :3]
                        target_alpha = alpha
                        
                        # Expand dimensions for broadcasting
                        target_alpha_3d = target_alpha[:, :, np.newaxis]
                        
                        # Blend: result = target * target_alpha + source * (1 - target_alpha)
                        blended_rgb = target_rgb * target_alpha_3d + source_rgb * (1.0 - target_alpha_3d)
                        output_img[:, :, :3][fill_mask] = blended_rgb[fill_mask]
                        
                        # Composite alpha: new_alpha = target_alpha + source_alpha * (1 - target_alpha)
                        new_alpha = target_alpha + source_alpha * (1.0 - target_alpha)
                        output_img[:, :, 3][fill_mask] = new_alpha[fill_mask]
                    else:
                        # Direct copy mode
                        output_img[:, :, :3][fill_mask] = source_rgb[fill_mask]
                        output_img[:, :, 3][fill_mask] = source_alpha[fill_mask]
            
            # Ensure output is in valid range
            output_img = np.clip(output_img, 0.0, 1.0)
            return output_img, filled_area_mask
        else:
            # Target has no alpha channel, return as is
            target_img = np.clip(target_img, 0.0, 1.0)
            return target_img, np.zeros((h, w), dtype=np.float32)
    
    def fill_holes(self, target, source, min_hole_size, max_hole_size, threshold, blend, 
                   ignore_left, ignore_right, ignore_top, ignore_bottom):
        # Convert from tensor to numpy array
        target_np = target.cpu().numpy()
        source_np = source.cpu().numpy()
        
        batch_size = target_np.shape[0]
        
        # Process batch in parallel using ThreadPoolExecutor
        def process_batch_item(i):
            target_img = target_np[i].copy()
            
            # Get or use first source image
            if i < source_np.shape[0]:
                source_img = source_np[i].copy()
            else:
                source_img = source_np[0].copy()
            
            return self.process_single_image(target_img, source_img, min_hole_size, max_hole_size, threshold, blend,
                                            ignore_left, ignore_right, ignore_top, ignore_bottom)
        
        # Use multithreading for batch processing
        with ThreadPoolExecutor(max_workers=min(batch_size, os.cpu_count() or 1)) as executor:
            results = list(executor.map(process_batch_item, range(batch_size)))
        
        # Separate images and masks
        output_images = [result[0] for result in results]
        output_masks = [result[1] for result in results]
        
        # Convert back to tensors
        output_tensor = torch.from_numpy(np.stack(output_images)).float()
        mask_tensor = torch.from_numpy(np.stack(output_masks)).float()
        
        # Ensure values are in 0-1 range
        output_tensor = torch.clamp(output_tensor, 0.0, 1.0)
        mask_tensor = torch.clamp(mask_tensor, 0.0, 1.0)
        
        return (output_tensor, mask_tensor)


class SmartProgressiveScaleImage:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE",),
                "start_scale": ("FLOAT", {"default": 1.0, "min": 0.01, "max": 100.0, "step": 0.01}),
                "end_scale": ("FLOAT", {"default": 1.0, "min": 0.01, "max": 100.0, "step": 0.01}),
                "scale_method": (["nearest", "bilinear", "bicubic", "lanczos", "area"],),
                "curve_exponent": ("FLOAT", {"default": 1.0, "min": 0.01, "max": 10.0, "step": 0.01}),
                "end_offset_x": ("INT", {"default": 0, "min": -8192, "max": 8192, "step": 1}),
                "end_offset_y": ("INT", {"default": 0, "min": -8192, "max": 8192, "step": 1}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "progressive_scale"
    CATEGORY = "SmartImageTools"

    def progressive_scale(self, images, start_scale, end_scale, scale_method, curve_exponent, end_offset_x, end_offset_y):
        batch_size = images.shape[0]
        
        # Map scale method names to cv2 interpolation constants
        method_map = {
            "nearest": cv2.INTER_NEAREST,
            "bilinear": cv2.INTER_LINEAR,
            "bicubic": cv2.INTER_CUBIC,
            "lanczos": cv2.INTER_LANCZOS4,
            "area": cv2.INTER_AREA,
        }
        
        interp_method = method_map.get(scale_method, cv2.INTER_LINEAR)
        
        scaled_images = []
        image_offsets = []
        
        for i in range(batch_size):
            # Calculate progress through the batch (0.0 to 1.0)
            if batch_size > 1:
                progress = i / (batch_size - 1)
            else:
                progress = 0.0
            
            # Apply curve exponent to progress
            curved_progress = progress ** curve_exponent
            
            # Calculate scale factor for this image
            scale_factor = start_scale + (end_scale - start_scale) * curved_progress
            
            # Calculate offset for this image (progressively increase from 0 to end_offset)
            offset_x = int(round(end_offset_x * curved_progress))
            offset_y = int(round(end_offset_y * curved_progress))
            image_offsets.append((offset_x, offset_y))
            
            # Get current image
            img_tensor = images[i]
            img_np = img_tensor.cpu().numpy()
            
            # Convert to uint8 for cv2 processing
            img_np = np.clip(img_np, 0.0, 1.0)
            img_uint8 = (img_np * 255.0).astype(np.uint8)
            
            h, w = img_uint8.shape[:2]
            new_h = max(1, int(round(h * scale_factor)))
            new_w = max(1, int(round(w * scale_factor)))
            
            # Resize image
            if h == 0 or w == 0:
                # Handle empty image case
                scaled_img = np.zeros((new_h, new_w, img_uint8.shape[2]), dtype=np.uint8)
            else:
                scaled_img = cv2.resize(img_uint8, (new_w, new_h), interpolation=interp_method)
            
            # Handle potential shape issues after resize (e.g., grayscale losing channel dim)
            if len(scaled_img.shape) == 2 and len(img_uint8.shape) == 3:
                scaled_img = np.expand_dims(scaled_img, axis=-1)
                if img_uint8.shape[2] > 1:
                    scaled_img = scaled_img.repeat(img_uint8.shape[2], axis=-1)
            
            # Convert back to float [0, 1]
            scaled_img_float = scaled_img.astype(np.float32) / 255.0
            scaled_images.append(torch.from_numpy(scaled_img_float))
        
        # Check if we need offset logic or can use simpler logic
        if end_offset_x == 0 and end_offset_y == 0:
            # No offsets - use original logic: only pad if sizes differ, and center images
            shapes = [img.shape for img in scaled_images]
            if len(set(shapes)) == 1:
                # All images have the same shape, stack normally
                output_batch = torch.stack(scaled_images)
            else:
                # Images have different shapes - pad to max size and center
                max_h = max(img.shape[0] for img in scaled_images)
                max_w = max(img.shape[1] for img in scaled_images)
                channels = scaled_images[0].shape[2]
                
                padded_images = []
                for img in scaled_images:
                    h, w = img.shape[:2]
                    if h == max_h and w == max_w:
                        padded_images.append(img)
                    else:
                        # Create padded image
                        padded = torch.zeros((max_h, max_w, channels), dtype=img.dtype)
                        # Center the image
                        y_offset = (max_h - h) // 2
                        x_offset = (max_w - w) // 2
                        padded[y_offset:y_offset+h, x_offset:x_offset+w] = img
                        padded_images.append(padded)
                
                output_batch = torch.stack(padded_images)
        else:
            # Offsets are used - calculate canvas size and position images
            max_h = 0
            max_w = 0
            min_offset_x = 0
            min_offset_y = 0
            
            for img, (off_x, off_y) in zip(scaled_images, image_offsets):
                h, w = img.shape[:2]
                # Track the minimum offset (for negative offsets)
                min_offset_x = min(min_offset_x, off_x)
                min_offset_y = min(min_offset_y, off_y)
                # Calculate maximum canvas size needed
                max_w = max(max_w, w + off_x)
                max_h = max(max_h, h + off_y)
            
            # Adjust canvas size to account for negative offsets
            canvas_w = max_w - min_offset_x
            canvas_h = max_h - min_offset_y
            
            channels = scaled_images[0].shape[2]
            
            # Create padded/positioned images on the canvas
            positioned_images = []
            for img, (off_x, off_y) in zip(scaled_images, image_offsets):
                h, w = img.shape[:2]
                
                # Create canvas with transparent/black background
                canvas = torch.zeros((canvas_h, canvas_w, channels), dtype=img.dtype)
                
                # Calculate position on canvas (accounting for negative offsets)
                y_pos = off_y - min_offset_y
                x_pos = off_x - min_offset_x
                
                # Place image on canvas
                canvas[y_pos:y_pos+h, x_pos:x_pos+w] = img
                positioned_images.append(canvas)
            
            output_batch = torch.stack(positioned_images)
        
        return (output_batch,)


class SmartColorFillMask:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "mask": ("MASK",),
                "invert_mask": ("BOOLEAN", {"default": False}),
                "color": ("COLOR", {"default": "#FFFFFF"}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "fill_color"
    CATEGORY = "SmartImageTools"

    def hex_to_rgb(self, hex_color):
        """Convert hex color string to RGB tuple (0-255 range)."""
        hex_color = hex_color.lstrip('#')
        return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

    def fill_color(self, image, mask, invert_mask, color):
        # Convert from tensor to numpy array
        input_image = 255. * image.cpu().numpy()
        mask_np = mask.cpu().numpy()
        batch_size = input_image.shape[0]
        output_images = []

        # Convert hex color to RGB
        fill_rgb = self.hex_to_rgb(color)

        for i in range(batch_size):
            img = input_image[i].copy()
            h, w = img.shape[:2]
            channels = img.shape[2] if len(img.shape) > 2 else 0

            # Get mask for this image
            if i < mask_np.shape[0]:
                current_mask = mask_np[i]
            else:
                # If batch size mismatch, use last available mask
                current_mask = mask_np[-1]

            # Invert mask if requested
            # Mask: 1.0 = fill with color, 0.0 = keep original
            if invert_mask:
                current_mask = 1.0 - current_mask

            # Ensure mask matches image dimensions
            if current_mask.shape[0] != h or current_mask.shape[1] != w:
                # Resize mask to match image
                current_mask = cv2.resize(current_mask, (w, h), interpolation=cv2.INTER_LINEAR)

            # Create mask for broadcasting (h, w, 1)
            mask_3d = current_mask[:, :, np.newaxis]

            if channels == 3:
                # RGB image
                result = img.copy()
                # Apply color where mask is 1.0
                fill_color_array = np.array(fill_rgb, dtype=np.float32)
                result = img * (1.0 - mask_3d) + fill_color_array * mask_3d
                result = np.clip(result, 0, 255).astype(np.uint8)
                output_images.append(result)

            elif channels == 4:
                # RGBA image
                result = img.copy()
                # Apply color to RGB channels where mask is 1.0
                fill_color_array = np.array(fill_rgb, dtype=np.float32)
                result[:, :, :3] = img[:, :, :3] * (1.0 - mask_3d) + fill_color_array * mask_3d
                # Keep alpha channel unchanged
                result = np.clip(result, 0, 255).astype(np.uint8)
                output_images.append(result)

            else:
                # Unexpected number of channels - pass through
                output_images.append(img.astype(np.uint8))

        # Convert back to tensor (0-1 range)
        output_array = np.stack(output_images) / 255.0
        output_tensor = torch.from_numpy(output_array).float()

        return (output_tensor,)


NODE_CLASS_MAPPINGS = {
    "SmartImagePaletteConvert": SmartImagePaletteConvert,
    "SmartImagesProcessor": SmartImagesProcessor,
    "SmartGenerateImage": SmartGenerateImage,
    "SmartBackgroundRemove": SmartBackgroundRemove,
    "SmartPoint": SmartPoint,
    "SmartPointSet": SmartPointSet,
    "SmartPointSetMerge": SmartPointSetMerge,
    "SmartImagePreviewScaled": SmartImagePreviewScaled,
    "SmartPreviewPalette": SmartPreviewPalette,
    "SmartImagePoint": SmartImagePoint,
    "SmartImageRegion": SmartImageRegion,
    "SmartImagePaletteExtract": SmartImagePaletteExtract,
    "SmartSemiTransparenceRemove": SmartSemiTransparenceRemove,
    "SmartVideoPreviewScaled": SmartVideoPreviewScaled,
    "SmartVideoPreviewScaledMasked": SmartVideoPreviewScaledMasked,
    "SmartSaveAnimatedPNG": SmartSaveAnimatedPNG,
    "SmartSavePNG": SmartSavePNG,
    "SmartDrawPoints": SmartDrawPoints,
    "SmartLoadGIFImage": SmartLoadGIFImage,
    "SmartImagePaletteCreate": SmartImagePaletteCreate,
    "SmartBackgroundFill": SmartBackgroundFill,
    "SmartGetMask": SmartGetMask,
    "SmartImagePadding": SmartImagePadding,
    "SmartGradientDeformation": SmartGradientDeformation,
    "SmartColorMatch": SmartColorMatch,
    "SmartImageCrop": SmartImageCrop,
    "SmartFillTransparentHoles": SmartFillTransparentHoles,
    "SmartProgressiveScaleImage": SmartProgressiveScaleImage,
    "SmartColorFillMask": SmartColorFillMask
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SmartImagePaletteConvert": "Smart Image Palette Convert",
    "SmartImagesProcessor": "Smart Images Processor",
    "SmartGenerateImage": "Smart Generate Image",
    "SmartBackgroundRemove": "Smart Remove Background",
    "SmartPoint": "Smart Point",
    "SmartPointSet": "Smart Point Set",
    "SmartPointSetMerge": "Smart Point Set Merge",
    "SmartImagePreviewScaled": "Smart Image Preview Scaled",
    "SmartPreviewPalette": "Smart Preview Palette",
    "SmartImagePoint": "Smart Image Point",
    "SmartImageRegion": "Smart Image Region",
    "SmartImagePaletteExtract": "Smart Image Palette Extract",
    "SmartSemiTransparenceRemove": "Smart Semi-Transparence Remove",
    "SmartVideoPreviewScaled": "Smart Video Preview Scaled",
    "SmartVideoPreviewScaledMasked": "Smart Video Preview Scaled Masked",
    "SmartSaveAnimatedPNG": "Smart Save Animated Image",
    "SmartSavePNG": "Smart Save PNG",
    "SmartDrawPoints": "Smart Draw Points",
    "SmartLoadGIFImage": "Smart Load GIF Image",
    "SmartImagePaletteCreate": "Smart Image Palette Create",
    "SmartBackgroundFill": "Smart Background Fill",
    "SmartGetMask": "Smart Get Mask",
    "SmartImagePadding": "Smart Image Padding",
    "SmartGradientDeformation": "Smart Gradient Deformation",
    "SmartColorMatch": "Smart Color Match",
    "SmartImageCrop": "Smart Image Crop",
    "SmartFillTransparentHoles": "Smart Fill Transparent Holes",
    "SmartProgressiveScaleImage": "Smart Progressive Scale Image",
    "SmartColorFillMask": "Smart Color Fill Mask"
} 