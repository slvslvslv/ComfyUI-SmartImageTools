import numpy as np
from PIL import Image
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
        if transparent_idx != -1:
             transparent_color = palette[transparent_idx].astype(np.float32)
             for y in numba.prange(height): # Use prange for potential parallelization
                 for x in range(width):
                     if alpha_mask[y, x]:
                         result[y, x] = transparent_color
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
            new_pixel = palette[closest_idx].astype(np.float32) # Use full palette (incl. alpha)

            # Update the pixel in the result image
            result[y, x] = new_pixel

            # Calculate error (only for RGB channels)
            # Alpha error is not typically diffused in dithering
            error = old_pixel - new_pixel

            # Distribute the error
            for i in range(len(err_weights_dx)):
                nx, ny = x + err_weights_dx[i], y + err_weights_dy[i]

                # Check bounds
                if 0 <= nx < width and 0 <= ny < height:
                     # Don't distribute error to transparent pixels
                    if not (has_alpha and alpha_mask[ny, nx]):
                         result[ny, nx] += error * err_weights_val[i]

    # Clip values at the end
    # Numba doesn't directly support np.clip with min/max arrays easily across dimensions
    # So we clip per channel
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
                "dithering_amount": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05}),
            },
            "optional": {
                "reference_image": ("IMAGE",),
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
            unique_colors = np.unique(rgb_opaque[:, :3], axis=0)

            # Create final palette with alpha channel
            palette = np.zeros((len(unique_colors) + 1, 4), dtype=np.uint8)
            palette[:len(unique_colors), :3] = unique_colors
            palette[:len(unique_colors), 3] = 255  # Full opacity
            palette[-1] = [0, 0, 0, 0]  # Add transparent color
        else:
            # No alpha channel, just get unique RGB colors
            unique_colors = np.unique(img_flat, axis=0)
            palette = unique_colors

        return self._sort_palette(palette)

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

        # Enhance vibrance - Find max intensity in original image and adjust palette
        max_intensity_orig = np.max(rgb, axis=0)
        max_intensity_palette = np.max(centers_rgb, axis=0)

        # Only increase intensity, don't decrease it
        scale_factors = np.maximum(max_intensity_orig / np.maximum(max_intensity_palette, 1), 1)

        # Apply scaling while keeping colors within valid range (vectorized)
        scaled = centers_rgb * scale_factors[np.newaxis, :]
        centers_rgb = np.clip(scaled, 0, 255).astype(np.uint8)

        # Add transparency color if needed
        if has_transparency:
            transparent_color = np.array([0, 0, 0, 0], dtype=np.uint8)
            if img.shape[2] == 4:
                # Create palette with RGBA values
                palette = np.zeros((colors_to_extract + 1, 4), dtype=np.uint8)
                palette[:-1, :3] = centers_rgb
                palette[:-1, 3] = 255  # Full opacity for color entries
                palette[-1] = transparent_color  # Last entry is transparent
            else:
                # Create palette with RGB values
                palette = np.zeros((colors_to_extract + 1, 3), dtype=np.uint8)
                palette[:-1] = centers_rgb
                palette[-1] = transparent_color[:3]  # Last entry is "transparent" (black)
        else:
            # No transparency needed
            if img.shape[2] == 4:
                palette = np.zeros((colors_to_extract, 4), dtype=np.uint8)
                palette[:, :3] = centers_rgb
                palette[:, 3] = 255
            else:
                palette = centers_rgb

        return self._sort_palette(palette)

    def apply_floyd_steinberg_dithering(self, img, palette, dithering_amount=1.0):
        """Apply Floyd-Steinberg dithering with variable amount and optimizations."""
        if dithering_amount <= 0:
            # If no dithering, just do direct color mapping (already vectorized)
            has_alpha = img.shape[2] == 4
            img_flat = img.reshape(-1, img.shape[2])

            # Determine the transparent color index from the palette
            transparent_idx = -1
            if has_alpha:
                for i, color in enumerate(palette):
                    if color[3] == 0:
                        transparent_idx = i
                        break
                if transparent_idx == -1:
                    transparent_idx = len(palette) -1 # Default if not found (shouldn't happen if generated correctly)

            if has_alpha:
                # Handle transparent pixels
                mask = img_flat[:, 3] < 1

                # Find closest color for non-transparent pixels
                # Use palette without alpha for distance calculation
                palette_rgb = palette[:, :3]
                closest_colors, _ = SmartImagePaletteConvert.find_closest_color_vectorized(img_flat[~mask, :3], palette_rgb)

                # Create result array
                result_flat = np.zeros_like(img_flat)

                # Set non-transparent pixels (reconstruct RGBA using palette alpha)
                mapped_rgba = palette[_, :][closest_colors] # Get full RGBA from palette indices
                result_flat[~mask] = mapped_rgba

                # Set transparent pixels to the palette's transparent color
                result_flat[mask] = palette[transparent_idx]
            else:
                # No transparency - simple mapping
                closest_colors, _ = SmartImagePaletteConvert.find_closest_color_vectorized(img_flat, palette)
                # result_flat = closest_colors # This was wrong, needs indexing
                result_flat = palette[closest_colors]


            return result_flat.reshape(img.shape)

        # --- Use Numba for dithering > 0 ---
        has_alpha = img.shape[2] == 4
        img_float = img.astype(np.float32)

        # Create a palette view without alpha for color comparisons if needed
        palette_rgb = palette[:, :3].copy() # Numba needs contiguous arrays sometimes

        # Pre-calculate transparent index if needed
        transparent_idx = -1
        if has_alpha:
            for i, color in enumerate(palette):
                if color[3] == 0:
                    transparent_idx = i
                    break
            if transparent_idx == -1 and len(palette) > 0:
                 # If not explicitly found, assume last entry might be transparent (fallback)
                 # Or handle error if transparency expected but not found
                 # For safety, we might default to last index if generated palette ensures it.
                 # Check if last entry IS transparent, otherwise no transparent color available
                 if palette[-1][3] == 0:
                     transparent_idx = len(palette) - 1
                 # else: transparent_idx remains -1, Numba func should handle

        # Ensure palette is contiguous float32 for Numba function
        palette_float = palette.astype(np.float32)

        # Call the Numba JIT function
        result_dithered = _apply_floyd_steinberg_dithering_numba(
            img_float,
            palette_float,
            palette_rgb.astype(np.float32), # Pass RGB part for distance calc
            dithering_amount,
            has_alpha,
            transparent_idx
        )

        return result_dithered

    def process_image(self, image, palette, dithering_amount):
        """Process a single image with the given palette."""
        return self.apply_floyd_steinberg_dithering(image, palette, dithering_amount)

    def convert_image(self, image, num_colors, dithering_amount, reference_image=None):
        # Convert from tensor to numpy array
        input_image = 255. * image.cpu().numpy()

        # Process reference image if provided
        if reference_image is not None:
            reference_np = 255. * reference_image.cpu().numpy()
            # Extract exact palette from reference image, ignoring num_colors
            palette = self.extract_palette_from_reference(reference_np[0])
        else:
            # Generate optimal palette from the input image using num_colors
            palette = self.generate_palette(input_image[0], num_colors)

        # Create output array with same batch size as input
        batch_size = input_image.shape[0]

        # Use ThreadPoolExecutor for parallel processing of batch images
        with ThreadPoolExecutor() as executor:
            futures = [
                executor.submit(self.process_image, input_image[i], palette, dithering_amount)
                for i in range(batch_size)
            ]

            output_images = [future.result() for future in futures]

        # Convert back to tensor
        output_tensor = torch.from_numpy(np.stack(output_images) / 255.0).float()

        # Create palette image (1 pixel high, with each color in the **sorted** palette)
        palette_height = 1
        palette_width = len(palette)
        # Create a NumPy array for the palette image
        palette_img = np.zeros((palette_height, palette_width, palette.shape[1]), dtype=np.uint8)

        # Fill each column with a color from the palette
        for i in range(palette_width):
            palette_img[:, i] = palette[i]

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
                           "gray", "orange", "purple", "brown", "pink", "transparent"],),
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
                # Allow both FLOAT and INT for scale_by
                "scale_by": ("FLOAT,INT", {"default": 1.0, "min": 0.01, "max": 100.0, "step": 0.1}),
            },
            "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
        }

    RETURN_TYPES = ()
    FUNCTION = "execute" # Use execute which then calls save_images internally
    OUTPUT_NODE = True
    CATEGORY = "SmartImageTools"

    # Use execute to potentially modify images before calling the parent's save logic
    def execute(self, images, scale_by=1.0, filename_prefix="SmartScaledPreview", prompt=None, extra_pnginfo=None):
        if abs(scale_by - 1.0) < 1e-6 or scale_by <= 0:
            # No scaling or invalid scale factor, use original save_images
            return self.save_images(images, filename_prefix, prompt, extra_pnginfo)

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
        return self.save_images(scaled_images_batch, filename_prefix, prompt, extra_pnginfo)


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
                "type": (["Rectangle"],), # Only Rectangle for now, more to be added later
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
            if not isinstance(region, dict):
                region = {"x1": 0.25, "y1": 0.25, "x2": 0.75, "y2": 0.75}
        except Exception as e:
            print(f"Error parsing region_data: {e}")
            region = {"x1": 0.25, "y1": 0.25, "x2": 0.75, "y2": 0.75}

        # 3. Generate mask image
        # Create a binary mask where the region is 1.0 and everywhere else is 0.0
        # Standard MASK format is [batch, height, width] (no channels dimension)
        mask_np = np.zeros((h, w), dtype=np.float32)

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

        # Convert to tensor with batch dimension [batch, height, width]
        mask_tensor = torch.from_numpy(mask_np).unsqueeze(0)

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

        # Return results for the workflow graph AND data for the UI
        return {"result": (mask_tensor, bbox), "ui": ui_data}


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
} 