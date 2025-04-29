# Smart Image Tools - Custom APNG Format Documentation

## Format Overview

The Smart Image Tools custom APNG (Animated PNG) format is a simple binary container format that stores a sequence of PNG images along with timing information. This format is designed to be easy to read and write, while maintaining the full quality of each individual PNG frame.

Unlike standard APNG files that follow the PNG specification extensions, this custom format uses a simpler structure optimized for storing image sequences created with ComfyUI.

## File Structure

The file consists of a header followed by a sequence of frame blocks:

```
[Header]
[Frame 1]
[Frame 2]
...
[Frame N]
```

### Header Structure (8 bytes total)

- Bytes 0-3: Total number of frames (4-byte signed integer, little-endian)
- Bytes 4-7: FPS value multiplied by 100 (4-byte signed integer, little-endian)
  - FPS is multiplied by 100 to store decimal precision without using floating point

### Frame Block Structure (variable length)

Each frame block has the following structure:

- Bytes 0-3: Length of the PNG data in bytes (4-byte signed integer, little-endian)
- Bytes 4-(4+length-1): Raw PNG file data

Frames are stored sequentially, with each frame's data directly following the previous frame's data.

## Reading the Format

To read this format:

1. Read the first 8 bytes to get frame count and FPS
2. For each frame:
   a. Read 4 bytes to get the PNG data length
   b. Read that many bytes to get the PNG data
   c. Decode the PNG data using any standard PNG decoder
   d. Repeat until all frames are read

Sample pseudocode for reading:

```python
def read_apng_file(file_path):
    with open(file_path, 'rb') as f:
        # Read header
        frame_count = struct.unpack('<i', f.read(4))[0]
        fps_int = struct.unpack('<i', f.read(4))[0]
        fps = fps_int / 100.0
        
        # Read frames
        frames = []
        for i in range(frame_count):
            length = struct.unpack('<i', f.read(4))[0]
            png_data = f.read(length)
            # Convert PNG data to image
            img = Image.open(io.BytesIO(png_data))
            frames.append(img)
            
        return frames, fps
```

## Using the Format

This format is intended for:

1. Storing animation sequences generated with ComfyUI
2. Easy playback in custom viewers
3. Efficient storage of lossless image sequences
4. Simple exchange format for workflows needing to process image sequences

## Limitations

- Not a standard format, requires custom code to read/write
- Does not support variable frame timing (all frames use the same FPS)
- No compression between frames (each PNG is stored independently)
- Maximum 2,147,483,647 frames (signed 32-bit integer limit)
- Maximum FPS value: 21,474,836.47 (due to int storage × 100)

## Differences from Standard APNG

This format differs from the official APNG format:
- Much simpler structure (no chunk-based organization)
- No backward compatibility with PNG viewers
- No frame-specific delay times (uses global FPS instead)
- No support for blending modes between frames

## Tool Usage

The format can be created using the "Smart Save Animated PNG" node in ComfyUI with the following inputs:
- images: Batch of image frames to save
- fps: Desired playback speed
- filename_prefix: Base name for the output file
- lossless: Whether to use maximum lossless compression 