import { app } from "../../../scripts/app.js";

// Image handling utilities for SmartImageTools nodes
export class SmartImage {
    constructor() {
        this.previewImage = null;
        this.img = null;
        this.dimensions = null;
        this.imageDrawRect = null;
    }

    // Load and store image data
    loadImage(imageData) {
        let newImageLoaded = false;
        
        if (imageData && imageData.startsWith("data:image/png;base64,")) {
            if (this.previewImage !== imageData) {
                this.previewImage = imageData;
                newImageLoaded = true;
                if (this.img) this.img.src = this.previewImage;
            }
        } else {
            this.previewImage = null;
        }
        
        return newImageLoaded;
    }

    // Store dimensions data
    setDimensions(dimensions) {
        if (dimensions) {
            this.dimensions = dimensions;
        } else {
            this.dimensions = null;
        }
    }

    // Draw image to canvas
    drawImage(ctx, x, y, width, height) {
        const margin = 5;
        const imageTopMargin = 10;
        const imageBottomMargin = 5;
        const titleHeight = LiteGraph.NODE_TITLE_HEIGHT || 20;
        
        const availableWidth = width - 2 * margin;
        const availableHeight = height - 2 * margin - titleHeight - imageBottomMargin - imageTopMargin;
        
        if (availableHeight <= 0 || availableWidth <= 0) return null;
        
        // Use the provided x, y coordinates and add margins
        let drawX = x + margin;
        let drawY = y + margin + titleHeight + imageTopMargin;
        let drawWidth = 0;
        let drawHeight = 0;
        
        // Draw Image Preview
        if (this.previewImage) {
            if (!this.img) {
                this.img = new Image();
                this.img.onload = () => { 
                    if (ctx.canvas.liteGraph?.setDirtyCanvas) {
                        ctx.canvas.liteGraph.setDirtyCanvas(true, false);
                    }
                };
                this.img.onerror = () => { 
                    this.previewImage = null; 
                    this.img = null; 
                    if (ctx.canvas.liteGraph?.setDirtyCanvas) {
                        ctx.canvas.liteGraph.setDirtyCanvas(true, false);
                    }
                };
            }
            
            if (this.img.src !== this.previewImage) {
                this.img.src = this.previewImage;
            }
            
            if (this.img.complete && this.img.naturalWidth > 0) {
                const imgWidth = this.img.naturalWidth;
                const imgHeight = this.img.naturalHeight;
                const aspectRatio = imgWidth / imgHeight;
                
                drawWidth = availableWidth;
                drawHeight = drawWidth / aspectRatio;
                if (drawHeight > availableHeight) {
                    drawHeight = availableHeight;
                    drawWidth = drawHeight * aspectRatio;
                }
                
                // Center the image horizontally within available space, but use the provided y-offset
                drawX = x + margin + (availableWidth - drawWidth) / 2;
                drawY = y + margin + titleHeight + imageTopMargin + (availableHeight - drawHeight) / 2;
                
                try {
                    ctx.drawImage(this.img, drawX, drawY, drawWidth, drawHeight);
                    this.imageDrawRect = { x: drawX, y: drawY, w: drawWidth, h: drawHeight };
                    return this.imageDrawRect;
                } catch (error) {
                    console.error("Error drawing image:", error);
                    this.previewImage = null;
                    this.img = null;
                    this.imageDrawRect = null;
                }
            }
        } else {
            // No image available - draw "No image" text
            ctx.fillStyle = "#999";
            ctx.font = "14px Arial";
            ctx.textAlign = "center";
            ctx.textBaseline = "middle";
            
            // Draw text in center of available image area, using provided y-offset
            const centerX = x + margin + (availableWidth / 2);
            const centerY = y + margin + titleHeight + imageTopMargin + (availableHeight / 2);
            ctx.fillText("No image", centerX, centerY);
            
            // Reset text alignment for later
            ctx.textAlign = "left";
            ctx.textBaseline = "alphabetic";
        }
        
        this.imageDrawRect = null;
        return null;
    }
    
    // Draw dimensions text
    drawDimensions(ctx, x, y, nodeWidth) {
        if (this.dimensions) {
            const text = `${this.dimensions[0]}x${this.dimensions[1]}`;
            ctx.fillStyle = "#CCC";
            ctx.font = `10px Arial`;
            ctx.textAlign = "right";
            ctx.fillText(text, nodeWidth - 5, y);
        }
    }
    
    // Helper function to check if point is inside rectangle
    pointInRect(point, rect) {
        return point.x >= rect.x && point.x <= rect.x + rect.w &&
               point.y >= rect.y && point.y <= rect.y + rect.h;
    }
} 