import { app } from "../../../scripts/app.js";
import { $el } from "../../../scripts/ui.js"; // Import $el for creating elements if needed
import { SmartImage } from "./image.js"; // Import our SmartImage class

// Displays an image preview directly within the node with a draggable rectangle region
app.registerExtension({
    name: "Comfy.SmartNodes.SmartImageRegion",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name === "SmartImageRegion") {

            // Find the region_data input widget
            function getRegionWidget(node) {
                return node.widgets.find(w => w.name === "region_data");
            }

            // Initialize region properties
            const initProperties = (ctx) => {
                if (!ctx.properties) {
                    ctx.properties = {};
                }
                
                // Get the hidden widget
                const regionWidget = getRegionWidget(ctx);
                let region = { x1: 0.25, y1: 0.25, x2: 0.75, y2: 0.75 };
                
                if (regionWidget && regionWidget.value) {
                    // Parse region from the widget value
                    try {
                        const parsed = JSON.parse(regionWidget.value);
                        if (typeof parsed === 'object' && parsed !== null) {
                            region = {
                                x1: parsed.x1 !== undefined ? parseFloat(parsed.x1) : 0.25,
                                y1: parsed.y1 !== undefined ? parseFloat(parsed.y1) : 0.25,
                                x2: parsed.x2 !== undefined ? parseFloat(parsed.x2) : 0.75,
                                y2: parsed.y2 !== undefined ? parseFloat(parsed.y2) : 0.75
                            };
                        }
                    } catch (e) {
                        console.error("Error parsing region_data:", e);
                    }
                }
                
                // Store region in properties
                ctx.properties.region = region;
                
                // Update widget with initial value
                if (regionWidget) {
                    regionWidget.value = JSON.stringify(region);
                }
                
                ctx.dragCorner = null; // Which corner is being dragged: null, "tl", "tr", "bl", "br"
                ctx.isDragging = false;
                ctx.resetButtonRect = null;
                
                // Initialize SmartImage
                ctx.smartImage = new SmartImage();
            };

            // Function to update widget when region changes
            const updateRegionWidget = (node) => {
                if (!node.properties.region) return;
                
                const regionWidget = getRegionWidget(node);
                if (regionWidget) {
                    // Convert y coordinates to bottom-left system (ComfyUI convention)
                    // No need to flip here as we do it in the drawing logic
                    regionWidget.value = JSON.stringify(node.properties.region);
                    
                    // Notify ComfyUI that a parameter changed
                    if (regionWidget.callback) {
                        regionWidget.callback(regionWidget.value);
                    }
                }
            };

            // Initialize on node creation
            const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function() {
                const result = onNodeCreated?.apply(this, arguments);
                initProperties(this);
                return result;
            };

            // Store image data and dimensions
            const onExecuted = nodeType.prototype.onExecuted;
            nodeType.prototype.onExecuted = function (message) {
                onExecuted?.apply(this, arguments);
                let newImageLoaded = false;
                
                if (message?.images && message?.images[0]) {
                    newImageLoaded = this.smartImage.loadImage(message.images[0]);
                } else {
                    this.smartImage.loadImage(null);
                }
                
                if (message?.dimensions && message?.dimensions[0]) {
                    this.smartImage.setDimensions(message.dimensions[0]);
                } else {
                    this.smartImage.setDimensions(null);
                }
                
                // Check if region_data is provided from the backend
                if (message?.region_data) {
                    try {
                        // Sanitize input before parsing
                        const regionDataString = String(message.region_data).trim();
                        
                        // Skip invalid or empty JSON
                        if (!regionDataString || regionDataString === "null" || regionDataString === "undefined") {
                            console.warn("Empty region_data received");
                        } else {
                            const regionFromBackend = JSON.parse(regionDataString);
                            
                            // Verify it's a valid object with required properties
                            if (typeof regionFromBackend === 'object' && regionFromBackend !== null) {
                                // Validate and ensure all required properties exist
                                const validRegion = {
                                    x1: 'x1' in regionFromBackend ? parseFloat(regionFromBackend.x1) : 0.25,
                                    y1: 'y1' in regionFromBackend ? parseFloat(regionFromBackend.y1) : 0.25,
                                    x2: 'x2' in regionFromBackend ? parseFloat(regionFromBackend.x2) : 0.75,
                                    y2: 'y2' in regionFromBackend ? parseFloat(regionFromBackend.y2) : 0.75
                                };
                                
                                // Ensure all values are in valid range
                                validRegion.x1 = Math.max(0, Math.min(1, validRegion.x1));
                                validRegion.y1 = Math.max(0, Math.min(1, validRegion.y1));
                                validRegion.x2 = Math.max(0, Math.min(1, validRegion.x2));
                                validRegion.y2 = Math.max(0, Math.min(1, validRegion.y2));
                                
                                // Update region properties
                                if (newImageLoaded || JSON.stringify(this.properties.region) !== JSON.stringify(validRegion)) {
                                    this.properties.region = validRegion;
                                }
                            } else {
                                console.warn("region_data is not a valid object:", regionFromBackend);
                            }
                        }
                    } catch (e) {
                        console.error("Error parsing region_data from backend:", e, "\nRaw value:", message.region_data);
                    }
                }
                
                this.setDirtyCanvas(true, true);
            };

            const onDrawForeground = nodeType.prototype.onDrawForeground;
            nodeType.prototype.onDrawForeground = function (ctx) {
                onDrawForeground?.apply(this, arguments);

                // Ensure properties are initialized if they were somehow lost
                if (!this.properties || !this.properties.region || !this.smartImage) {
                    initProperties(this);
                }

                const margin = 5;
                const titleHeight = LiteGraph.NODE_TITLE_HEIGHT || 20;
                const bottomBarHeight = 16; // Height for buttons and text
                const buttonWidth = 60; // Larger button
                const textFontSize = 10;
                const nodeWidth = this.size[0];
				const nodeHeight = this.size[1];
				const imageTopMargin = 30;
                
                // Get canvas zoom and adjust handle size accordingly
                const baseHandleSize = 5;
				const canvasScale = 1;  //app.canvas?.ds?.scale || 1;
                const handleSize = baseHandleSize / canvasScale;
                
                // Draw the image using SmartImage
                const imageRect = this.smartImage.drawImage(ctx, 0, imageTopMargin, nodeWidth, nodeHeight - bottomBarHeight - imageTopMargin);
                
                if (imageRect) {
                    // Get normalized region coordinates
                    const region = this.properties.region;
                    
                    // Convert normalized coordinates to canvas coordinates
                    const x1Canvas = imageRect.x + region.x1 * imageRect.w;
                    const y1Canvas = imageRect.y + region.y1 * imageRect.h;
                    const x2Canvas = imageRect.x + region.x2 * imageRect.w;
                    const y2Canvas = imageRect.y + region.y2 * imageRect.h;
                    
                    // Draw rectangle outline
                    ctx.strokeStyle = "rgba(0, 255, 255, 0.8)"; // Cyan color
                    ctx.lineWidth = 1;
                    ctx.beginPath();
                    ctx.rect(
                        Math.min(x1Canvas, x2Canvas), 
                        Math.min(y1Canvas, y2Canvas),
                        Math.abs(x2Canvas - x1Canvas),
                        Math.abs(y2Canvas - y1Canvas)
                    );
                    ctx.stroke();
                    
                    // Draw semi-transparent fill
                    ctx.fillStyle = "rgba(0, 255, 255, 0.2)"; // Cyan with transparency
                    ctx.fill();
                    
                    // Draw corner handles (squares)
                    const cornerPoints = [
                        { x: x1Canvas, y: y1Canvas, id: "tl" }, // Top-left
                        { x: x2Canvas, y: y1Canvas, id: "tr" }, // Top-right
                        { x: x1Canvas, y: y2Canvas, id: "bl" }, // Bottom-left
                        { x: x2Canvas, y: y2Canvas, id: "br" }  // Bottom-right
                    ];
                    
                    ctx.fillStyle = "rgba(255, 255, 255, 0.9)"; // White fill
                    ctx.strokeStyle = "rgba(0, 0, 0, 0.9)"; // Black outline
                    ctx.lineWidth = 0.5;
                    
                    for (const corner of cornerPoints) {
                        // Draw corner handle (centered on corner point)
                        ctx.beginPath();
                        ctx.rect(
                            corner.x - handleSize/2, 
                            corner.y - handleSize/2,
                            handleSize,
                            handleSize
                        );
                        ctx.fill();
                        ctx.stroke();
                    }
                }

                // Draw Bottom Bar with reset button
                const bottomY = nodeHeight - margin - bottomBarHeight;
                
                // Display region dimensions as text
                ctx.fillStyle = "#CCC";
                ctx.font = `${textFontSize}px Arial`;
                ctx.textAlign = "left";
                const regionWidth = Math.abs(this.properties.region.x2 - this.properties.region.x1);
                const regionHeight = Math.abs(this.properties.region.y2 - this.properties.region.y1);
                let displayText = `Region: ${(regionWidth*100).toFixed(0)}% × ${(regionHeight*100).toFixed(0)}%`;
                if (this.smartImage.dimensions) {
                    const pixelWidth = Math.round(regionWidth * this.smartImage.dimensions[0]);
                    const pixelHeight = Math.round(regionHeight * this.smartImage.dimensions[1]);
                    displayText += ` (${pixelWidth}×${pixelHeight}px)`;
                }
                ctx.fillText(displayText, margin, bottomY + bottomBarHeight/2 + textFontSize/3);
                
                // "Reset" Button (right aligned)
                this.resetButtonRect = { 
                    x: nodeWidth - margin - buttonWidth, 
                    y: bottomY, 
                    w: buttonWidth, 
                    h: bottomBarHeight 
                };
                ctx.fillStyle = "#444"; // Button background
                ctx.fillRect(this.resetButtonRect.x, this.resetButtonRect.y, this.resetButtonRect.w, this.resetButtonRect.h);
                ctx.fillStyle = "#CCC";
                ctx.textAlign = "center";
                ctx.fillText("Reset", this.resetButtonRect.x + buttonWidth/2, bottomY + bottomBarHeight/2 + textFontSize/3);
            };

            nodeType.prototype.onMouseDown = function(e, localPos, graphCanvas) {
                if (!e.isPrimary) return false; // Only left click

                const clickPos = { x: localPos[0], y: localPos[1] };

                // Check Reset Button Click
                if (this.resetButtonRect && this.smartImage.pointInRect(clickPos, this.resetButtonRect)) {
                    // Reset to default region (centered at 50% of width/height)
                    this.properties.region = { x1: 0.25, y1: 0.25, x2: 0.75, y2: 0.75 };
                    updateRegionWidget(this);
                    this.setDirtyCanvas(true, true);
                    return true; // Event handled
                }

                // Check for corner dragging
                if (this.smartImage.imageDrawRect) {
                    const imageRect = this.smartImage.imageDrawRect;
                    const region = this.properties.region;
                    
                    // Convert normalized coordinates to canvas coordinates
                    const x1Canvas = imageRect.x + region.x1 * imageRect.w;
                    const y1Canvas = imageRect.y + region.y1 * imageRect.h;
                    const x2Canvas = imageRect.x + region.x2 * imageRect.w;
                    const y2Canvas = imageRect.y + region.y2 * imageRect.h;
                    
                    // Define corner hit areas with zoom adjustment
                    const baseHandleSize = 12; // Slightly larger than visual size for easier clicking
                    const canvasScale = app.canvas?.ds?.scale || 1;
                    const handleSize = baseHandleSize / canvasScale;
                    
                    const cornerPoints = [
                        { x: x1Canvas, y: y1Canvas, id: "tl" }, // Top-left
                        { x: x2Canvas, y: y1Canvas, id: "tr" }, // Top-right
                        { x: x1Canvas, y: y2Canvas, id: "bl" }, // Bottom-left
                        { x: x2Canvas, y: y2Canvas, id: "br" }  // Bottom-right
                    ];
                    
                    // Check if click is on a corner handle
                    for (const corner of cornerPoints) {
                        const dx = clickPos.x - corner.x;
                        const dy = clickPos.y - corner.y;
                        if (Math.abs(dx) <= handleSize/2 && Math.abs(dy) <= handleSize/2) {
                            this.isDragging = true;
                            this.dragCorner = corner.id;
                            this.setDirtyCanvas(true, false);
                            return true; // Event handled
                        }
                    }
                    
                    // Check if click is inside rectangle (for moving the entire rectangle)
                    const rectLeft = Math.min(x1Canvas, x2Canvas);
                    const rectTop = Math.min(y1Canvas, y2Canvas);
                    const rectWidth = Math.abs(x2Canvas - x1Canvas);
                    const rectHeight = Math.abs(y2Canvas - y1Canvas);
                    
                    if (clickPos.x >= rectLeft && clickPos.x <= rectLeft + rectWidth &&
                        clickPos.y >= rectTop && clickPos.y <= rectTop + rectHeight) {
                        // Store drag start position and region state for move operation
                        this.isDragging = true;
                        this.dragCorner = "move";
                        this.dragStartPos = { x: clickPos.x, y: clickPos.y };
                        this.dragStartRegion = { ...this.properties.region };
                        this.setDirtyCanvas(true, false);
                        return true; // Event handled
                    }
                }

                return false; // Event not handled
            };

            nodeType.prototype.onMouseMove = function(e, localPos, graphCanvas) {
                if (!this.isDragging || !this.dragCorner || !this.smartImage.imageDrawRect) return false;

                const rect = this.smartImage.imageDrawRect;
                
                // Convert mouse position to normalized image coordinates
                const mouseNormX = Math.max(0, Math.min(1, (localPos[0] - rect.x) / rect.w));
                const mouseNormY = Math.max(0, Math.min(1, (localPos[1] - rect.y) / rect.h));
                
                // Get current region
                const region = this.properties.region;
                
                if (this.dragCorner === "move") {
                    // Moving the entire rectangle
                    // Calculate the delta in canvas coordinates
                    const deltaCanvasX = localPos[0] - this.dragStartPos.x;
                    const deltaCanvasY = localPos[1] - this.dragStartPos.y;
                    
                    // Convert to normalized coordinates
                    const deltaNormX = deltaCanvasX / rect.w;
                    const deltaNormY = deltaCanvasY / rect.h;
                    
                    // Original width and height
                    const width = this.dragStartRegion.x2 - this.dragStartRegion.x1;
                    const height = this.dragStartRegion.y2 - this.dragStartRegion.y1;
                    
                    // Calculate new position ensuring it stays within bounds
                    let newX1 = Math.max(0, Math.min(1 - width, this.dragStartRegion.x1 + deltaNormX));
                    let newY1 = Math.max(0, Math.min(1 - height, this.dragStartRegion.y1 + deltaNormY));
                    
                    // Update all corners preserving width/height
                    region.x1 = newX1;
                    region.y1 = newY1;
                    region.x2 = newX1 + width;
                    region.y2 = newY1 + height;
                } else {
                    // Dragging a corner
                    switch (this.dragCorner) {
                        case "tl": // Top-left
                            region.x1 = mouseNormX;
                            region.y1 = mouseNormY;
                            break;
                        case "tr": // Top-right
                            region.x2 = mouseNormX;
                            region.y1 = mouseNormY;
                            break;
                        case "bl": // Bottom-left
                            region.x1 = mouseNormX;
                            region.y2 = mouseNormY;
                            break;
                        case "br": // Bottom-right
                            region.x2 = mouseNormX;
                            region.y2 = mouseNormY;
                            break;
                    }
                }
                
                // Update widget with new region values
                updateRegionWidget(this);
                
                this.setDirtyCanvas(true, false);
                return true; // Event handled
            };

            nodeType.prototype.onMouseUp = function(e, localPos, graphCanvas) {
                if (this.isDragging) {
                    this.isDragging = false;
                    this.dragCorner = null;
                    this.dragStartPos = null;
                    this.dragStartRegion = null;
                    this.setDirtyCanvas(true, false);
                    return true; // Event handled
                }
                return false;
            };

            // Make node resizable
            nodeType.prototype.onResize = function(size) {
                this.setDirtyCanvas(true, true);
            };
        }
    },
}); 