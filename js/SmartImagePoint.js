import { app } from "../../../scripts/app.js";
import { $el } from "../../../scripts/ui.js"; // Import $el for creating elements if needed
import { SmartImage } from "./image.js"; // Import our new SmartImage class

// Displays an image preview directly within the node with draggable points
app.registerExtension({
    name: "Comfy.SmartNodes.SmartImagePoint",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name === "SmartImagePoint") {

            // Find the points_data input widget
            function getPointsWidget(node) {
                return node.widgets.find(w => w.name === "points_data");
            }

            // Initialize dot properties
            const initProperties = (ctx) => {
                if (!ctx.properties) {
                    ctx.properties = {};
                }
                
                // Get the hidden widget
                const pointsWidget = getPointsWidget(ctx);
                let dots = [];
                
                if (pointsWidget && pointsWidget.value) {
                    // Parse dots from the widget value
                    try {
                        dots = JSON.parse(pointsWidget.value);
                        if (!Array.isArray(dots) || dots.length === 0) {
                            dots = [{ x: 0.5, y: 0.5 }];
                        }
                    } catch (e) {
                        console.error("Error parsing points_data:", e);
                        dots = [{ x: 0.5, y: 0.5 }];
                    }
                } else {
                    // Default if widget not found
                    dots = [{ x: 0.5, y: 0.5 }];
                }
                
                // Store dots in properties
                ctx.properties.dots = dots;
                
                // Update widget with initial value
                if (pointsWidget) {
                    pointsWidget.value = JSON.stringify(dots);
                }
                
                ctx.draggingDotIndex = null;
                ctx.isDraggingDot = false;
                ctx.addButtonRect = null;
                ctx.removeButtonRect = null;
                
                // Initialize SmartImage
                ctx.smartImage = new SmartImage();
            };

            // Function to update widget when dots change
            const updatePointsWidget = (node) => {
                if (!node.properties.dots) return;
                
                const pointsWidget = getPointsWidget(node);
                if (pointsWidget) {
                    // Convert y coordinates to bottom-left system (ComfyUI convention)
                    const convertedDots = node.properties.dots.map(dot => ({
                        x: dot.x,
                        y: 1.0 - dot.y  // Flip Y (top-left to bottom-left)
                    }));
                    pointsWidget.value = JSON.stringify(convertedDots);
                    // Needed to notify ComfyUI that a parameter changed
                    if (pointsWidget.callback) {
                        pointsWidget.callback(pointsWidget.value);
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

            // Store image data and dimensions, reset dots on new image
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
                
                // Check if points_data is provided from the backend
                if (message?.points_data) {
                    try {
                        // Sanitize input before parsing
                        const pointsDataString = String(message.points_data).trim();
                        
                        // Skip invalid or empty JSON
                        if (!pointsDataString || pointsDataString === "null" || pointsDataString === "undefined") {
                            console.warn("Empty points_data received");
                        } else {
                            const dotsFromBackend = JSON.parse(pointsDataString);
                            
                            // Verify it's a valid array with elements
                            if (Array.isArray(dotsFromBackend)) {
                                // Filter out any invalid dot objects
                                const validDots = dotsFromBackend.filter(dot => 
                                    dot && typeof dot === 'object' && 
                                    'x' in dot && 'y' in dot &&
                                    !isNaN(parseFloat(dot.x)) && !isNaN(parseFloat(dot.y))
                                );
                                
                                if (validDots.length > 0) {
                                    // Convert valid dots (ensuring numeric values)
                                    const convertedDots = validDots.map(dot => ({
                                        x: parseFloat(dot.x),
                                        y: 1.0 - parseFloat(dot.y)  // Flip Y (bottom-left to top-left)
                                    }));
                                    
                                    const currentDotsStr = JSON.stringify(this.properties.dots || []);
                                    const newDotsStr = JSON.stringify(convertedDots);
                                    
                                    if (newImageLoaded || currentDotsStr !== newDotsStr) {
                                        this.properties.dots = convertedDots;
                                    }
                                } else {
                                    console.warn("No valid dots found in points_data");
                                }
                            } else {
                                console.warn("points_data is not an array:", dotsFromBackend);
                            }
                        }
                    } catch (e) {
                        console.error("Error parsing points_data from backend:", e, "\nRaw value:", message.points_data);
                    }
                }
                
                // If it's a new image and no valid points were loaded, reset to center
                if (newImageLoaded && (!this.properties.dots || this.properties.dots.length === 0)) {
                    this.properties.dots = [{ x: 0.5, y: 0.5 }];
                    updatePointsWidget(this);
                }
                
                this.setDirtyCanvas(true, true);
            };

            const onDrawForeground = nodeType.prototype.onDrawForeground;
            nodeType.prototype.onDrawForeground = function (ctx) {
                onDrawForeground?.apply(this, arguments);

                // Ensure properties are initialized if they were somehow lost
                if (!this.properties || !this.properties.dots || !this.smartImage) {
                    initProperties(this);
                }

                const margin = 5;
                const titleHeight = LiteGraph.NODE_TITLE_HEIGHT || 20;
                const bottomBarHeight = 16; // Height for buttons and text
                const buttonWidth = 45;
                const buttonMargin = 5;
                const textFontSize = 10;
                const nodeWidth = this.size[0];
                const nodeHeight = this.size[1];
                
                // Draw the image using SmartImage
                const imageRect = this.smartImage.drawImage(ctx, 0, 0, nodeWidth, nodeHeight - bottomBarHeight);
                
                if (imageRect && this.properties.dots.length > 0) {
                    // Draw all dots
                    const mainRadius = 5;
                    const centerRadius = 1;
                    const outlineWidth = 1;
                    for (const dot of this.properties.dots) {
                        const dotCanvasX = imageRect.x + dot.x * imageRect.w;
                        const dotCanvasY = imageRect.y + dot.y * imageRect.h;

                        ctx.fillStyle = "rgba(255, 0, 0, 0.5)"; // Main circle color (example red)
                        ctx.beginPath();
                        ctx.arc(dotCanvasX, dotCanvasY, mainRadius, 0, 2 * Math.PI);
                        ctx.fill();

                        ctx.strokeStyle = "#000000";
                        ctx.lineWidth = outlineWidth;
                        ctx.beginPath();
                        ctx.arc(dotCanvasX, dotCanvasY, centerRadius, 0, 2 * Math.PI);
                        ctx.stroke();

                        ctx.fillStyle = "#FFFFFF";
                        ctx.beginPath(); // Need new path for fill after stroke
                        ctx.arc(dotCanvasX, dotCanvasY, centerRadius, 0, 2 * Math.PI);
                        ctx.fill();
                    }
                }

                // Draw Bottom Bar (Buttons and Text)
                const bottomY = nodeHeight - margin - bottomBarHeight;
                const textY = bottomY + bottomBarHeight / 2 + textFontSize / 3; // Vertical center
                const textMargin = 5;

                // Points Count Text (Left Aligned)
                const pointsCountText = `Points: ${this.properties.dots.length}`;
                ctx.fillStyle = "#CCC";
                ctx.font = `${textFontSize}px Arial`;
                ctx.textAlign = "left";
                const pointsTextMetrics = ctx.measureText(pointsCountText);
                const pointsTextWidth = pointsTextMetrics.width;
                ctx.fillText(pointsCountText, margin, textY);

                // Adjust starting position for buttons
                let currentX = margin + pointsTextWidth + textMargin * 2; // Start buttons after text

                // Draw dimensions text
                this.smartImage.drawDimensions(ctx, 0, textY, nodeWidth);

                // "Add" Button
                const addX = currentX; // Use updated starting position
                this.addButtonRect = { x: addX, y: bottomY, w: buttonWidth, h: bottomBarHeight };
                ctx.fillStyle = "#444"; // Button background
                ctx.fillRect(this.addButtonRect.x, this.addButtonRect.y, this.addButtonRect.w, this.addButtonRect.h);
                ctx.fillStyle = "#CCC";
                ctx.textAlign = "center";
                ctx.fillText("Add", addX + buttonWidth / 2, textY);

                // "Remove" Button
                const removeX = addX + buttonWidth + buttonMargin;
                this.removeButtonRect = { x: removeX, y: bottomY, w: buttonWidth, h: bottomBarHeight };
                ctx.fillStyle = this.properties.dots.length <= 1 ? "#222" : "#444"; // Dim if cannot remove
                ctx.fillRect(this.removeButtonRect.x, this.removeButtonRect.y, this.removeButtonRect.w, this.removeButtonRect.h);
                ctx.fillStyle = this.properties.dots.length <= 1 ? "#666" : "#CCC"; // Dim text
                ctx.textAlign = "center";
                ctx.fillText("Remove", removeX + buttonWidth / 2, textY);
            };

            // --- Mouse Interaction for Dragging and Buttons ---

            nodeType.prototype.onMouseDown = function(e, localPos, graphCanvas) {
                if (!e.isPrimary) return false; // Only left click

                const clickPos = { x: localPos[0], y: localPos[1] };

                // Check Add Button Click
                if (this.addButtonRect && this.smartImage.pointInRect(clickPos, this.addButtonRect)) {
                    this.properties.dots.push({ x: 0.5, y: 0.5 }); // Add new dot at center
                    updatePointsWidget(this); // Update the widget value
                    this.setDirtyCanvas(true, true);
                    return true; // Event handled
                }

                // Check Remove Button Click (only if more than one dot)
                if (this.removeButtonRect && this.properties.dots.length > 1 && this.smartImage.pointInRect(clickPos, this.removeButtonRect)) {
                    this.properties.dots.pop(); // Remove the last dot
                    updatePointsWidget(this); // Update the widget value
                    this.setDirtyCanvas(true, true);
                    return true; // Event handled
                }

                // Check Dot Click/Drag
                if (this.smartImage.imageDrawRect && this.properties.dots.length > 0) {
                    const rect = this.smartImage.imageDrawRect;
                    const clickRadiusSq = 10 * 10; // Larger click area squared

                    // Check dots in reverse order so top ones are selected first
                    for (let i = this.properties.dots.length - 1; i >= 0; i--) {
                        const dot = this.properties.dots[i];
                        const dotCanvasX = rect.x + dot.x * rect.w;
                        const dotCanvasY = rect.y + dot.y * rect.h;

                        const dx = localPos[0] - dotCanvasX;
                        const dy = localPos[1] - dotCanvasY;

                        if (dx * dx + dy * dy < clickRadiusSq) {
                            this.isDraggingDot = true;
                            this.draggingDotIndex = i;
                            this.setDirtyCanvas(true, false);
                            return true; // Event handled
                        }
                    }
                }

                return false; // Event not handled
            };

            nodeType.prototype.onMouseMove = function(e, localPos, graphCanvas) {
                if (!this.isDraggingDot || this.draggingDotIndex === null || !this.smartImage.imageDrawRect) return false;

                const rect = this.smartImage.imageDrawRect;
                const draggedDot = this.properties.dots[this.draggingDotIndex];

                // Convert mouse position to normalized image coordinates
                let newDotX = (localPos[0] - rect.x) / rect.w;
                let newDotY = (localPos[1] - rect.y) / rect.h;

                // Clamp values and update dot position
                draggedDot.x = Math.max(0, Math.min(1, newDotX));
                draggedDot.y = Math.max(0, Math.min(1, newDotY));
                
                // Update widget when dot position changes
                updatePointsWidget(this);
                
                this.setDirtyCanvas(true, false);
                return true; // Event handled
            };

            nodeType.prototype.onMouseUp = function(e, localPos, graphCanvas) {
                if (this.isDraggingDot) {
                    this.isDraggingDot = false;
                    this.draggingDotIndex = null;
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