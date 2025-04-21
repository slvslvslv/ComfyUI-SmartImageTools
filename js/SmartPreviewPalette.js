import { app } from "../../../scripts/app.js";
import { api } from "../../../scripts/api.js";
import { ComfyWidgets } from "../../../scripts/widgets.js";

// Displays a palette image as a grid of color squares
app.registerExtension({
    name: "Comfy.SmartNodes.SmartPreviewPalette",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name === "SmartPreviewPalette") {
            const onDrawForeground = nodeType.prototype.onDrawForeground;

            nodeType.prototype.onDrawForeground = function (ctx) {
                if (onDrawForeground) {
                    onDrawForeground.apply(this, arguments);
                }
                const data = this.colorData;
                if (data && data.length > 0) {
                    const colors = data[0]; // We wrapped it in the python code
                    if (!colors || colors.length === 0) return;

                    const numColors = colors.length;
                    const nodeWidth = this.size[0];
                    const nodeHeight = this.size[1];
                    const margin = 5; // Small margin around the grid
                    const titleHeight = (this.properties.title_height || 20);
                    const textHeight = 20; // Reserve space for the text label

                    // Calculate grid dimensions
                    let cols = Math.ceil(Math.sqrt(numColors));

                    // Calculate square size based on available space, *excluding* text area
                    const availableWidth = nodeWidth - 2 * margin;
                    const availableHeightForGrid = nodeHeight - 2 * margin - titleHeight - textHeight; // Reduced height for grid

                    if (availableHeightForGrid <= 0) return; // Not enough space to draw grid

                    let rows = Math.ceil(numColors / cols);
                    const squareWidth = Math.max(1, Math.floor(availableWidth / cols));
                    const squareHeight = Math.max(1, Math.floor(availableHeightForGrid / rows)); // Use reduced height
                    let squareSize = Math.min(squareWidth, squareHeight);

                    // Ensure square size is at least 1 pixel
                    squareSize = Math.max(1, squareSize);

                    // Recalculate cols/rows based on actual square size to better fit
                    cols = Math.floor(availableWidth / squareSize);
                    rows = Math.floor(availableHeightForGrid / squareSize);

                    // Center the grid within its allocated space
                    const gridWidth = cols * squareSize;
                    const gridHeight = rows * squareSize;
                    const startX = margin + (availableWidth - gridWidth) / 2;
                    // Start grid below the title
                    const startY = margin + titleHeight + (availableHeightForGrid - gridHeight) / 2;

                    let drawnCount = 0;
                    for (let r = 0; r < rows; r++) {
                        for (let c = 0; c < cols; c++) {
                            const index = r * cols + c;
                            if (index < numColors) {
                                ctx.fillStyle = colors[index];
                                ctx.fillRect(
                                    startX + c * squareSize,
                                    startY + r * squareSize,
                                    squareSize,
                                    squareSize
                                );
                                drawnCount++;
                            } else {
                                break; // No more colors to draw
                            }
                        }
                         if (drawnCount >= numColors) break;
                    }

                    // Draw the color count text below the grid, within the reserved space
                    if (colors && colors.length > 0) {
                        const text = `Colors: ${numColors}`;
                        ctx.fillStyle = "#CCC"; // Light gray text color
                        ctx.font = "12px Arial";
                        ctx.textAlign = "center";
                        // Position text at the bottom, centered horizontally
                        const textY = nodeHeight - margin - (textHeight / 2) + 4; // Center vertically in the reserved space (+4 for font baseline)
                        ctx.fillText(text, nodeWidth / 2, textY);
                    }
                }
            };

            // Store the color data when the node is executed
            const onExecuted = nodeType.prototype.onExecuted;
            nodeType.prototype.onExecuted = function (message) {
                if (onExecuted) {
                    onExecuted.apply(this, arguments);
                }
                if (message?.colors) {
                    this.colorData = message.colors;
                }
                this.setDirtyCanvas(true, true);
            };

             // Make node resizable
             this.resizable = true;
        }
    },
}); 