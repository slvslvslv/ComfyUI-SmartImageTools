import { app } from "../../../scripts/app.js";

app.registerExtension({
    name: "Comfy.SmartNodes.SmartImagePreviewScaled",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name === "SmartImagePreviewScaled") {
            // Store original dimensions
            const onExecuted = nodeType.prototype.onExecuted;
            nodeType.prototype.onExecuted = function(message) {
                onExecuted?.apply(this, arguments);
                
                // Store original dimensions if provided
                if (message?.original_dimensions && message.original_dimensions.length > 0) {
                    this.originalDimensions = message.original_dimensions[0];
                } else {
                    this.originalDimensions = null;
                }
            };

            // Override the onDrawForeground to display original dimensions
            const onDrawForeground = nodeType.prototype.onDrawForeground;
            nodeType.prototype.onDrawForeground = function(ctx) {
                const result = onDrawForeground?.apply(this, arguments);
                
                if (this.flags.collapsed) return result;
                
                // Draw original dimensions at the bottom right
                if (this.originalDimensions) {
                    const text = `${this.originalDimensions[0]}x${this.originalDimensions[1]}`;
                    const bottomY = this.size[1] - 5;
                    
                    ctx.fillStyle = "#CCC";
                    ctx.font = "10px Arial";
                    ctx.textAlign = "right";
                    ctx.textBaseline = "alphabetic";
                    ctx.fillText(text, this.size[0] - 5, bottomY);
                }
                
                return result;
            };
        }
    },
});

