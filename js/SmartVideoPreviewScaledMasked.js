import { app } from "../../../scripts/app.js";
import { api } from "../../../scripts/api.js";

// Displays an animated video preview using a sequence of images with mask overlay
app.registerExtension({
    name: "Comfy.SmartNodes.SmartVideoPreviewScaledMasked",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name === "SmartVideoPreviewScaledMasked") {

            // Initialize node properties
            const initProperties = (node) => {
                node.fps = 12.0;
                node.currentFrameIndex = 0;
                node.lastFrameTime = 0;
                node.animationFrameId = null;
                node.imageElements = []; // Store loaded Image elements
                node.isLoading = false;
                node.dimensions = null; // Store original dimensions [width, height]
            };

            // Load images from the backend data
            const loadImages = (node, frameInfoList) => {
                node.isLoading = true;
                node.imageElements = []; // Clear previous images
                node.dimensions = null; // Reset dimensions
                node.setDirtyCanvas(true, true);

                if (!frameInfoList || frameInfoList.length === 0) {
                    node.isLoading = false;
                    node.setDirtyCanvas(true, true);
                    return;
                }

                let imagesLoaded = 0;
                const totalImages = frameInfoList.length;

                frameInfoList.forEach((imageInfo, index) => {
                    const img = new Image();
                    node.imageElements[index] = img; // Store placeholder initially
                    img.onload = () => {
                        imagesLoaded++;
                        if (index === 0) { // Get dimensions from the first image
                            node.dimensions = [img.naturalWidth, img.naturalHeight];
                        }
                        if (imagesLoaded === totalImages) {
                            node.isLoading = false;
                            node.currentFrameIndex = 0;
                            node.lastFrameTime = performance.now();
                            node.setDirtyCanvas(true, true);
                            if (node.animationFrameId === null) {
                                startAnimation(node);
                            }
                        }
                    };
                    img.onerror = () => {
                        imagesLoaded++; // Count errors too to avoid getting stuck
                        console.error("Error loading image:", imageInfo.filename);
                        node.imageElements[index] = null; // Mark as failed
                        if (imagesLoaded === totalImages) {
                            node.isLoading = false;
                            node.setDirtyCanvas(true, true);
                        }
                    };
                    // Use api.apiURL() to construct the full URL for temp images
                    img.src = api.apiURL('/view?filename=' + encodeURIComponent(imageInfo.filename) + '&type=' + imageInfo.type + '&subfolder=' + encodeURIComponent(imageInfo.subfolder || ''));
                });
            };

            // Animation loop
            const animationLoop = (node, timestamp) => {
                if (!node.imageElements || node.imageElements.length === 0 || node.isLoading) {
                    node.animationFrameId = requestAnimationFrame((t) => animationLoop(node, t));
                    return;
                }

                const frameInterval = 1000 / node.fps;
                const elapsed = timestamp - node.lastFrameTime;

                if (elapsed >= frameInterval) {
                    node.lastFrameTime = timestamp - (elapsed % frameInterval);
                    node.currentFrameIndex = (node.currentFrameIndex + 1) % node.imageElements.length;
                    node.setDirtyCanvas(true, false); // Redraw without resizing
                }

                node.animationFrameId = requestAnimationFrame((t) => animationLoop(node, t));
            };

            // Start the animation
            const startAnimation = (node) => {
                if (node.animationFrameId !== null) {
                    cancelAnimationFrame(node.animationFrameId);
                }
                node.lastFrameTime = performance.now();
                node.animationFrameId = requestAnimationFrame((t) => animationLoop(node, t));
            };

            // Stop the animation
            const stopAnimation = (node) => {
                if (node.animationFrameId !== null) {
                    cancelAnimationFrame(node.animationFrameId);
                    node.animationFrameId = null;
                }
            };

            // Initialize on node creation
            const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function() {
                onNodeCreated?.apply(this, arguments);
                initProperties(this);
            };

            // Handle data from the backend
            const onExecuted = nodeType.prototype.onExecuted;
            nodeType.prototype.onExecuted = function(message) {
                onExecuted?.apply(this, arguments);

                // Look for 'video_frames' instead of 'images'
                if (message?.video_frames) {
                    // Check if message.fps is an array and take the first element
                    this.fps = Array.isArray(message.fps) && message.fps.length > 0 ? message.fps[0] : (message.fps || 12.0);
                    stopAnimation(this); // Stop previous animation before loading new images
                    // Pass the video_frames data to loadImages
                    loadImages(this, message.video_frames);
                } else {
                    // No frames received, clear existing ones
                    this.imageElements = [];
                    this.dimensions = null;
                    stopAnimation(this);
                    this.setDirtyCanvas(true, true);
                }
            };

            // Custom drawing logic
            const onDrawForeground = nodeType.prototype.onDrawForeground;
            nodeType.prototype.onDrawForeground = function(ctx) {
                onDrawForeground?.apply(this, arguments);

                if (this.flags.collapsed) return;

                // Define explicit padding values
                const margin = 10; // Horizontal margin
                const topPadding = 75; // Increased top padding
                const bottomPadding = 30; // Added bottom padding
                const titleHeight = LiteGraph.NODE_TITLE_HEIGHT || 20;
                const infoBarHeight = 16; // Height of the bottom info bar

                // Calculate available size based on node size and padding
                let availableWidth = this.size[0] - margin * 2;
                let availableHeight = this.size[1] - titleHeight - topPadding - bottomPadding - infoBarHeight;
                const top = titleHeight + topPadding; // Adjusted top position
                const left = margin;

                // Prevent negative available height/width
                availableWidth = Math.max(0, availableWidth);
                availableHeight = Math.max(0, availableHeight);

                if (this.isLoading) {
                    ctx.fillStyle = "#CCC";
                    ctx.font = "12px Arial";
                    ctx.textAlign = "center";
                    ctx.fillText("Loading...", this.size[0] / 2, top + availableHeight / 2);
                    return;
                }

                if (!this.imageElements || this.imageElements.length === 0 || this.currentFrameIndex >= this.imageElements.length) {
                    ctx.fillStyle = "#555";
                    ctx.fillRect(left, top, availableWidth, availableHeight);
                    ctx.fillStyle = "#CCC";
                    ctx.font = "12px Arial";
                    ctx.textAlign = "center";
                    ctx.fillText("No Video", this.size[0] / 2, top + availableHeight / 2);
                    return;
                }

                const currentImage = this.imageElements[this.currentFrameIndex];
                if (!currentImage || !currentImage.complete || currentImage.naturalWidth === 0) {
                    // Image might still be loading or failed
                    ctx.fillStyle = "#555";
                    ctx.fillRect(left, top, availableWidth, availableHeight);
                    ctx.fillStyle = "#CCC";
                    ctx.font = "14px Arial";
                    ctx.textAlign = "center";
                    ctx.fillText("Frame Error", this.size[0] / 2, top + availableHeight / 2);
                    return;
                }

                let drawWidth, drawHeight, drawX, drawY;
                // Use natural dimensions of the *already scaled* image
                const imgWidth = currentImage.naturalWidth;
                const imgHeight = currentImage.naturalHeight;
                const aspectRatio = imgWidth / imgHeight;

                // Fit the image within the available node space while maintaining aspect ratio
                if (availableWidth / aspectRatio <= availableHeight) {
                    // Fit width
                    drawWidth = availableWidth;
                    drawHeight = drawWidth / aspectRatio;
                } else {
                    // Fit height
                    drawHeight = availableHeight;
                    drawWidth = drawHeight * aspectRatio;
                }

                // Center the image within the available space
                drawX = left + (availableWidth - drawWidth) / 2;
                drawY = top + (availableHeight - drawHeight) / 2;

                try {
                    ctx.drawImage(currentImage, drawX, drawY, drawWidth, drawHeight);
                } catch (error) {
                    console.error("Error drawing image:", error);
                    ctx.fillStyle = "#F55"; // Indicate drawing error
                    ctx.fillRect(drawX, drawY, drawWidth, drawHeight);
                }

                // Optionally, draw frame number/info at the very bottom
                ctx.fillStyle = "rgba(0, 0, 0, 0.6)";
                // Position info bar respecting the new bottom padding
                const infoBarY = this.size[1] - bottomPadding - infoBarHeight + 10;
                ctx.fillRect(left, infoBarY, availableWidth, infoBarHeight);
                ctx.fillStyle = "#FFF";
                ctx.font = "12px Arial";
                ctx.textAlign = "center";
                const frameText = `Frame: ${this.currentFrameIndex + 1}/${this.imageElements.length} (Masked)`;
                // Adjust text Y position within the info bar
                ctx.fillText(frameText, left + availableWidth/2, infoBarY + infoBarHeight - 4);
            };

            // Clean up animation on node removal
            const onRemoved = nodeType.prototype.onRemoved;
            nodeType.prototype.onRemoved = function() {
                stopAnimation(this);
                onRemoved?.apply(this, arguments);
            };
        }
    },
}); 

