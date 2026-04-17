import { app } from "../../../scripts/app.js";

function normalizeHexColor(value, fallback = "#000000") {
    if (typeof value !== "string") {
        return fallback;
    }

    const normalized = value.startsWith("#") ? value : `#${value}`;
    return /^#[0-9a-fA-F]{6}$/.test(normalized) ? normalized : fallback;
}

function getBrightness(hexColor) {
    const hex = normalizeHexColor(hexColor).slice(1);
    const r = parseInt(hex.slice(0, 2), 16);
    const g = parseInt(hex.slice(2, 4), 16);
    const b = parseInt(hex.slice(4, 6), 16);
    return (r * 299 + g * 587 + b * 114) / 1000;
}

function createColorWidget(name, value) {
    const widget = {
        name,
        type: "SMART_COLOR",
        value: normalizeHexColor(value),
        options: { default: "#000000" },
        draw(ctx, node, widgetWidth, widgetY, height) {
            const border = 3;

            ctx.fillStyle = "#000";
            ctx.fillRect(0, widgetY, widgetWidth, height);

            ctx.fillStyle = this.value;
            ctx.fillRect(
                border,
                widgetY + border,
                widgetWidth - border * 2,
                height - border * 2
            );

            ctx.fillStyle = getBrightness(this.value) > 125 ? "#000" : "#fff";
            ctx.font = "14px Arial";
            ctx.textAlign = "center";
            ctx.fillText(this.name, widgetWidth * 0.5, widgetY + 14);
        },
        mouse(event, pos, node) {
            if (event.type !== "pointerdown") {
                return false;
            }

            const widgets = node.widgets.filter((widgetItem) => widgetItem.type === "SMART_COLOR");

            for (const colorWidget of widgets) {
                const rect = [colorWidget.last_y, colorWidget.last_y + 32];
                if (pos[1] <= rect[0] || pos[1] >= rect[1]) {
                    continue;
                }

                const picker = document.createElement("input");
                picker.type = "color";
                picker.value = normalizeHexColor(this.value);

                Object.assign(picker.style, {
                    position: "fixed",
                    left: `${event.clientX}px`,
                    top: `${event.clientY}px`,
                    height: "0px",
                    width: "0px",
                    padding: "0px",
                    opacity: 0,
                });

                picker.addEventListener("blur", () => {
                    this.callback?.(this.value);
                    node.graph._version++;
                    picker.remove();
                });

                picker.addEventListener("input", () => {
                    if (!picker.value) {
                        return;
                    }

                    this.value = normalizeHexColor(picker.value, this.value);
                    app.canvas.setDirty(true);
                });

                document.body.appendChild(picker);

                requestAnimationFrame(() => {
                    if (picker.showPicker) {
                        picker.showPicker();
                    } else {
                        picker.click();
                    }
                    picker.focus();
                });

                return true;
            }

            return false;
        },
        computeSize(width) {
            return [width, 32];
        },
    };

    return widget;
}

function hideWidget(node, widget) {
    if (widget._hidden) return;
    widget._hidden = true;
    widget._origType = widget.type;
    widget._origComputeSize = widget.computeSize;
    widget.type = "hidden";
    widget.computeSize = () => [0, -4];
}

function showWidget(node, widget) {
    if (!widget._hidden) return;
    widget._hidden = false;
    widget.type = widget._origType;
    widget.computeSize = widget._origComputeSize;
}

function updateColorVisibility(node) {
    const colorsWidget = node.widgets?.find(w => w.name === "colors");
    if (!colorsWidget) return;

    const count = colorsWidget.value;
    for (let i = 1; i <= 5; i++) {
        const cw = node.widgets.find(w => w.name === `color_${i}`);
        if (!cw) continue;
        if (i <= count) {
            showWidget(node, cw);
        } else {
            hideWidget(node, cw);
        }
    }

    node.setSize(node.computeSize());
    app.canvas.setDirty(true);
}

app.registerExtension({
    name: "Comfy.SmartNodes.SmartColorWidget",
    getCustomWidgets() {
        return {
            SMART_COLOR: (node, inputName, inputData) => ({
                widget: node.addCustomWidget(
                    createColorWidget(inputName, inputData?.[1]?.default || "#000000")
                ),
                minWidth: 150,
                minHeight: 32,
            }),
        };
    },
    beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== "SmartImagePaletteCreate") return;

        const origOnNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            origOnNodeCreated?.apply(this, arguments);

            const colorsWidget = this.widgets?.find(w => w.name === "colors");
            if (colorsWidget) {
                const origCallback = colorsWidget.callback;
                colorsWidget.callback = (value) => {
                    origCallback?.call(colorsWidget, value);
                    updateColorVisibility(this);
                };
            }

            requestAnimationFrame(() => updateColorVisibility(this));
        };

        const origOnConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function (data) {
            origOnConfigure?.apply(this, arguments);
            requestAnimationFrame(() => updateColorVisibility(this));
        };
    },
});
