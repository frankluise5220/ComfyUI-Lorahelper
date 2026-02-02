import { app } from "../../scripts/app.js";
import { ComfyWidgets } from "../../scripts/widgets.js";

// Helper function to create a resizable DOM widget
function setupSuperTextWidget(node, widgetName, inputName, app) {
    const widget = node.widgets?.find((w) => w.name === widgetName);
    if (!widget || !widget.inputEl) return;

    const w = widget;
    
    // Custom styling
    w.inputEl.style.border = "1px solid #333";
    w.inputEl.style.borderRadius = "4px";
    w.inputEl.style.padding = "4px";
    w.inputEl.style.lineHeight = "1.4";
    
    // Enable Resizing
    w.inputEl.style.resize = "vertical";
    w.inputEl.style.overflowY = "auto";
    
    // Selection styles (Ensures text can be copied)
    w.inputEl.style.userSelect = "text";
    w.inputEl.style.webkitUserSelect = "text";
    w.inputEl.style.pointerEvents = "auto";
    w.inputEl.style.cursor = "text";

    // Update state based on connection
    const updateState = () => {
        const input = node.inputs?.find(i => i.name === inputName);
        const isConnected = input && input.link !== null;
        
        if (isConnected) {
            w.inputEl.readOnly = true;
            w.inputEl.style.opacity = 0.6;
        } else {
            w.inputEl.readOnly = false;
            w.inputEl.style.opacity = 1.0;
        }
    };

    // Hook into connection changes
    const onConnectionsChange = node.onConnectionsChange;
    node.onConnectionsChange = function() {
        if (onConnectionsChange) onConnectionsChange.apply(this, arguments);
        updateState();
    };

    // Initial state
    updateState();
}

app.registerExtension({
	name: "Comfy.LoraHelper.Widgets",
	async beforeRegisterNodeDef(nodeType, nodeData, app) {
        // 1. LH_SuperText (SuperText) Logic
        // Finalized: 2026-02-02 - DO NOT MODIFY
		if (nodeData.name === "LH_SuperText") {
            const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function () {
                const r = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;
                
                setTimeout(() => {
                    setupSuperTextWidget(this, "showtext", "showtext", app);
                    
                    // Add tooltip (User suggestion)
                    const widget = this.widgets?.find(w => w.name === "showtext");
                    if (widget?.inputEl) {
                        widget.inputEl.title = "Connected: Copyable (Ctrl+C)\nDisconnected: Editable";
                    }
                }, 50);
                
                return r;
            };

            const onExecuted = nodeType.prototype.onExecuted;
            nodeType.prototype.onExecuted = async function (message) {
                onExecuted?.apply(this, arguments);

                if (!message?.text || !Array.isArray(message.text) || message.text.length === 0) {
                    return;
                }

                const displayValue = message.text[0] || "";

                // Strategy: 
                // 1. Try to use existing 'showtext' widget if it's visible (not converted to input).
                // 2. If not found, use/create 'display_text' widget.

                let targetWidget = this.widgets?.find(w => w.name === "showtext");
                
                // If showtext is missing (converted to input), we need display_text
                if (!targetWidget) {
                    if (!this.widgets) this.widgets = [];
                    targetWidget = this.widgets.find(w => w.name === "display_text");
                    
                    if (!targetWidget) {
                        targetWidget = this.addWidget(
                            "text", 
                            "display_text", 
                            displayValue, 
                            () => {}, 
                            { multiline: true }
                        );
                        
                        // Async styling for the new widget
                        requestAnimationFrame(() => {
                             if (targetWidget.inputEl) {
                                const el = targetWidget.inputEl;
                                el.readOnly = true;
                                el.style.userSelect = "text";
                                el.style.webkitUserSelect = "text";
                                el.style.pointerEvents = "auto";
                                el.style.cursor = "text";
                                el.style.opacity = "0.9";
                                el.style.backgroundColor = "#1a1a1a";
                                el.style.color = "#e0e0e0";
                                el.style.border = "1px solid #444";
                                el.style.padding = "6px";
                                el.style.fontFamily = "monospace";
                                el.style.whiteSpace = "pre-wrap";
                                el.title = "Copyable (Ctrl+C)";
                             }
                        });
                    }
                } else {
                    // If showtext exists, ensure we remove any stale display_text to clean up
                    const displayIdx = this.widgets?.findIndex(w => w.name === "display_text");
                    if (displayIdx !== undefined && displayIdx >= 0) {
                        this.widgets.splice(displayIdx, 1);
                    }
                }

                // Update the found/created widget
                if (targetWidget) {
                    targetWidget.value = displayValue;
                    if (targetWidget.inputEl) {
                        targetWidget.inputEl.value = displayValue;
                    }
                }

                // Force Redraw
                requestAnimationFrame(() => {
                    const sz = this.computeSize();
                    if (sz[0] < this.size[0]) sz[0] = this.size[0];
                    if (sz[1] < this.size[1]) sz[1] = this.size[1];
                    this.onResize?.(sz);
                    this.setDirtyCanvas(true, true);
                });
            };
		}

        // 2. LH_MultiTextSelector Logic
        if (nodeData.name === "LH_MultiTextSelector") {
            const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function () {
                const r = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;
                
                setTimeout(() => {
                    // LH_MultiTextSelector uses 'widget_text'
                    setupSuperTextWidget(this, "widget_text", "batch_text", app);
                }, 1);
                
                return r;
            };
        }
	},
});
