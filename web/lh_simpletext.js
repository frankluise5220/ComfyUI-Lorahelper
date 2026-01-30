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
    
    // Selection styles
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
        // 1. LH_SimpleText (SuperText) Logic
		if (nodeData.name === "LH_SimpleText") {
            const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function () {
                const r = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;
                
                setTimeout(() => {
                    setupSuperTextWidget(this, "widget_text", "showtext", app);
                }, 1);
                
                return r;
            };

			const onExecuted = nodeType.prototype.onExecuted;
			nodeType.prototype.onExecuted = function (message) {
				onExecuted?.apply(this, arguments);
				if (message && message.text) {
                    const texts = message.text; // List of strings
                    const mainWidget = this.widgets.find((w) => w.name === "widget_text");
                    
                    // If connected, update the widget value to show what's being passed
                    const input = this.inputs?.find(i => i.name === "showtext");
                    if (input && input.link !== null && mainWidget) {
                        mainWidget.value = texts[0];
                    }

                    for (let i = 1; i < texts.length; i++) {
                        const val = texts[i];
                        const name = `generated_text_${i}`;
                        let w = this.widgets.find(w => w.name === name);
                        if (!w) {
                            const wObj = ComfyWidgets["STRING"](
                                this, 
                                name, 
                                ["STRING", { multiline: true, default: "" }], 
                                app
                            );
                            w = wObj.widget;
                        w.inputEl.style.border = "1px solid #333";
                        w.inputEl.style.borderRadius = "4px";
                        w.inputEl.style.padding = "4px";
                        w.inputEl.style.lineHeight = "1.4";
                        w.inputEl.style.resize = "vertical";
                        w.inputEl.style.overflowY = "auto";
                        w.inputEl.style.marginTop = "5px";
                        
                        // Ensure dynamic widgets are also selectable
                        w.inputEl.style.userSelect = "text";
                        w.inputEl.style.webkitUserSelect = "text";
                        w.inputEl.style.pointerEvents = "auto";
                        w.inputEl.style.cursor = "text";
                        w.inputEl.readOnly = true;
                        w.inputEl.style.opacity = 0.6;
                    }
                    w.value = val;
                    }

                    // Remove excess widgets
                    for (let i = texts.length; i < 100; i++) {
                        const name = `generated_text_${i}`;
                        const idx = this.widgets.findIndex(w => w.name === name);
                        if (idx !== -1) {
                            this.widgets[idx].inputEl?.remove();
                            this.widgets.splice(idx, 1);
                        } else {
                            if (i > texts.length + 5) break; 
                        }
                    }
                    
                    this.onResize?.(this.size);
                    app.graph.setDirtyCanvas(true, false);
				}
			};
		}

        // 2. LH_MultiTextSelector Logic
        if (nodeData.name === "LH_MultiTextSelector") {
            const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function () {
                const r = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;
                
                setTimeout(() => {
                    setupSuperTextWidget(this, "widget_text", "batch_text", app);
                }, 1);
                
                return r;
            };
        }
	},
});
