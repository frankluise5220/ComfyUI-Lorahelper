import { app } from "../../scripts/app.js";
import { ComfyWidgets } from "../../scripts/widgets.js";

// Helper function to create a resizable DOM widget
function createResizableDOMWidget(node, widgetName, app) {
    const widgetIndex = node.widgets?.findIndex((w) => w.name === widgetName);
    
    // If found and it's a standard canvas widget (no inputEl), replace it
    if (widgetIndex !== -1 && !node.widgets[widgetIndex].inputEl) {
        const defaultVal = node.widgets[widgetIndex].value;

        // Remove the canvas widget
        node.widgets.splice(widgetIndex, 1);

        // Create DOM-based widget
        const wObj = ComfyWidgets["STRING"](
            node, 
            widgetName, 
            ["STRING", { multiline: true, default: defaultVal }], 
            app
        );
        
        const w = wObj.widget;
        w.value = defaultVal;
        
        // Custom styling
        w.inputEl.style.border = "1px solid #333";
        w.inputEl.style.borderRadius = "4px";
        w.inputEl.style.padding = "4px";
        w.inputEl.style.lineHeight = "1.4";
        
        // Enable Resizing
        w.inputEl.style.resize = "vertical"; // Allow vertical resizing
        w.inputEl.style.overflowY = "auto";  // Scrollbar when needed
        
        return w;
    }
    return null;
}

app.registerExtension({
	name: "Comfy.LoraHelper.Widgets",
	async beforeRegisterNodeDef(nodeType, nodeData, app) {
        // 1. LH_SimpleText Logic
		if (nodeData.name === "LH_SimpleText") {
            const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function () {
                const r = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;
                createResizableDOMWidget(this, "text", app);
                return r;
            };

			const onExecuted = nodeType.prototype.onExecuted;
			nodeType.prototype.onExecuted = function (message) {
				onExecuted?.apply(this, arguments);
				if (message && message.text) {
                    const texts = message.text; // List of strings
                    const mainWidget = this.widgets.find((w) => w.name === "text");
                    
                    for (let i = 0; i < texts.length; i++) {
                        const val = texts[i];
                        if (i === 0 && mainWidget) {
                            mainWidget.value = val;
                            // Ensure generated_text_0 is gone (cleanup if mainWidget reappeared)
                            const badIdx = this.widgets.findIndex(w => w.name === "generated_text_0");
                            if (badIdx !== -1) { 
                                this.widgets[badIdx].inputEl?.remove(); 
                                this.widgets.splice(badIdx, 1); 
                            }
                        } else {
                            // Create/Update generated_text_i
                            const name = `generated_text_${i}`;
                            let w = this.widgets.find(w => w.name === name);
                            if (!w) {
                                // Create new widget
                                const wObj = ComfyWidgets["STRING"](
                                    this, 
                                    name, 
                                    ["STRING", { multiline: true, default: "" }], 
                                    app
                                );
                                w = wObj.widget;
                                // Apply custom styles to match main widget
                                w.inputEl.style.border = "1px solid #333";
                                w.inputEl.style.borderRadius = "4px";
                                w.inputEl.style.padding = "4px";
                                w.inputEl.style.lineHeight = "1.4";
                                w.inputEl.style.resize = "vertical";
                                w.inputEl.style.overflowY = "auto";
                                w.inputEl.style.marginTop = "5px"; // Add some spacing
                            }
                            w.value = val;
                        }
                    }

                    // Remove excess widgets
                    for (let i = texts.length; i < 100; i++) {
                        const name = `generated_text_${i}`;
                        const idx = this.widgets.findIndex(w => w.name === name);
                        if (idx !== -1) {
                            this.widgets[idx].inputEl?.remove();
                            this.widgets.splice(idx, 1);
                        } else {
                            // If generated_text_i doesn't exist, likely i+1 doesn't either
                            // But let's check a few more just in case
                            if (i > texts.length + 5) break; 
                        }
                    }
                    
                    // Auto-resize node to fit new widgets
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
                // Replace 'batch_text' with a resizable DOM widget
                createResizableDOMWidget(this, "batch_text", app);
                return r;
            };
        }
	},
});
