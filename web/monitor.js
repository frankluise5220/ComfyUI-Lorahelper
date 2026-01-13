import { app } from "../../scripts/app.js";
import { ComfyWidgets } from "../../scripts/widgets.js";

app.registerExtension({
	name: "Comfy.LoraHelper.Monitor",
	async beforeRegisterNodeDef(nodeType, nodeData, app) {
		if (nodeData.name === "LH_History_Monitor") {
			function populate(text) {
                // [Fix] Do NOT clear widgets if it's the "clear_history" switch!
                // The node has input widgets (like the switch) that should be preserved.
                // We only want to update the "display_text" widget.

                // Find existing display widget or create if missing
                let displayWidget = this.widgets ? this.widgets.find(w => w.name === "display_text") : null;
                
                const joinedText = Array.isArray(text) ? text.join("\n") : text;

                if (displayWidget) {
                    displayWidget.value = joinedText;
                } else {
                    // Create if not exists (should be rare if onNodeCreated works)
                    const w = ComfyWidgets["STRING"](
                        this, 
                        "display_text", 
                        ["STRING", { multiline: true }], 
                        app
                    ).widget;
                    w.inputEl.readOnly = true;
                    w.inputEl.style.opacity = 0.6;
                    w.value = joinedText;
                }

				requestAnimationFrame(() => {
					const sz = this.computeSize();
					if (sz[0] < this.size[0]) {
						sz[0] = this.size[0];
					}
					if (sz[1] < this.size[1]) {
						sz[1] = this.size[1];
					}
					this.onResize?.(sz);
					app.graph.setDirtyCanvas(true, false);
				});
			}

			const onExecuted = nodeType.prototype.onExecuted;
			nodeType.prototype.onExecuted = function (message) {
				onExecuted?.apply(this, arguments);
				if (message && message.text) {
                    populate.call(this, message.text);
                }
			};
            
            // Handle configuration (reload)
            const onConfigure = nodeType.prototype.onConfigure;
            nodeType.prototype.onConfigure = function () {
                onConfigure?.apply(this, arguments);
                // If we have saved values, restore them
                if (this.widgets_values && this.widgets_values.length) {
                     populate.call(this, this.widgets_values);
                }
            };
            
            // Initial setup - Create a placeholder if needed
             const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function() {
                onNodeCreated?.apply(this, arguments);
                
                // [Fix] Always ensure display_text widget exists, regardless of other widgets
                const hasDisplay = this.widgets && this.widgets.some(w => w.name === "display_text");
                
                if (!hasDisplay) {
                     const w = ComfyWidgets["STRING"](
                        this, 
                        "display_text", 
                        ["STRING", { multiline: true }], 
                        app
                    ).widget;
                    w.inputEl.readOnly = true;
                    w.inputEl.style.opacity = 0.6;
                    w.value = "Waiting for history...";
                }
            };
		}
	},
});