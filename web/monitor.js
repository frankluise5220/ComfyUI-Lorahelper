import { app } from "../../scripts/app.js";
import { ComfyWidgets } from "../../scripts/widgets.js";

app.registerExtension({
	name: "Comfy.LoraHelper.Monitor",
	async beforeRegisterNodeDef(nodeType, nodeData, app) {
		if (nodeData.name === "LH_History_Monitor") {
			function populate(text) {
                // Clear existing widgets to avoid duplication/stale state
				if (this.widgets) {
					for (let i = 0; i < this.widgets.length; i++) {
						this.widgets[i].onRemove?.();
					}
					this.widgets.length = 0;
				}

                // Ensure text is an array
				const v = Array.isArray(text) ? text : [text];
				
                // Combine into a single string for better display as a single block (User preference usually)
                // Or if user wants blocks, we can iterate. 
                // Given the chat history context, a single large text area is usually better for copy-paste/viewing.
                // But ShowText iterates. Let's join them to be safe and ensure one big box.
                const joinedText = v.join("\n");

                // Create standard ComfyUI STRING widget
                // Using ComfyWidgets["STRING"] ensures correct event handling and DOM structure
				const w = ComfyWidgets["STRING"](
                    this, 
                    "display_text", 
                    ["STRING", { multiline: true }], 
                    app
                ).widget;

				w.inputEl.readOnly = true;
				w.inputEl.style.opacity = 0.6;
				w.value = joinedText;

                // Auto-resize logic using standard computeSize
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
                if (!this.widgets || this.widgets.length === 0) {
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