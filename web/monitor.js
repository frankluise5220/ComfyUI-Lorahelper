import { app } from "../../scripts/app.js";

app.registerExtension({
	name: "Comfy.LoraHelper.Monitor",
	async beforeRegisterNodeDef(nodeType, nodeData, app) {
		if (nodeData.name === "LH_History_Monitor") {
			const onExecuted = nodeType.prototype.onExecuted;
			nodeType.prototype.onExecuted = function (message) {
				onExecuted?.apply(this, arguments);

				if (message && message.text) {
					// Join with newlines to preserve formatting
					const text_val = message.text.join("\n");
					
					// Find existing widget
					let widget = this.widgets?.find((w) => w.name === "display_text");
					
					// If not found, create it
					if (!widget) {
						// addWidget(type, name, value, callback, options)
                        // Use "customtext" if available, else "text" with multiline
                        // Actually, standard ComfyUI just uses "text" with multiline options for big boxes
						widget = this.addWidget(
							"text", 
							"display_text", 
							text_val, 
							function(v) {}, 
							{ multiline: true } 
						);
					}
                    
                    // Update value
                    widget.value = text_val;
                    
					// Auto-resize node logic
					const lineCount = text_val.split('\n').length;
                    // Approximate height: line count * line height + header/padding
					const estimatedHeight = Math.max(240, lineCount * 18 + 80);
                    const currentWidth = this.size[0];
                    
                    // Resize only if needed to avoid jitter, but ensure minimum size
					if (this.size[1] < estimatedHeight || this.size[1] > estimatedHeight + 100) {
						this.setSize([Math.max(currentWidth, 400), estimatedHeight]);
					}
                    
                    // Style the DOM element (input) if it exists
                    // We need to force it to be a textarea and set height
                    setTimeout(() => {
                        if (widget.inputEl) {
                            widget.inputEl.readOnly = true;
                            widget.inputEl.style.backgroundColor = "#222";
                            widget.inputEl.style.color = "#00ff00"; // Green text for monitor style
                            widget.inputEl.style.fontFamily = "Consolas, monospace";
                            widget.inputEl.style.fontSize = "12px";
                            widget.inputEl.style.lineHeight = "1.5";
                            widget.inputEl.style.padding = "10px";
                            widget.inputEl.style.border = "1px solid #444";
                            widget.inputEl.style.borderRadius = "4px";
                            
                            // FORCE HEIGHT to fill node
                            // Leave some space for header (approx 40-60px)
                            const widgetHeight = this.size[1] - 60;
                            widget.inputEl.style.height = `${widgetHeight}px`;
                            widget.inputEl.style.maxHeight = `${widgetHeight}px`;
                            
                            widget.inputEl.scrollTop = widget.inputEl.scrollHeight; // Auto scroll to bottom
                        }
                    }, 50);

                    // Force redraw
					this.setDirtyCanvas(true, true);
				}
			};
            
            // Handle manual resize to update widget height
            const onResize = nodeType.prototype.onResize;
            nodeType.prototype.onResize = function(size) {
                onResize?.apply(this, arguments);
                const widget = this.widgets?.find((w) => w.name === "display_text");
                if (widget && widget.inputEl) {
                     // Sync widget height with new node height
                     const widgetHeight = size[1] - 60;
                     widget.inputEl.style.height = `${widgetHeight}px`;
                     widget.inputEl.style.maxHeight = `${widgetHeight}px`;
                }
            };
            
            // Handle creation
            const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function() {
                onNodeCreated?.apply(this, arguments);
            };
		}
	},
});
